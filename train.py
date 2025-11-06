import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import dataclass
from jiwer import cer, wer
import os
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch.distributed as dist

@dataclass 
class TrainingConfig: 
    epochs: int 
    batch_size: int 
    learning_rate: float

class Trainer:
    def __init__(self, 
                 student_model,
                 optimizer,
                 scheduler,
                 tokenizer,
        
                 config: TrainingConfig):
        
        self.student_model = student_model
        #self.teacher_model = teacher_model
        #self.teacher_processor = teacher_processor

        


        #for param in self.teacher_model.parameters():
        #    param.requires_grad = False


        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.config = config
        self.teacher_processor = AutoProcessor.from_pretrained("openai/whisper-medium")
        self.teacher_model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-medium")

        self.teacher_conv1 = self.teacher_model.model.encoder.conv1
        self.teacher_conv2 = self.teacher_model.model.encoder.conv2

        for param in self.teacher_conv1.parameters():
            param.requires_grad = False
        for param in self.teacher_conv2.parameters():
            param.requires_grad = False

        self.train_losses = []
        self.val_losses = []
        self.best_wer = float("inf")
    
    def compute_kd_loss(self, student_logits, teacher_logits_list, output_lengths, temperature=1.0, device="cuda"):
        """
        Compute knowledge distillation loss (KL divergence) between student and teacher
        for variable-length sequences with per-sample interpolation.

        Args:
            student_logits: Tensor[B, T_s, C] - student logits
            teacher_logits_list: List of tensors, each [T_t_i, C] - teacher logits per sample
            output_lengths: Tensor[B] - valid lengths of student sequences
            temperature: float - KD temperature
            device: str or torch.device

        Returns:
            kd_loss: scalar tensor
        """
        kd_loss = 0.0
        B = student_logits.size(0)
        T = temperature

        for i in range(B):
            s_len = int(output_lengths[i].item())
            t_logits = teacher_logits_list[i].to(device)          
            s_logits = student_logits[i, :s_len].to(device)     

            s_feat = s_logits.transpose(0, 1).unsqueeze(0)       # (1, C, T_s_i)
            s_interp = F.interpolate(
                s_feat, size=t_logits.size(0), mode="linear", align_corners=False
            )
            s_interp = s_interp.squeeze(0).transpose(0, 1)       # (T_t_i, C)

            kd_loss += F.kl_div(
                F.log_softmax(s_interp / T, dim=-1),
                F.softmax(t_logits / T, dim=-1),
                reduction="batchmean"
            )

        kd_loss /= B
        kd_loss *= T ** 2

        return kd_loss
    
    def masked_mse_loss(self, student_feat, teacher_feat, seq_lens):
        B, C, T = student_feat.shape
        mask = torch.zeros((B, 1, T), dtype=torch.bool, device=student_feat.device)
        for i, l in enumerate(seq_lens):
            mask[i, :, :l] = 1
        loss = (student_feat - teacher_feat) ** 2
        loss = loss.masked_fill(~mask, 0)
        return loss.sum() / mask.sum()


    def forward(self, train_dataloader, val_dataloader=None, device="cpu", accum_steps=1):
        
        self.student_model.to(device)
        self.teacher_model.to(device)


        for epoch in range(self.config.epochs):
            tqdm.write(f"Epoch {epoch+1}/{self.config.epochs}")
            self.student_model.train()

            running_loss = 0.0  #
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False)):
                student_logits, output_lengths, s1, s2 = self.student_model(
                    batch["input_values"].to(device),
                    batch["seq_lens"].to(device)
                    
                )
                with torch.no_grad():
                    t1 = self.teacher_conv1(batch["input_values"].to(device))
                    t1 = F.gelu(t1)
                    t2 = self.teacher_conv2(t1)
                    t2 = F.gelu(t2)

                
                
 #               inputs = self.teacher_processor(batch["raw_audios"], sampling_rate=16000, return_tensors="pt", padding=True)
                #inputs = {k: v.to(device) for k, v in inputs.items()}
#
                #with torch.no_grad():
                #    teacher_logits = self.teacher_model(**inputs).logits


                
                log_probs = nn.functional.log_softmax(student_logits, dim=-1).transpose(0,1)

                distill_loss1 = self.masked_mse_loss(s1, t1.detach(), batch["seq_lens"])
                distill_loss2 = self.masked_mse_loss(s2, t2.detach(), (batch["seq_lens"] // 2))

                                          
                ctc_loss = nn.functional.ctc_loss(
                    log_probs=log_probs,
                    targets=batch["labels"].to(device),
                    input_lengths=output_lengths,
                    target_lengths=batch["target_lengths"],
                    blank=self.tokenizer.pad_token_id,
                    reduction="mean"
                )

                lambda1 = 0.5   
                lambda2 = 1.0   
                
                loss = 3 * ctc_loss + lambda1 * distill_loss1 + lambda2 * distill_loss2




                loss = loss / accum_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
                running_loss += loss.item() * accum_steps 

                if (step + 1) % accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()

                if (step + 1) % 200 == 0:
                    avg_loss = running_loss / 200
                    tqdm.write(f"[Epoch {epoch+1}, Step {step+1}] Avg Loss: {avg_loss:.4f}")
                    running_loss = 0.0

                self.train_losses.append(loss.item() * accum_steps)

            if val_dataloader:
                val_loss, val_cer, val_wer = self.evaluate(val_dataloader, device)
                self.val_losses.append(val_loss)
                tqdm.write(f"Validation Loss: {val_loss:.4f}, CER: {val_cer:.4f}, WER: {val_wer:.4f}")
                
                with open("validation_wer_log.txt", "a") as f: 
                    f.write(f"{val_wer:.4f}\n")

                if val_wer < self.best_wer:
                    self.best_wer = val_wer
                    torch.save(self.student_model.state_dict(), "kd-best_asr_student_model-460.pt")
                    tqdm.write(f"✅ Saved new best student_model (WER={val_wer:.4f})")

    def evaluate(self, dataloader, device="cpu"):
        is_ddp = dist.is_initialized() if "dist" in globals() else False
        rank = dist.get_rank() if is_ddp else 0
    
        self.student_model.eval().to(device)
        total_cer, total_wer = 0.0, 0.0
        num_examples = 0
    
        with torch.inference_mode():
            printed_example = False
            for batch in tqdm(dataloader, desc="Validating", leave=False):
                logits, output_lengths, _, _ = self.student_model(
                    batch["input_values"].to(device),
                    batch["seq_lens"].to(device),
                )
                log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
    
                pred_ids = torch.argmax(log_probs, dim=-1).transpose(0, 1)
                start_idx = 0
                for i, pred in enumerate(pred_ids):
                    target_len = batch["target_lengths"][i]
                    target_seq = batch["labels"][start_idx:start_idx + target_len]
                    start_idx += target_len
    
                    pred_str = self.tokenizer.decode(pred.tolist(), skip_special_tokens=True)
                    target_str = self.tokenizer.decode(target_seq.tolist(), skip_special_tokens=True)
    
                    if rank == 0 and not printed_example:
                        print(f"Target  : {target_str}")
                        print(f"Pred    : {pred_str}")
                        printed_example = True
    
                    total_cer += cer(target_str, pred_str)
                    total_wer += wer(target_str, pred_str)
                    num_examples += 1
    
        if is_ddp:
            total_cer_tensor = torch.tensor(total_cer, device=device)
            total_wer_tensor = torch.tensor(total_wer, device=device)
            num_examples_tensor = torch.tensor(num_examples, device=device)
    
            dist.all_reduce(total_cer_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_wer_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_examples_tensor, op=dist.ReduceOp.SUM)
    
            total_cer = total_cer_tensor.item()
            total_wer = total_wer_tensor.item()
            num_examples = num_examples_tensor.item()
    
        avg_cer = total_cer / num_examples if num_examples > 0 else float("inf")
        avg_wer = total_wer / num_examples if num_examples > 0 else float("inf")
        avg_loss = float("nan")  
    
        self.student_model.train()
        return avg_loss, avg_cer, avg_wer

