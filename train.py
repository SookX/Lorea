import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler         # ✅ NEW API (replaces torch.cuda.amp)
from tqdm import tqdm
from dataclasses import dataclass
from jiwer import cer, wer
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


@dataclass 
class TrainingConfig: 
    epochs: int 
    batch_size: int 
    learning_rate: float


class Trainer:
    def __init__(self, student_model, optimizer, scheduler, tokenizer, config: TrainingConfig):

        self.student_model = student_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.config = config

        self.scaler = GradScaler("cuda")

        self.teacher_processor = AutoProcessor.from_pretrained("openai/whisper-medium")
        self.teacher_model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-medium")

        self.teacher_conv1 = self.teacher_model.model.encoder.conv1
        self.teacher_conv2 = self.teacher_model.model.encoder.conv2

        for p in self.teacher_conv1.parameters(): p.requires_grad = False
        for p in self.teacher_conv2.parameters(): p.requires_grad = False

        self.train_losses = []
        self.val_losses = []
        self.best_wer = float("inf")

    def masked_mse_loss(self, student_feat, teacher_feat, seq_lens):
        B, C, T = student_feat.shape
        mask = torch.zeros((B, 1, T), dtype=torch.bool, device=student_feat.device)
        for i, l in enumerate(seq_lens):
            mask[i, :, :l] = 1
        loss = (student_feat - teacher_feat) ** 2
        loss = loss.masked_fill(~mask, 0)
        return loss.sum() / mask.sum()

    def forward(self, train_dataloader, val_dataloader=None, device="cuda", accum_steps=4):

        self.student_model.to(device)
        self.teacher_model.to(device)

        for epoch in range(self.config.epochs):
            tqdm.write(f"Epoch {epoch+1}/{self.config.epochs}")
            self.student_model.train()

            running_loss = 0.0

            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False)):

                with torch.no_grad():
                    t1 = F.gelu(self.teacher_conv1(batch["input_values"].to(device)))
                    t2 = F.gelu(self.teacher_conv2(t1))

                with autocast(device_type="cuda", dtype=torch.float16):
                    student_logits, output_lengths, s1, s2 = self.student_model(
                        batch["input_values"].to(device),
                        batch["seq_lens"].to(device)
                    )

                    log_probs = torch.log_softmax(student_logits, dim=-1).transpose(0,1)

                    distill_loss1 = self.masked_mse_loss(s1, t1.detach(), batch["seq_lens"])
                    distill_loss2 = self.masked_mse_loss(s2, t2.detach(), (batch["seq_lens"] // 2))

                    ctc_loss = nn.functional.ctc_loss(
                        log_probs,
                        targets=batch["labels"].to(device),
                        input_lengths=output_lengths,
                        target_lengths=batch["target_lengths"],
                        blank=self.tokenizer.pad_token_id,
                        reduction="mean"
                    )

                    loss = 3 * ctc_loss + 0.5 * distill_loss1 + 1.0 * distill_loss2
                    loss = loss / accum_steps

                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)

                running_loss += loss.item() * accum_steps

                if (step + 1) % accum_steps == 0:
                    self.scaler.step(self.optimizer)   
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()

                if (step + 1) % 200 == 0:
                    tqdm.write(f"[Epoch {epoch+1}, Step {step+1}] Avg Loss: {running_loss/200:.4f}")
                    running_loss = 0.0

                self.train_losses.append(loss.item() * accum_steps)

            if val_dataloader:
                val_loss, val_cer, val_wer = self.evaluate(val_dataloader, device)
                tqdm.write(f"Validation Loss: {val_loss:.4f}, CER: {val_cer:.4f}, WER: {val_wer:.4f}")

                if val_wer < self.best_wer:
                    self.best_wer = val_wer
                    torch.save(self.student_model.state_dict(), "kd-best_asr_student_model-460.pt")
                    tqdm.write(f"✅ Saved new best student model (WER={val_wer:.4f})")

    def evaluate(self, dataloader, device="cuda"):
        self.student_model.eval()

        total_cer = 0
        total_wer = 0
        total_samples = 0

        with torch.inference_mode(), autocast(device_type="cuda", dtype=torch.float16):
            for batch in tqdm(dataloader, desc="Validating", leave=False):

                logits, output_lengths, _, _ = self.student_model(
                    batch["input_values"].to(device),
                    batch["seq_lens"].to(device),
                )

                log_probs = torch.log_softmax(logits, dim=-1)
                pred_ids = torch.argmax(log_probs, dim=-1)

                for i in range(pred_ids.size(0)):
                    target_len = batch["target_lengths"][i]
                    target_seq = batch["labels"][i][:target_len]

                    pred_str = self.tokenizer.decode(pred_ids[i].tolist(), skip_special_tokens=False)
                    target_str = self.tokenizer.decode(target_seq.tolist(), skip_special_tokens=False)

                    total_cer += cer(target_str, pred_str)
                    total_wer += wer(target_str, pred_str)
                    total_samples += 1

        return 0.0, total_cer / total_samples, total_wer / total_samples
