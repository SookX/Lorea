import torch
import torch.nn as nn
from tqdm import tqdm
from dataclasses import dataclass
from jiwer import cer, wer

@dataclass 
class TrainingConfig: 
    epochs: int 
    batch_size: int 
    learning_rate: float

class Trainer:
    def __init__(self, 
                 model,
                 optimizer,
                 scheduler,
                 tokenizer,
                 config: TrainingConfig):
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.config = config

        self.train_losses = []
        self.val_losses = []
        self.best_wer = float("inf")

    def forward(self, train_dataloader, val_dataloader=None, device="cpu", accum_steps=4):
        self.model.to(device)

        for epoch in range(self.config.epochs):
            tqdm.write(f"Epoch {epoch+1}/{self.config.epochs}")
            self.model.train()

            running_loss = 0.0  # for averaging loss
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False)):
                logits, output_lengths = self.model(
                    batch["input_values"].to(device),
                    batch["seq_lens"].to(device)
                    
                )
                #output_lengths = batch["seq_lens"] // 2
                log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0,1)
                
                loss = nn.functional.ctc_loss(
                    log_probs=log_probs,
                    targets=batch["labels"].to(device),
                    input_lengths=output_lengths,
                    target_lengths=batch["target_lengths"],
                    blank=self.tokenizer.pad_token_id,
                    reduction="mean"
                )

                # Scale loss for gradient accumulation
                loss = loss / accum_steps
                loss.backward()

                running_loss += loss.item() * accum_steps  # scale back for logging

                if (step + 1) % accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()

                # Log average loss every 200 steps
                if (step + 1) % 200 == 0:
                    avg_loss = running_loss / 200
                    tqdm.write(f"[Epoch {epoch+1}, Step {step+1}] Avg Loss: {avg_loss:.4f}")
                    running_loss = 0.0

                self.train_losses.append(loss.item() * accum_steps)

            # Validation
            if val_dataloader:
                val_loss, val_cer, val_wer = self.evaluate(val_dataloader, device)
                self.val_losses.append(val_loss)
                tqdm.write(f"Validation Loss: {val_loss:.4f}, CER: {val_cer:.4f}, WER: {val_wer:.4f}")

                if val_wer < self.best_wer:
                    self.best_wer = val_wer
                    torch.save(self.model.state_dict(), "best_asr_model-460.pt")
                    tqdm.write(f"✅ Saved new best model (WER={val_wer:.4f})")

    def evaluate(self, dataloader, device="cpu"):
        self.model.eval()
        total_loss, total_cer, total_wer = 0.0, 0.0, 0.0
        num_batches = 0

        with torch.inference_mode():
            printed_example = False
            for batch in tqdm(dataloader, desc="Validating", leave=False):
                logits, output_lengths = self.model(
                    batch["input_values"].to(device),
                    batch["seq_lens"].to(device),
                    
                )
                log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0,1)
                #output_lengths = batch["seq_lens"] // 2
                loss = nn.functional.ctc_loss(
                    log_probs=log_probs,
                    targets=batch["labels"].to(device),
                    input_lengths=output_lengths,
                    target_lengths=batch["target_lengths"],
                    blank=self.tokenizer.pad_token_id,
                    reduction="mean"
                )
                total_loss += loss.item()
                num_batches += 1

                # --- CER computation ---
                pred_ids = torch.argmax(log_probs, dim=-1).transpose(0,1)  # (B, T)
                start_idx = 0
                for i, pred in enumerate(pred_ids):
                    target_len = batch["target_lengths"][i]
                    target_seq = batch["labels"][start_idx:start_idx + target_len]
                    start_idx += target_len   
                    pred_str = self.tokenizer.decode(pred.tolist(), skip_special_tokens=False)
                    target_str = self.tokenizer.decode(target_seq.tolist(), skip_special_tokens=False)
                    if not printed_example:
                        print(f"Target  : {target_str}")
                        print(f"Pred    : {pred_str}")
                        printed_example = True
                    total_cer += cer(target_str, pred_str)
                    total_wer += wer(target_str, pred_str)

        avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
        avg_cer = total_cer / len(dataloader.dataset) if len(dataloader.dataset) > 0 else float("inf")
        avg_wer = total_wer/ len(dataloader.dataset) if len(dataloader.dataset) > 0 else float("inf")

        self.model.train()
        return avg_loss, avg_cer, avg_wer
