import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import time
from torch.profiler import profile, record_function, ProfilerActivity
from sklearn.metrics import accuracy_score
import jiwer
import random

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def train_step(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    epochs: int,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device="cpu"
) -> None:
    """
    Trains a PyTorch model and evaluates on validation set after each epoch.

    Args:
        model: PyTorch model.
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for validation data.
        epochs: Number of training epochs.
        loss_fn: Loss function (e.g., CrossEntropyLoss).
        optimizer: Optimizer (e.g., Adam).
        device: 'cuda' or 'cpu'.
    """

    logging.info("Initializing model training...\n")

    model.train()
    torch.autograd.set_detect_anomaly(True)

    for epoch in tqdm(range(epochs), desc="Epochs"):
        start_time = time.time()
        total_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False)

        for i, batch in enumerate(progress_bar):
            src_input_ids = batch["src_input_ids"].to(device)
            tgt_input_ids = batch["tgt_input_ids"].to(device)
            tgt_pad_mask = batch["tgt_pad_mask"].to(device)
            tgt_outputs = batch["tgt_outputs"].to(device)

            with torch.amp.autocast(device.type):

                logits = model(src_input_ids, tgt_input_ids, tgt_pad_mask)

                logits=logits.flatten(0, 1)
                tgt_outputs=tgt_outputs.flatten()

                loss = loss_fn(logits, tgt_outputs)
                preds = logits.argmax(dim=-1).cpu().numpy()
                targets = tgt_outputs.cpu().numpy()
                mask = targets != -100
                filtered_preds = preds[mask]
                filtered_targets = targets[mask]

                accuracy = accuracy_score(filtered_targets, filtered_preds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)
            progress_bar.set_postfix(loss=loss.item(), accuracy=f"{accuracy*100:.2f}%")

            if i % 32 == 0:
                model.eval()
                with torch.inference_mode():
                    val_batch = next(iter(val_dataloader))

                    val_src_input_ids = val_batch["src_input_ids"].to(device)
                    val_tgt_input_ids = val_batch["tgt_input_ids"].to(device)
                    val_tgt_pad_mask = val_batch["tgt_pad_mask"].to(device)
                    val_tgt_outputs = val_batch["tgt_outputs"].to(device)

                    with torch.amp.autocast(device.type):
                        val_logits = model(val_src_input_ids, val_tgt_input_ids, val_tgt_pad_mask)

                        val_logits_flat = val_logits.flatten(0, 1)
                        val_targets_flat = val_tgt_outputs.flatten()

                        val_loss = loss_fn(val_logits_flat, val_targets_flat)

                    print(f"[VAL][Step {i}] Loss: {val_loss.item():.4f}")
                    val_logits = val_logits.argmax(dim=-1) 

                    pred_ids = val_logits[0].tolist()
                    label_ids = val_tgt_outputs[0].tolist()

                    tokenizer = train_dataloader.dataset.tokenizer

                    pred_ids = [id for id in pred_ids if id != tokenizer.special_tokens["<PAD>"]]
                    label_ids = [id for id in label_ids if id != tokenizer.special_tokens["<PAD>"] and id != -100]
                    
                    pred_text = tokenizer.decode(pred_ids)
                    
                    label_text = tokenizer.decode(label_ids)
                    wer = jiwer.wer(label_text, pred_text)
                    cer = jiwer.cer(label_text, pred_text)

                    print("\n[Prediction vs Ground Truth]")
                    print(f"\033[94mPRED:\033[0m {pred_text}")
                    print(f"\033[92mREAL:\033[0m {label_text}")
                    print(f"\033[91mWER: \033[0m{wer:.3f}\n")
                    print("\033[93mCER:\033[0m  {:.4f}".format(cer))

                    

        model.train()
        elapsed = time.time() - start_time
        avg_train_loss  = total_loss / len(train_dataloader)
        print(
        f"Epoch {epoch+1}/{epochs} - "
        f"Train Loss: {avg_train_loss:.4f} - "
        f"Time: {elapsed:.2f}s"
            )   