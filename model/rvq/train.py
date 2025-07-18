import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import time
from torch.profiler import profile, record_function, ProfilerActivity

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
    model.to(device)

    for epoch in tqdm(range(epochs)):
        start_time = time.time()
        model.train()
        epoch_loss = 0.0
        all_preds, all_labels = [], []

        for wf, label in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            wf, label = wf.to(device), label.to(device)
            with torch.amp.autocast(device):
                logits, decoded, codebook_losses, comitment_losses = model(wf)
                rec_loss = torch.mean((wf-decoded)**2)
                λ_cls = 1.0       # main priority
                λ_rec = 0.1       # smaller weight for reconstruction
                λ_vq = 0.1        # small weight for codebook + commitment losses

                loss = λ_cls * loss_fn(logits, label) + λ_rec * rec_loss + λ_vq * (codebook_losses + comitment_losses)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

                epoch_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.detach().cpu().tolist())
                all_labels.extend(label.detach().cpu().tolist())

        train_acc = accuracy_score(all_labels, all_preds)
        train_loss = epoch_loss / len(train_dataloader)
        train_time = time.time() - start_time

        # ---- VALIDATION STEP ----
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for wf, label in tqdm(val_dataloader):
                wf, label = wf.to(device), label.to(device)

                logits, decoded, codebook_losses, comitment_losses = model(wf)
                rec_loss = torch.mean((wf-decoded)**2)
                λ_cls = 1.0       # main priority
                λ_rec = 0.1       # smaller weight for reconstruction
                λ_vq = 0.1        # small weight for codebook + commitment losses

                loss = λ_cls * loss_fn(logits, label) + λ_rec * rec_loss + λ_vq * (codebook_losses + comitment_losses)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.detach().cpu().tolist())
                val_labels.extend(label.detach().cpu().tolist())

        val_acc = accuracy_score(val_labels, val_preds)
        val_loss /= len(val_dataloader)

        # ---- LOGGING ----
        logging.info(
            f"Epoch [{epoch+1}/{epochs}] "
            f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% "
            f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% "
            f"| Time: {train_time:.2f}s"
        )
