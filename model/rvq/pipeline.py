import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchaudio.datasets import SPEECHCOMMANDS
from rvq import RVQ_VAE
from train import train_step
import os
from tqdm import tqdm

class SpeechCommandsModified(Dataset):
    def __init__(self, root = "../../dataset/data"):
        os.makedirs(root, exist_ok=True)
        self.dataset = SPEECHCOMMANDS(root=root, download=False)
        self.labels = [
            "backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
            "follow", "forward", "four", "go", "happy", "house", "learn", "left",
            "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila",
            "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero"
        ]

        self.label_to_index = {label: i for i, label in enumerate(self.labels)}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        wf, sr, label, _, _ = self.dataset[index]
        label = self.label_to_index[label]
        return wf, sr, label

    

    
def collate_fn(batch):
    waveforms = [item[0] for item in batch]
    labels = [item[2] for item in batch]
    lengths = [w.size(1) for w in waveforms]
    max_len = max(lengths)
    padded_waveforms = []
    for w in waveforms:
        pad_len = max_len - w.size(1)
        padded_waveforms.append(torch.nn.functional.pad(w, (0, pad_len)))
    waveforms = torch.stack(padded_waveforms)
    labels = torch.tensor(labels, dtype=torch.long)
    return waveforms, labels

if __name__ == "__main__":
    device = "cuda"
    batch_size = 4
    dataset = SpeechCommandsModified()



    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
    train_set, 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=collate_fn, 
    num_workers=4,  # increase to leverage multiprocessing for data loading
    pin_memory=True,  # speeds up transfer of data to GPU
    drop_last=True   # drop last incomplete batch (usually good for training)
)

    val_dataloader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=4, 
        pin_memory=True,
        drop_last=False  # usually want to evaluate on all data
    )


    model = RVQ_VAE(latent_dim=256).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), 3e-4)

    train_step(model, train_dataloader, val_dataloader, 50, loss_fn, optimizer, device)
