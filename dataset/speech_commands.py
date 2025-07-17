import os
from torch.utils.data import Dataset
from torchaudio.datasets import SPEECHCOMMANDS

class SpeechCommandsModified(Dataset):
    def __init__(self, root = "./data"):
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
    

if __name__ == "__main__":
    ds = SpeechCommandsModified()
    wf, sr, label = ds[15000]