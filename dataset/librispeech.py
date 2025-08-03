from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import Dataset
from dataset.tokenizer.tokenizer import GPT4Tokenizer
import torch
from tqdm import tqdm


def dataset_to_corpus(dataset: Dataset, output_path="./tokenizer/corpus.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(len(dataset))):
            line = dataset[i]["transcript"].strip()
            if line:
                f.write(line + "\n")

class LibriSpeechReformated(Dataset):
    def __init__(self, path, url = 'train-clean-100', download = True):
        super().__init__()
        self.dataset = LIBRISPEECH(path, url=url, download=download) 
        self.tokenizer = GPT4Tokenizer()
        self.tokenizer.load("./dataset/tokenizer/tokenizer.json")

    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        waveform, _, transcript, _, _, _ = self.dataset[index]
        transcript_ids = self.tokenizer.encode(transcript.lower())
        return {
            "waveform": waveform.clone().detach(),
            "transcript": transcript.lower(),
            "transcript_ids": torch.tensor(transcript_ids, dtype=torch.long)
        }



    
# Dataset downloader
#train_dataset = LIBRISPEECH("./data", url='train-clean-100', download=False) 
#val_dataset = LIBRISPEECH("./data", url="dev-clean", download=False)

if __name__ == "__main__":
    train_dataset = LibriSpeechReformated("./data")
    #print(train_dataset[0])
    #dataset_to_corpus(train_dataset)
   #collate_fn = Collator(train_dataset.tokenizer)
#
    #data_loader = DataLoader(train_dataset, 1, True, collate_fn=collate_fn)
    #for sample in data_loader:
    #    print(sample)
    #    break