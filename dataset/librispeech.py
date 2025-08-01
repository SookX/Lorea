from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer
import torch
import matplotlib.pyplot as plt

class LibriSpeechReformated(Dataset):
    def __init__(self, path, url = 'train-clean-100', download = False):
        super().__init__()
        self.dataset = LIBRISPEECH(path, url=url, download=download) 
        self.tokenizer = Tokenizer()
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        waveform, _, transcript, _, _, _ = self.dataset[index]
        transcript = self.tokenizer.encode(transcript)
        return {
            "waveform": torch.tensor(waveform),
            "transcript": transcript
        }

class Collator:
    def __init__(self, tokenizer):
        self.pad_token = tokenizer.tokenizer.pad_token_id

    def __call__(self, batch):
        src_ids = [i["waveform"].squeeze(0) for i in batch] 
        tgt_ids = [i["transcript"] for i in batch]

        print(tgt_ids)
        src_padded = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=0.0)
        tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=self.pad_token)
        
        input_tgt = tgt_padded[:, :-1].clone()
        output_tgt = tgt_padded[:, 1:].clone()

        input_tgt_mask = (input_tgt != self.pad_token)
        output_tgt[output_tgt == self.pad_token] = -100

        return {
            "src_input_ids": src_padded,
            "tgt_input_ids": input_tgt,
            "tgt_pad_mask": input_tgt_mask,
            "tgt_outputs": output_tgt,
        }
    
# Dataset downloader
#train_dataset = LIBRISPEECH("./data", url='train-clean-100', download=False) 
#val_dataset = LIBRISPEECH("./data", url="dev-clean", download=False)

if __name__ == "__main__":
    train_dataset = LibriSpeechReformated("./data")
    collate_fn = Collator(train_dataset.tokenizer)

    data_loader = DataLoader(train_dataset, 4, True, collate_fn=collate_fn)
    for sample in data_loader:
        print(sample)
        break