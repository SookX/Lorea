import os
from tqdm.notebook import tqdm
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from transformers import Wav2Vec2CTCTokenizer

class LibriSpeechDataset(Dataset):
    """
    LibriSpeechDataset downloaded from OpenSLR: https://www.openslr.org/12
    Training splits: ["train-clean-100", "train-clean-360", "train-other-500"]
    Validation splits: ["dev-clean", "test-clean"]
    """

    def __init__(self, 
                 path_to_data_root, 
                 include_splits=["train-clean-100", "train-clean-360", "train-other-500"],
                 sampling_rate=16000,
                 num_audio_channels=1,
                 tokenizer=None):
        
        if isinstance(include_splits, str):
            include_splits = [include_splits]

        self.sampling_rate = sampling_rate
        self.num_audio_channels = num_audio_channels
        self.tokenizer = tokenizer

        ### GET PATH TO ALL AUDIO/TEXT FILES ###
        self.librispeech_data = []
        for split in include_splits:
            path_to_split = os.path.join(path_to_data_root, split)
            for speaker in os.listdir(path_to_split):
                path_to_speaker = os.path.join(path_to_split, speaker)
                for section in os.listdir(path_to_speaker):
                    path_to_section = os.path.join(path_to_speaker, section)
                    files = os.listdir(path_to_section)
                    transcript_file = [path for path in files if ".txt" in path][0]

                    with open(os.path.join(path_to_section, transcript_file), "r") as f:
                        transcripts = f.readlines()

                    for line in transcripts:
                        split_line = line.split()
                        audio_root = split_line[0]
                        audio_file = audio_root + ".flac"
                        full_path_to_audio_file = os.path.join(path_to_section, audio_file)
                        transcript = " ".join(split_line[1:]).strip()
                        self.librispeech_data.append((full_path_to_audio_file, transcript))
   
        self.audio2mels = T.MelSpectrogram(
            sample_rate=sampling_rate,
            n_mels=80
        )

        self.amp2db = T.AmplitudeToDB(top_db=80.0)
        
    def __len__(self):
        return len(self.librispeech_data)
    
    def __getitem__(self, idx):
        def ctc_decode(pred_ids):
            decoded = []
            prev = None
            for p in pred_ids:
                if p != prev and p != self.tokenizer.pad_token_id:
                    decoded.append(p)
                prev = p
            return self.tokenizer.decode(decoded, skip_special_tokens=True)

        path_to_audio, transcript = self.librispeech_data[idx]

        audio, orig_sr = torchaudio.load(path_to_audio, normalize=True)
        if orig_sr != self.sampling_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=self.sampling_rate)
        
        mel = self.audio2mels(audio)
        mel = self.amp2db(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        # Tokenize text
        tokenized_transcript = torch.tensor(self.tokenizer.encode(transcript))

        # Decode without skipping repeated tokens (for inspection)
        #decoded_transcript = self.tokenizer.decode(tokenized_transcript.tolist(), skip_special_tokens=True)

       # ctc_decoded = ctc_decode(tokenized_transcript.tolist())

       # # Print for debugging
       # print("Original    :", transcript)
       # print("Token IDs   :", tokenized_transcript.tolist())
       # print("Decoded     :", decoded_transcript)
       # print("CTC Decoded :", ctc_decoded)

        sample = {
            "input_values": mel[0].T,  # transpose for time-axis
            "labels": tokenized_transcript
        }
        return sample

# --- Sample usage ---
if __name__ == "__main__":
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")

    dataset = LibriSpeechDataset(
        path_to_data_root="./",
        include_splits="train-clean-100",
        tokenizer=tokenizer
    )

    sample = next(iter(dataset))

    plt.figure(figsize=(15,5))
    plt.imshow(sample["input_values"].T)
    plt.axis("off")
    plt.gca().invert_yaxis()
    plt.show()
