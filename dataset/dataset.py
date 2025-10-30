import os
from tqdm.notebook import tqdm
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import random
from transformers import Wav2Vec2CTCTokenizer
import torchaudio.transforms as AT

import torch

import torch
import torchaudio

class ConformerSpecAugment(torch.nn.Module):
    def __init__(self,
                 time_mask_param=40,
                 freq_mask_param=30,
                 num_time_masks=2,
                 num_freq_masks=2,
                 time_warp=False,
                 time_warp_param=80):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.time_warp = time_warp

        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        if time_warp:
            self.time_warp_tf = torchaudio.transforms.TimeStretch()

    def forward(self, spec):
        out = spec.clone()

        if self.time_warp:
            out = self.time_warp_tf(out)

        for _ in range(self.num_freq_masks):
            out = self.freq_mask(out)

        # Apply time masking
        for _ in range(self.num_time_masks):
            out = self.time_mask(out)

        return out



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
                 tokenizer=None,
                 train_split = True,
                 apply_spec_augment=True,
                 apply_audio_augment=True):
        
        if isinstance(include_splits, str):
            include_splits = [include_splits]

        self.sampling_rate = sampling_rate
        self.num_audio_channels = num_audio_channels
        self.tokenizer = tokenizer
        self.teacher_logits_dir = "./teacher_logits"
        self.train_split = train_split
        self.apply_spec_augment = apply_spec_augment
        self.apply_audio_augment= apply_audio_augment
        self.spec_augment = ConformerSpecAugment()

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
        self.time_mask_transform = torchaudio.transforms.TimeMasking(time_mask_param=40)
        self.freq_mask_transform = torchaudio.transforms.FrequencyMasking(freq_mask_param=30)
        self.time_warp_transform = torchaudio.transforms.TimeStretch()

        # simple audio augmentations
        # self.pitch_shift = AT.PitchShift(sample_rate=sampling_rate, n_steps=random.choice([-2, -1, 1, 2]))
        # self.time_stretch = AT.TimeStretch(n_freq=80)
        # self.reverb = AT.Vol(1.0)
        
    def __len__(self):
        return len(self.librispeech_data)
    

    
    def __getitem__(self, idx):
        path_to_audio, transcript = self.librispeech_data[idx]
        audio, orig_sr = torchaudio.load(path_to_audio, normalize=True)
        if orig_sr != self.sampling_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=self.sampling_rate)



        mel = self.audio2mels(audio)
        mel = self.amp2db(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        if self.train_split:
            mel = self.spec_augment(mel)

        tokenized_transcript = torch.tensor(self.tokenizer.encode(transcript))
        uid = os.path.splitext(os.path.basename(path_to_audio))[0]
        
        sample = {
            "input_values": mel[0].T,
            "raw_audio": audio,
            "labels": tokenized_transcript,
            "uid": uid
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
