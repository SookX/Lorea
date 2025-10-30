import os
import torch
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ConformerForCTC
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from dataset.dataset import LibriSpeechDataset
from dataset.collate import collate_fn
from transformers import Wav2Vec2CTCTokenizer, get_cosine_schedule_with_warmup, Wav2Vec2Processor, Wav2Vec2ConformerForCTC
from model.lorea import Lorea
import warnings
from utils import read_config_
from train import Trainer, TrainingConfig
import torchaudio
from quartznet import QuartzNet

# Initialize teacher model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large-960h-ft")
teacher_model = Wav2Vec2ConformerForCTC.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large-960h-ft").to(device)
teacher_model.eval()

# Load dataset
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")

dataset = LibriSpeechDataset("./dataset", include_splits=["train-clean-100", "train-clean-360"], tokenizer=tokenizer)

output_dir = "./teacher_logits"
os.makedirs(output_dir, exist_ok=True)

# Precompute
for sample in tqdm(dataset, desc="Precomputing teacher logits"):
    uid = sample["uid"]
    audio = sample["raw_audio"].squeeze(0).numpy()

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = teacher_model(**inputs).logits.cpu()

    torch.save(logits, os.path.join(output_dir, f"{uid}.pt"))
