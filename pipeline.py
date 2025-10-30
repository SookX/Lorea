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

if __name__ == "__main__":
    warnings.filterwarnings(
    "ignore",
    message=".*torchaudio.load_with_torchcodec.*"
)

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")

    cfg_dict = read_config_("./config.yaml")
    cfg = TrainingConfig(**cfg_dict["training"])

    BATCH_SIZE = int(cfg_dict["training"]['batch_size'])
    LEARNING_RATE = float(cfg_dict["training"]["learning_rate"])
    EPOCHS = int(cfg_dict["training"]["epochs"])
    
    train_dataset=  LibriSpeechDataset("./dataset", include_splits=["train-clean-100", "train-clean-360"], tokenizer=tokenizer)
    val_dataset = LibriSpeechDataset("./dataset", include_splits=["dev-clean"], tokenizer=tokenizer, train_split=False)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    student_model = Lorea(288, tokenizer.vocab_size).to(device)

    teacher_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large-960h-ft")
    teacher_model = Wav2Vec2ConformerForCTC.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large-960h-ft").to(device)

    #output_dir = "teacher_logits_per_sample"
    #os.makedirs(output_dir, exist_ok=True)

    #for idx, sample in enumerate(tqdm(train_dataloader)):
    #    audio = sample["raw_audios"]
    #    inputs = teacher_processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).to("cuda")
    #    inputs = {k: v.to(device) for k, v in inputs.items()}
    #    with torch.no_grad():
    #        logits = teacher_model(**inputs).logits  # shape: [1, T, V]
    #    torch.save(logits.cpu(), os.path.join(output_dir, f"{idx:05d}.pt"))
    #
    #_quartznet5x5_config = [
    #    {'filters': 256, 'repeat': 1, 'kernel': 33, 'stride': 2, 'dilation': 1, 'dropout': 0.2, 'residual': False, 'separable': True},
#
    #    {'filters': 256, 'repeat': 5, 'kernel': 33, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': True, 'separable': True},
#
    #    {'filters': 256, 'repeat': 5, 'kernel': 39, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': True, 'separable': True},
#
    #    {'filters': 512, 'repeat': 5, 'kernel': 51, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': True, 'separable': True},
#
    #    {'filters': 512, 'repeat': 5, 'kernel': 63, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': True, 'separable': True},
#
    #    {'filters': 512, 'repeat': 5, 'kernel': 75, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': True, 'separable': True},
#
    #    {'filters': 512, 'repeat': 1, 'kernel': 87, 'stride': 1, 'dilation': 2, 'dropout': 0.2, 'residual': False, 'separable': True},
#
    #    {'filters': 1024, 'repeat': 1, 'kernel': 1, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': False, 'separable': False}
    #]
#
    ## Instantiate model
    #model = QuartzNet(model_config=_quartznet5x5_config, feat_in=80, vocab_size=tokenizer.vocab_size)
    pytorch_total_params = sum(p.numel() for p in student_model.parameters())
    print(f"Total model parameters: {pytorch_total_params}")

    num_training_steps = (len(train_dataset) / BATCH_SIZE) * EPOCHS
    optimizer = torch.optim.AdamW(student_model.parameters(), LEARNING_RATE)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=1500, 
                                            num_training_steps=EPOCHS * len(train_dataset))
    
    
    trainer = Trainer(student_model, teacher_model, teacher_processor, optimizer, scheduler, tokenizer, cfg)
    #trainer.evaluate(val_dataloader)
    trainer.forward(train_dataloader, val_dataloader, device)
            
