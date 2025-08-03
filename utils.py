import torch
import yaml
import logging
from tqdm import tqdm
import time
import os
import json

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    
    return "cpu"

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_number_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def data_split(dataset):
    total_len = len(dataset)
    train_len = int(total_len * 0.8)
    val_len = total_len - train_len

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])
    return train_dataset, val_dataset


def load_pretrained(model, optimizer, scheduler, checkpoint_path):

    device = next(model.parameters()).device
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint