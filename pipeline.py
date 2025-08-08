import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from colorama import Fore, Style, init

from dataset.librispeech import LibriSpeechReformated
from dataset.collator import Collator
from model.lorea import Lorea

from train import train_step
from transformers import get_scheduler
from utils import load_config, get_device, get_number_of_parameters, load_pretrained


init(autoreset=True)  

def main():
    DEVICE = torch.device(get_device())
    print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} Training will run on: {Fore.CYAN}{DEVICE.type.upper()}")

    config = load_config()
    MODEL_NAME = str(config["model"]["model_name"])

    EPOCHS = int(config["training"]["epochs"])
    BATCH_SIZE = int(config["training"]["batch_size"])
    LEARNING_RATE = float(config["training"]["learning_rate"])

    SAVE_CHECKPOINT_EVERY = int(config["logging"]["save_checkpoint_every"])
    CHECKPOINT_DIR = str(config["logging"]["checkpoint_dir"])

    train_dataset = LibriSpeechReformated("./dataset/data")
    val_dataset = LibriSpeechReformated("./dataset/data", "dev-clean")

    collate_fn = Collator(train_dataset.tokenizer)

    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, True, collate_fn=collate_fn, num_workers=4)
    val_dataloader = DataLoader(val_dataset, BATCH_SIZE, False, collate_fn=collate_fn)
    print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} {Fore.YELLOW}All data loaders have been successfully initialized.{Style.RESET_ALL}")

    model = Lorea().to(DEVICE)
    num_params = get_number_of_parameters(model)
    print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} Model {Fore.CYAN}Lorea{Style.RESET_ALL} loaded with "
          f"{Fore.MAGENTA}{num_params:,}{Style.RESET_ALL} trainable parameters.\n")

    optimizer = torch.optim.AdamW(model.parameters(), LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    num_training_steps = (len(train_dataset) / BATCH_SIZE) * EPOCHS
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=1500, num_training_steps=num_training_steps)

    checkpoint = load_pretrained(model, optimizer, scheduler, checkpoint_path="./checkpoints/lorea-asr-3/lorea-asr-3_epoch_15.pt")
    

    train_step(model, train_dataloader, val_dataloader, EPOCHS, loss_fn, optimizer, scheduler, DEVICE, CHECKPOINT_DIR, MODEL_NAME, start_epoch=checkpoint["epoch"])

if __name__ == "__main__":
    main()
