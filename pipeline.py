import os
import warnings
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from dataset.dataset import LibriSpeechDataset
from dataset.collate import collate_fn
from transformers import Wav2Vec2CTCTokenizer, get_cosine_schedule_with_warmup
from model.lorea import Lorea
from utils import read_config_
from train import Trainer, TrainingConfig

from torch.nn.parallel import DistributedDataParallel as DDP

warnings.filterwarnings("ignore", message=".*torchaudio.load_with_torchcodec.*")


def main():

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1

    if use_ddp:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0


    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")
    cfg_dict = read_config_("./config.yaml")
    cfg = TrainingConfig(**cfg_dict["training"])

    BATCH_SIZE = int(cfg_dict["training"]["batch_size"])
    LR = float(cfg_dict["training"]["learning_rate"])
    EPOCHS = int(cfg_dict["training"]["epochs"])


    train_dataset = LibriSpeechDataset(
        "./dataset/datasets/librispeech/LibriSpeech",
        include_splits=["train-clean-100", "train-clean-360"],
        tokenizer=tokenizer
    )

    val_dataset = LibriSpeechDataset(
        "./dataset/datasets/librispeech/LibriSpeech",
        include_splits=["test-clean"],
        tokenizer=tokenizer,
        train_split=False,
    )


    train_sampler = DistributedSampler(train_dataset) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if (use_ddp and val_dataset) else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    student_model = Lorea(288, tokenizer.vocab_size).to(device)

    if use_ddp:
        student_model = DDP(student_model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LR)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1500,
        num_training_steps=max(1, EPOCHS * len(train_dataset)),
    )

    if local_rank == 0:
        print(f"Total model parameters: {sum(p.numel() for p in student_model.parameters()):,}")
    trainer = Trainer(student_model, optimizer, scheduler, tokenizer, cfg)

    try:
        trainer.forward(train_dataloader, val_dataloader, device)
    finally:
        if use_ddp and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
