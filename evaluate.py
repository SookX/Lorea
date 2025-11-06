import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2CTCTokenizer
from jiwer import cer, wer
from tqdm import tqdm

from dataset.dataset import LibriSpeechDataset
from dataset.collate import collate_fn
from model.lorea import Lorea


MODEL_PATH = "./kd-best_asr_student_model-460.pt"
BATCH_SIZE = 32
DEVICE = "cuda"


def load_checkpoint(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    new_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        k = k.replace("conv_layer_norm", "conv_module.layer_norm")
        k = k.replace("conv_conv_block", "conv_block")
        new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    return model

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")


val_dataset = LibriSpeechDataset(
    "./dataset/datasets/librispeech/LibriSpeech",
    include_splits=["test-clean"],
    tokenizer=tokenizer,
    train_split=False,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
)
model = Lorea(288, tokenizer.vocab_size)
model = load_checkpoint(model, MODEL_PATH)
model.to(DEVICE)
model.eval()

def evaluate(model, tokenizer, dataloader, device="cuda"):
    model.to(device)
    total_cer, total_wer = 0.0, 0.0
    printed_example = False

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            logits, output_lengths, _, _ = model(
                batch["input_values"].to(device),
                batch["seq_lens"].to(device),
            )

            log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)  

            pred_ids = torch.argmax(log_probs, dim=-1).transpose(0, 1) 
            start_idx = 0
            for i, pred in enumerate(pred_ids):
                target_len = batch["target_lengths"][i]
                target_seq = batch["labels"][start_idx:start_idx + target_len]
                start_idx += target_len

                pred_str = tokenizer.decode(pred.tolist(), skip_special_tokens=False)
                target_str = tokenizer.decode(target_seq.tolist(), skip_special_tokens=False)

                if not printed_example:
                    print(f"Target  : {target_str}")
                    print(f"Pred    : {pred_str}")
                    printed_example = True

                total_cer += cer(target_str, pred_str)
                total_wer += wer(target_str, pred_str)

    avg_cer = total_cer / len(dataloader.dataset) if len(dataloader.dataset) > 0 else float("inf")
    avg_wer = total_wer / len(dataloader.dataset) if len(dataloader.dataset) > 0 else float("inf")

    return avg_cer, avg_wer


avg_cer, avg_wer = evaluate(model, tokenizer, val_dataloader)
print(f"Average CER: {avg_cer:.4f}")
print(f"Average WER: {avg_wer:.4f}")
