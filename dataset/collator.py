import torch

class Collator:
    def __init__(self, tokenizer):
        self.pad_token = tokenizer.special_tokens["<PAD>"]

    def __call__(self, batch):
        src_ids = [i["waveform"].squeeze(0) for i in batch] 
        tgt_ids = [i["transcript_ids"] for i in batch]


        src_padded = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=0.0)
        tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=self.pad_token)
        
        input_tgt = tgt_padded[:, :-1].clone()
        output_tgt = tgt_padded[:, 1:].clone()

        input_tgt_mask = (input_tgt != self.pad_token)
        output_tgt[output_tgt == self.pad_token] = -100

        return {
            "src_input_ids": src_padded.unsqueeze(1),
            "tgt_input_ids": input_tgt,
            "tgt_pad_mask": input_tgt_mask,
            "tgt_outputs": output_tgt,
            "transcript": [i["transcript"] for i in batch]
        }
    