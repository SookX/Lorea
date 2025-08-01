from transformers import AutoTokenizer
import torch

class Tokenizer:
    def __init__(self):
        super().__init__()
        self.tokenizer = self.create_tokenizer()

    @staticmethod
    def create_tokenizer():
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        tokenizer.add_special_tokens(
            {'bos_token': '<|bos|>', 
            'eos_token': '<|eos|>',
            'pad_token': '<|pad|>',
            'unk_token': '<|unk|>'})
        
        return tokenizer
    
    def encode(self, text):
        text = text.lower()

        tokenized_text = self.tokenizer(
            text,
            add_special_tokens=False 
        )['input_ids']
    
        tokenized_text = [self.tokenizer.bos_token_id] + tokenized_text + [self.tokenizer.eos_token_id]
        return torch.tensor(tokenized_text)

    def decode(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

if __name__ == "__main__":
    tokenizer = Tokenizer()
    text = "Hello"
    print(tokenizer.decode(tokenizer.encode(text)))