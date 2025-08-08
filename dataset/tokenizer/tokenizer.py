import regex as re
import json
from tqdm import tqdm

class GPT4Tokenizer:
    def __init__(self):
        self.special_tokens = {}
        self.vocab_size = None
        self.merges = {}
        self.re = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""



    def save(self, path: str):
        merges_str_keys = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}

        data = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "merges": merges_str_keys
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.vocab_size = data["vocab_size"]
        merges = {}
        for k, v in data["merges"].items():
            pair = tuple(int(x) for x in k.split(","))
            merges[pair] = v
        self.merges = merges
        for k, v in data["special_tokens"].items():
            self.special_tokens[k] = int(v)



    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]): 
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def fit(self, corpus, vocab_size = 260):
        
        self.vocab_size = vocab_size

        self.special_tokens = {
            "<SOS>": vocab_size - 4,
            "<EOS>": vocab_size - 3,
            "<PAD>": vocab_size - 2,
            "<UNK>": vocab_size - 1
        }

        preprocess_corpus = re.findall(self.re, corpus)
        tokenized = [list(x.encode("utf-8")) for x in preprocess_corpus]  
        
        num_merges = self.vocab_size - 260

        for i in tqdm(range(num_merges)):
            global_stats = {}
            for bite in tokenized:
                stats = self.get_stats(bite)
                for pair, count in stats.items():
                    global_stats[pair] = global_stats.get(pair, 0) + count  
            
            if not global_stats:
                break

            pair = max(global_stats, key=global_stats.get)
            idx = 256 + i
            # print(f"Merging {pair} into a new token {idx}")

            for j in range(len(tokenized)):
                tokenized[j] = self.merge(tokenized[j], pair, idx)
            
            self.merges[pair] = idx  

        


    def encode(self, text):
        tokens = list(str(text).encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break 
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        tokens.insert(0, self.special_tokens["<SOS>"])
        tokens.append(self.special_tokens["<EOS>"])
        return tokens
        

    def decode(self, ids, skip_special = True):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        if not skip_special:
            for token, idx in self.special_tokens.items():
                vocab[idx] = token.encode("utf-8")

        tokens_bytes = b"".join(vocab.get(idx, b"") for idx in ids)
        try:
            return tokens_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return tokens_bytes.decode("utf-8", errors="replace")

if __name__ == "__main__":
    #tok = GPT4Tokenizer()
    #with open("./corpus.txt", "r") as f:
    #    corpus = f.read()

    #corpus = """Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the current vocabulary).
    #            When adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix of the model so that its embedding matrix matches the tokenizer."""
    #tok.fit(corpus, vocab_size=15000)
    #print(tok.decode(tok.encode("Hello World!"), False))
    #tok.save("tokenizer.json")
    #
#
    tok1 = GPT4Tokenizer()
    tok1.load("tokenizer.json")
