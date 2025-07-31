import re
from collections import Counter

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

class Vocab:
    def __init__(self, min_freq=2, max_size=10000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.token2idx = {}
        self.idx2token = {}

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        # Start with special tokens
        most_common = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + [
            word for word, freq in counter.items()
            if freq >= self.min_freq
        ][:self.max_size - 4]

        self.token2idx = {token: idx for idx, token in enumerate(most_common)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

    def tokenize(self, text):
        # Basic word-level tokenizer
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip().split()

    def numericalize(self, text):
        tokens = self.tokenize(text)
        return [self.token2idx.get(token, self.token2idx[UNK_TOKEN]) for token in tokens]

    def denumericalize(self, ids):
        return [self.idx2token.get(idx, UNK_TOKEN) for idx in ids]

    def __len__(self):
        return len(self.token2idx)
