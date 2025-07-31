import torch
from torch.utils.data import Dataset
import pandas as pd
from utils.vocab import Vocab, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN

class RecipeDataset(Dataset):
    def __init__(self, file_path, input_vocab=None, target_vocab=None, max_len=50):
        self.data = pd.read_csv(file_path)
        self.max_len = max_len

        self.inputs = self.data['input'].tolist()
        self.targets = self.data['output'].tolist()

        self.input_vocab = input_vocab or Vocab()
        self.target_vocab = target_vocab or Vocab()

        self.input_vocab.build_vocab(self.inputs)
        self.target_vocab.build_vocab(self.targets)

    def __len__(self):
        return len(self.inputs)

    def pad_sequence(self, seq, pad_token_id):
        if len(seq) > self.max_len:
            return seq[:self.max_len]
        return seq + [pad_token_id] * (self.max_len - len(seq))

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]

        input_ids = self.input_vocab.numericalize(input_text)
        target_ids = self.target_vocab.numericalize(target_text)

        # Add special tokens
        input_ids = self.pad_sequence(input_ids, self.input_vocab.token2idx[PAD_TOKEN])
        target_ids = [self.target_vocab.token2idx[SOS_TOKEN]] + target_ids + [self.target_vocab.token2idx[EOS_TOKEN]]
        target_ids = self.pad_sequence(target_ids, self.target_vocab.token2idx[PAD_TOKEN])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long)
        }
