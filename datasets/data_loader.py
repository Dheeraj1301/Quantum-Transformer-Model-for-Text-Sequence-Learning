import torch
from torch.utils.data import Dataset
import os
import json
from .tokenizer import GateTokenizer

class QuantumCircuitDataset(Dataset):
    def __init__(self, data_dir='data/raw'):
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        self.data_dir = data_dir
        self.tokenizer = GateTokenizer()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        with open(path, 'r') as f:
            json_data = f.read()
        tokens = self.tokenizer.encode(json_data)
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(tokens, dtype=torch.long)
