import os
import json
from torch.utils.data import Dataset
# In data_loader.py (near the top)
from datasets.tokenizer import GateTokenizer


class QuantumCircuitDataset(Dataset):
    def __init__(self, data_dir='data/raw'):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
        self.tokenizer = GateTokenizer()

    def __len__(self):
        return len(self.files)

    import torch

    def __getitem__(self, idx):
        with open(self.files[idx], 'r') as f:
            json_data = json.load(f)
        
        # Validate the input JSON structure before encoding
        if not ("moments" in json_data or "circuit" in json_data):
            raise ValueError(f"Missing 'moments' or 'circuit' in {self.files[idx]}")
        
        tokens = self.tokenizer.encode(json_data)  # presumably a list or array of tokens
        label = json_data.get('label', 0)
        
        # Convert to torch tensors
        tokens_tensor = torch.tensor(tokens, dtype=torch.float32)  # or int64 depending on your tokenizer output
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return tokens_tensor, label_tensor
