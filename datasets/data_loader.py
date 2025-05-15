import os
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets.tokenizer import GateTokenizer

class QuantumCircuitDataset(Dataset):
    def __init__(self, data_dir='data/raw'):
        self.data_dir = data_dir
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
        self.tokenizer = GateTokenizer()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Error reading {file_path}: {str(e)}")

        if not ("moments" in json_data or "circuit" in json_data):
            raise ValueError(f"Missing 'moments' or 'circuit' key in {file_path}")

        # Tokenize circuit
        tokens = self.tokenizer.encode(json_data)
        token_tensor = torch.tensor(tokens, dtype=torch.long)

        # Labels
        label = json_data.get("label")

        if label is None:
            label_tensor = torch.zeros_like(token_tensor)  # fallback dummy labels
        elif isinstance(label, int):
            # Single label for full sequence classification
            label_tensor = torch.tensor(label, dtype=torch.long)
            return token_tensor, label_tensor
        elif isinstance(label, list):
            if len(label) != len(tokens):
                raise ValueError(f"Label length {len(label)} doesn't match token length {len(tokens)} in {file_path}")
            label_tensor = torch.tensor(label, dtype=torch.long)
        else:
            raise ValueError(f"Invalid label type in {file_path}")

        return token_tensor, label_tensor


def collate_fn(batch):
    """
    Pads a batch of sequences and their corresponding labels.
    Labels are padded with -100 to be ignored in loss.
    """
    token_seqs = [item[0] for item in batch]
    label_seqs = [item[1] for item in batch]

    # Pad tokens with 0 (assumed to be <PAD> token id)
    token_padded = pad_sequence(token_seqs, batch_first=True, padding_value=0)

    # If labels are scalars (classification), return as is
    if all(len(label.shape) == 0 for label in label_seqs):
        label_tensor = torch.stack(label_seqs)
    else:
        label_tensor = pad_sequence(label_seqs, batch_first=True, padding_value=-100)

    return token_padded, label_tensor
