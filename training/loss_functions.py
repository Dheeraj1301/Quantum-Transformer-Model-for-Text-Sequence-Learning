import torch.nn as nn

def get_loss_function():
    return nn.CrossEntropyLoss(ignore_index=5)  # Ignore <PAD> token (index 5)
