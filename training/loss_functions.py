import torch.nn as nn

def get_loss_function():
    # CrossEntropyLoss expects inputs: [N, C], targets: [N]
    return nn.CrossEntropyLoss()
