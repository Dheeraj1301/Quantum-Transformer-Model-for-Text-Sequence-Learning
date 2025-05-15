import torch
import time

def calculate_fidelity(preds, targets):
    # Basic fidelity metric: 1 - Hamming Distance ratio
    correct = (preds == targets).float()
    fidelity = correct.sum(dim=1) / preds.size(1)
    return fidelity.mean().item()

def calculate_gate_depth(circuit_tokens):
    # Very naive gate depth estimation: number of unique time steps
    return circuit_tokens.size(1)

def measure_inference_time(model, input_tensor, device='cpu'):
    input_tensor = input_tensor.to(device)
    model = model.to(device)
    model.eval()
    start = time.time()
    with torch.no_grad():
        _ = model(input_tensor)
    return time.time() - start
