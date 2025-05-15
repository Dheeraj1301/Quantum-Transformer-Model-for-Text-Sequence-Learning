import torch
import time

def calculate_fidelity(output, target):
    # Convert logits to predicted class indices if output is multi-class
    if output.dim() > 1 and output.size(-1) > 1:
        predicted = torch.argmax(output, dim=-1)
    else:
        predicted = output.round()  # For binary classification

    # Ensure target is the same type/shape as predicted
    if predicted.shape != target.shape:
        target = target.view_as(predicted)

    correct = (predicted == target).sum().item()
    total = target.numel()
    return correct / total if total > 0 else 0.0


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
