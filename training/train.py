import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os

from datasets.data_loader import QuantumCircuitDataset
from model.transformer_model import TransformerModel
from training.loss_functions import get_loss_function
from utils.metrics import calculate_fidelity, calculate_gate_depth

writer = SummaryWriter(log_dir="runs/qc_optimization")



def train_model(epochs=10, batch_size=16, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = QuantumCircuitDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Task type: 'sequence' or 'token'
    task_type = 'sequence'

    model = TransformerModel(
        vocab_size=6,
        task_type=task_type,
        num_layers=2,
        dropout=0.2
    ).to(device)

    loss_fn = get_loss_function()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)

    # Initialize lists to collect metrics per epoch
    loss_list = []
    fidelity_list = []
    gate_depth_list = []
    inference_time_list = []
    epoch_time_list = []

    print("Training model...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_fidelity = 0.0
        total_gate_depth = 0.0
        total_inference_time = 0.0
        start_time = time.time()

        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            inference_start = time.time()
            output = model(x)
            inference_end = time.time()

            # Handle token vs sequence tasks
            if task_type == 'token':
                output = output.view(-1, output.size(-1))  # (B*S, num_classes)
                y = y.view(-1)
            elif task_type != 'sequence':
                raise ValueError(f"Unknown task_type: {task_type}")

            loss = loss_fn(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Metrics
            batch_fidelity = calculate_fidelity(output, y)
            batch_depth = calculate_gate_depth(output)
            inference_time = inference_end - inference_start

            total_loss += loss.item()
            total_fidelity += batch_fidelity
            total_gate_depth += batch_depth
            total_inference_time += inference_time

            writer.add_scalar("Batch/Loss", loss.item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar("Batch/Fidelity", batch_fidelity, epoch * len(dataloader) + batch_idx)
            writer.add_scalar("Batch/GateDepth", batch_depth, epoch * len(dataloader) + batch_idx)
            writer.add_scalar("Batch/InferenceTime", inference_time, epoch * len(dataloader) + batch_idx)

        avg_loss = total_loss / len(dataloader)
        avg_fidelity = total_fidelity / len(dataloader)
        avg_gate_depth = total_gate_depth / len(dataloader)
        avg_inference_time = total_inference_time / len(dataloader)
        epoch_time = time.time() - start_time

        loss_list.append(avg_loss)
        fidelity_list.append(avg_fidelity)
        gate_depth_list.append(avg_gate_depth)
        inference_time_list.append(avg_inference_time)
        epoch_time_list.append(epoch_time)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Fidelity: {avg_fidelity:.4f}")
        print(f"Gate Depth: {avg_gate_depth:.2f}")
        print(f"Inference Time: {avg_inference_time:.4f}s")
        print(f"Epoch Time: {epoch_time:.2f}s")

        writer.add_scalar("Epoch/Loss", avg_loss, epoch)
        writer.add_scalar("Epoch/Fidelity", avg_fidelity, epoch)
        writer.add_scalar("Epoch/GateDepth", avg_gate_depth, epoch)
        writer.add_scalar("Epoch/InferenceTime", avg_inference_time, epoch)

    # Save model and close writer
    torch.save(model.state_dict(), "checkpoints/tqco_model.pt")
    writer.close()
    print("\n✅ Training complete. Model saved.\n")

    # Return metrics for dashboard
    return {
        "loss": loss_list,
        "fidelity": fidelity_list,
        "gate_depth": gate_depth_list,
        "inference_time": inference_time_list,
        "epoch_time": epoch_time_list
    }

# Optional: Only run training if this script is executed directly
if __name__ == '__main__':
    train_model()
