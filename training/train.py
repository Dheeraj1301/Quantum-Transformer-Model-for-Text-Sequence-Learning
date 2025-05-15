import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.data_loader import QuantumCircuitDataset
from model.transformer_model import TransformerCircuitOptimizer
from training.loss_functions import get_loss_function

def train_model(epochs=10, batch_size=16, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = QuantumCircuitDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerCircuitOptimizer(vocab_size=6).to(device)
    loss_fn = get_loss_function()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            out = out.view(-1, out.size(-1))
            y = y.view(-1)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader)}")

    torch.save(model.state_dict(), "checkpoints/tqco_model.pt")

if __name__ == '__main__':
    train_model()
