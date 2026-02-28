"""
cpu_benchmark.py
Trains a 784->256->128->10 MLP on MNIST (same architecture as the HLS design),
then benchmarks single-image CPU inference latency and compares to FPGA synthesis result.

Requirements:
    pip install torch torchvision numpy

Usage:
    python cpu_benchmark.py
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── Config ─────────────────────────────────────────────────────────────────────
FPGA_LATENCY_US  = 8.89      # from Vitis HLS synthesis report (cycles=889, clk=10ns)
BENCHMARK_IMAGES = 1000      # number of images to average over
TRAIN_EPOCHS     = 5
BATCH_SIZE       = 256
LEARNING_RATE    = 1e-3
DATA_DIR         = './data'
MODEL_SAVE_PATH  = './mlp_trained.pth'

# ── Model (mirrors HLS architecture exactly) ───────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)

# ── Data ───────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("Loading MNIST dataset...")
train_dataset = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=1,           shuffle=False)

# ── Train ──────────────────────────────────────────────────────────────────────
device = torch.device('cpu')   # keep on CPU — we're benchmarking CPU inference
model  = MLP().to(device)
opt    = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

print(f"\nTraining {TRAIN_EPOCHS} epochs on CPU...")
for epoch in range(1, TRAIN_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    correct = 0
    for imgs, labels in train_loader:
        imgs   = imgs.view(-1, 784).to(device)
        labels = labels.to(device)
        opt.zero_grad()
        out  = model(imgs)
        loss = loss_fn(out, labels)
        loss.backward()
        opt.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
    acc = correct / len(train_dataset) * 100
    print(f"  Epoch {epoch}/{TRAIN_EPOCHS}  loss={total_loss/len(train_dataset):.4f}  train_acc={acc:.2f}%")

# ── Evaluate accuracy ──────────────────────────────────────────────────────────
model.eval()
correct = 0
with torch.no_grad():
    for imgs, labels in DataLoader(test_dataset, batch_size=256):
        imgs = imgs.view(-1, 784)
        correct += (model(imgs).argmax(1) == labels).sum().item()
test_acc = correct / len(test_dataset) * 100
print(f"\nTest accuracy: {test_acc:.2f}%")

# ── Benchmark: PyTorch single-image latency ────────────────────────────────────
print(f"\nBenchmarking single-image latency over {BENCHMARK_IMAGES} images...")

model.eval()
torch_latencies = []

# Warm-up (avoid timing JIT / cache cold start)
with torch.no_grad():
    for imgs, _ in test_loader:
        _ = model(imgs.view(1, 784))
        break

with torch.no_grad():
    for i, (imgs, _) in enumerate(test_loader):
        if i >= BENCHMARK_IMAGES:
            break
        x = imgs.view(1, 784)
        t0 = time.perf_counter()
        _  = model(x)
        t1 = time.perf_counter()
        torch_latencies.append((t1 - t0) * 1e6)   # microseconds

torch_avg_us = np.mean(torch_latencies)
torch_p50_us = np.percentile(torch_latencies, 50)
torch_p99_us = np.percentile(torch_latencies, 99)

# ── Benchmark: Pure NumPy single-image latency (closer to bare-metal C) ────────
print(f"Benchmarking NumPy inference...")

# Extract weights as numpy arrays
with torch.no_grad():
    w1 = model.net[0].weight.numpy()   # (256, 784)
    b1 = model.net[0].bias.numpy()     # (256,)
    w2 = model.net[2].weight.numpy()   # (128, 256)
    b2 = model.net[2].bias.numpy()     # (128,)
    w3 = model.net[4].weight.numpy()   # (10, 128)
    b3 = model.net[4].bias.numpy()     # (10,)

def mlp_numpy(x):
    h1 = np.maximum(0.0, x @ w1.T + b1)
    h2 = np.maximum(0.0, h1 @ w2.T + b2)
    return h2 @ w3.T + b3

numpy_latencies = []

# Warm-up
for imgs, _ in test_loader:
    _ = mlp_numpy(imgs.view(1, 784).numpy())
    break

for i, (imgs, _) in enumerate(test_loader):
    if i >= BENCHMARK_IMAGES:
        break
    x = imgs.view(1, 784).numpy()
    t0 = time.perf_counter()
    _  = mlp_numpy(x)
    t1 = time.perf_counter()
    numpy_latencies.append((t1 - t0) * 1e6)

numpy_avg_us = np.mean(numpy_latencies)
numpy_p50_us = np.percentile(numpy_latencies, 50)
numpy_p99_us = np.percentile(numpy_latencies, 99)

# ── Results ────────────────────────────────────────────────────────────────────
torch_speedup = torch_avg_us / FPGA_LATENCY_US
numpy_speedup = numpy_avg_us / FPGA_LATENCY_US

print(f"""
╔══════════════════════════════════════════════════════════════╗
║          CPU vs FPGA Inference Benchmark Results             ║
╠══════════════════════════════════════════════════════════════╣
║  Architecture: 784 → 256 → 128 → 10 (ReLU, int8 weights)     ║
║  Dataset: MNIST test set  ({BENCHMARK_IMAGES} images sampled)║
║  Test Accuracy:{test_acc:6.2f}%                              ║
╠══════════════════════════════════════════════════════════════╣
║  PyTorch (CPU)                                               ║
║    Avg latency : {torch_avg_us:8.1f} µs                      ║
║    p50 latency : {torch_p50_us:8.1f} µs                      ║
║    p99 latency : {torch_p99_us:8.1f} µs                      ║
╠══════════════════════════════════════════════════════════════╣
║  NumPy (CPU, bare-metal closer)                              ║
║    Avg latency : {numpy_avg_us:8.1f} µs                      ║
║    p50 latency : {numpy_p50_us:8.1f} µs                      ║
║    p99 latency : {numpy_p99_us:8.1f} µs                      ║
╠══════════════════════════════════════════════════════════════╣
║  FPGA (Vitis HLS synthesis, 100 MHz)                         ║
║    Latency     :     8.89 µs  (889 cycles @ 10 ns)           ║
╠══════════════════════════════════════════════════════════════╣
║  Speedup over PyTorch : {torch_speedup:5.1f}×                ║
║  Speedup over NumPy   : {numpy_speedup:5.1f}×                ║
╚══════════════════════════════════════════════════════════════╝
""")

# ── Save model for reproducibility ────────────────────────────────────────────
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model weights saved to {MODEL_SAVE_PATH}")
