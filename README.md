# FPGA Neural Network Accelerator (Vitis HLS)

int8-quantized MLP inference accelerator for MNIST digit classification, implemented in C++ using Vitis HLS and targeting the AMD KR260 Robotics Starter Kit (Zynq UltraScale+).

## Results

| Metric | Value |
|---|---|
| C-Sim Accuracy | 99% (99/100 MNIST test images) |
| Full Test Accuracy | 97.44% (10,000 images) |
| FPGA Inference Latency | **8.89 µs** (889 cycles @ 100 MHz) |
| Timing (post-synthesis) | 7.3 ns achieved vs 10 ns target |
| DSP Utilization | 93% (1168/1248) |
| LUT Utilization | 60% (70974/117120) |
| FF Utilization | 33% (78945/234240) |

### CPU vs FPGA Latency (single-image, 1000-image average)

| Platform | Avg Latency | Speedup |
|---|---|---|
| PyTorch (CPU) | 53.5 µs | 1× baseline |
| NumPy (CPU) | 31.8 µs | 1.7× |
| **FPGA (HLS, synthesized)** | **8.89 µs** | **6× over PyTorch** |

> Benchmarked on a laptop CPU (single-image mode). The FPGA advantage is largest in real-time, low-latency scenarios where batch processing is not an option.

## Architecture

```
Input (784)
    │
    ▼
Linear(784→256) + ReLU        [Layer 1: 4.22 µs]
    │
    ▼
Linear(256→128) + ReLU        [Layer 2: 2.45 µs]
    │
    ▼
Linear(128→10) + argmax       [Output:  1.48 µs]
    │
    ▼
Predicted Digit (0–9)
```

Weights are int8-quantized from float32 training, stored as compile-time constants in `mnist.h`. Inference uses fixed-point arithmetic (`ap_fixed<16,6>`) throughout to maximize DSP utilization.

## Repository Structure

```
EECS221-HLS/
├── MLP.cpp               # HLS top-level function (synthesized)
├── MLP.h                 # Data types, layer dimensions
├── MLP_tb.cpp            # C-sim testbench (loads real MNIST binary files)
├── mnist.h               # Int8 weights/biases (generated, see .gitignore)
├── generate_weights.py   # Train MLP, quantize to int8, export mnist.h
├── cpu_benchmark.py      # CPU vs FPGA latency comparison
├── mlp_fast/
│   └── hls_config.cfg    # Vitis HLS project config (part, clock, files)
└── data/                 # MNIST dataset (downloaded automatically)
```

## Tools

| Tool | Version |
|---|---|
| Vitis HLS | 2025.1 |
| Target Device | xck26-sfvc784-2LV-c (AMD KR260) |
| Python | 3.x |
| PyTorch | for training + CPU benchmark |

## Setup

### 1. Generate weights

Train the model and export quantized weights to `mnist.h`:

```bash
pip install torch torchvision numpy
python generate_weights.py
```

This trains for 5 epochs (~97% accuracy), quantizes weights to int8, and writes `mnist.h` with scale factors printed to stdout.

### 2. Download MNIST binary files

The testbench requires the raw (uncompressed) MNIST binary files:

```bash
# Linux / WSL
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz
```

```powershell
# Windows PowerShell (via Google mirror)
Invoke-WebRequest -Uri "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz" -OutFile t10k-images.gz
Invoke-WebRequest -Uri "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz" -OutFile t10k-labels.gz
python -c "
import gzip, shutil
for name in [('t10k-images.gz','t10k-images-idx3-ubyte'),('t10k-labels.gz','t10k-labels-idx1-ubyte')]:
    with gzip.open(name[0],'rb') as f_in, open(name[1],'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
"
```

### 3. Run C Simulation

```bash
vitis-run.bat --mode hls --csim \
  --config mlp_fast/hls_config.cfg \
  --work_dir mlp_fast
```

Expected output:
```
Running C-sim on 100 MNIST images...
  [0] predicted=7 actual=7 PASS
  ...
Accuracy: 99/100 = 99%
CSim done with 0 errors.
```

### 4. Run C Synthesis

```bash
vitis-run.bat --mode hls --syn \
  --config mlp_fast/hls_config.cfg \
  --work_dir mlp_fast
```

### 5. Run CPU benchmark

```bash
python cpu_benchmark.py
```

## Known Issues / Future Work

**BRAM over-subscription:** Synthesis reports 192% BRAM utilization (554/288 BRAM_18K). This is caused by 16 independent AXI master ports generated for the weight/bias pointer arguments. Fix: bundle all `m_axi` ports into a single interface, or move weights to internal ROM arrays to eliminate AXI adapters entirely. This is the primary remaining optimization before physical deployment.

**HBM driver packaging error:** Vitis 2025.1 attempts to cross-compile an ARM64 (`aarch64`) driver binary on x86 Windows when targeting the KR260. This fails at the packaging step but does not affect synthesis correctness. Workaround: set `package.output.format=xo` in `hls_config.cfg`.