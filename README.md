# OptiPerf

`OptiPerf` is a benchmark application for measuring and comparing machine-learning optimizer implementations across CPU, OpenCL, and CUDA backends. The current implementation focuses on the Adam optimizer and supports two benchmark modes:

- `GEMM`: synthetic matrix-multiplication workload
- `Training`: MNIST linear-classifier training workload

## Table of Contents

- [Overview](#overview)
- [Repository Layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Installation and Build](#installation-and-build)
- [Configuration](#configuration)
- [Running the Program](#running-the-program)
- [Running Multiple Times](#running-multiple-times)
- [Validation Against PyTorch](#validation-against-pytorch)
- [Scripts and Data Analysis](#scripts-and-data-analysis)
- [Output Files](#output-files)

## Overview

The program reads its runtime settings from `config.yaml` by default and then registers the matching benchmark at startup. You can choose:

- `runtime.workload`: `GEMM` or `Training`
- `runtime.optimizer`: currently `Adam`
- `runtime.framework`: `CPU`, `OpenCL`, or `CUDA`

The main executable is:

```bash
./build/app
```

## Repository Layout

Important paths:

- `config.yaml`: runtime, optimizer, and workload configuration
- `install.sh`: build script for local and HPC environments
- `run.sh`: helper script to execute the benchmark multiple times
- `src/`: C++ and CUDA implementation
- `kernels/`: OpenCL kernels
- `scripts/`: dataset, validation, and plotting utilities
- `data/mnist/`: MNIST dataset files
- `data/logs/`: generated benchmark CSV files

## Prerequisites

To build and run the project you generally need:

- CMake 3.18 or newer
- a C++20 compiler
- CUDA toolkit and `nvcc`
- OpenCL headers and runtime
- `gcc-11` and `g++-11` on non-HPC systems, or matching GCC modules on HPC
- Git submodules enabled

Optional but useful:

- Python 3 for helper scripts
- `matplotlib`, `pandas`, and `torch` for analysis and validation scripts

The file `requirements.sh` contains an example setup for Ubuntu-like systems, but it should be treated as a reference rather than a complete package manager script.

## Installation and Build

Clone the repository:

```bash
git clone https://github.com/lhahner/opti-perf.git
cd opti-perf
```

Make the installer executable:

```bash
chmod +x install.sh
```

Run the build script:

```bash
./install.sh
```

What `install.sh` does:

- initializes git submodules
- configures CMake in `build/`
- builds the `app` target
- selects different compiler and CUDA settings for HPC vs non-HPC environments

Notes:

- On non-HPC systems, the script expects `gcc-11` and `g++-11`.
- You can override the CUDA architecture with `CUDA_ARCH`, for example:

```bash
CUDA_ARCH=80 ./install.sh
```

## Configuration

The default configuration file is `config.yaml`:

```yaml
runtime:
  workload: "GEMM"
  optimizer: "Adam"
  framework: "OpenCL"

optimizer:
  learning_rate: 0.001
  beta_1: 0.9
  beta_2: 0.999
  epsilon: 1e-08
  dim_m: 3024
  dim_k: 3024
  dim_n: 256
  batch_size: 256

workload:
  dataset_dir: "data/mnist"
  max_samples: 1024
  num_epochs: 5
```

### Configuration Sections

`runtime`

- `workload`: benchmark type, either `GEMM` or `Training`
- `optimizer`: currently `Adam`
- `framework`: `CPU`, `OpenCL`, or `CUDA`

`optimizer`

- `learning_rate`, `beta_1`, `beta_2`, `epsilon`: Adam hyperparameters
- `dim_m`, `dim_k`, `dim_n`: GEMM dimensions
- `batch_size`: number of optimizer steps per run

`workload`

- `dataset_dir`: MNIST directory used by the training workload
- `max_samples`: number of training samples to use
- `num_epochs`: number of training epochs for the `Training` workload

### Using a Different Config File

You can point the application to another config file with `OPTI_PERF_CONFIG`:

```bash
OPTI_PERF_CONFIG=/path/to/your-config.yaml ./build/app
```

## Running the Program

After building, run the benchmark with:

```bash
./build/app
```

The program reads `config.yaml`, registers the matching benchmark, and executes it.

### Example: GEMM Benchmark

Use this for a synthetic GEMM benchmark with OpenCL:

```yaml
runtime:
  workload: "GEMM"
  optimizer: "Adam"
  framework: "OpenCL"

optimizer:
  learning_rate: 0.001
  beta_1: 0.9
  beta_2: 0.999
  epsilon: 1e-08
  dim_m: 3024
  dim_k: 3024
  dim_n: 256
  batch_size: 256

workload:
  dataset_dir: "data/mnist"
  max_samples: 1024
  num_epochs: 5
```

Then run:

```bash
./build/app
```

### Example: Training Benchmark

To benchmark Adam inside a real training loop, set:

```yaml
runtime:
  workload: "Training"
  optimizer: "Adam"
  framework: "CUDA"
```

This runs a mini-batch MNIST linear-classifier workload using the selected backend.

### MNIST Dataset for Training

The `Training` workload expects the raw MNIST IDX files in `data/mnist/`, including:

```text
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

You can download and unzip them with:

```bash
python3 scripts/load_download_mnist.py -d data/mnist
```

## Running Multiple Times

If you want to execute the current configuration repeatedly, use `run.sh`:

```bash
./run.sh 10
```

This runs `./build/app` ten times in a row.

## Validation Against PyTorch

The repository also provides a PyTorch reference script for the MNIST training workload:

```bash
python3 scripts/pytorch_mnist_reference.py \
  --dataset-dir data/mnist \
  --batch-size 256 \
  --epochs 5 \
  --max-samples 1024
```

Useful options:

- `--learning-rate`
- `--beta1`
- `--beta2`
- `--epsilon`
- `--warmup-steps`
- `--output`

By default, the script writes its results to:

```text
data/logs/pytorch-mnist-reference.csv
```

## Scripts and Data Analysis

Most helper scripts live in `scripts/`. For plotting, make sure the required Python packages are installed.

### Post-process benchmark logs

`postprocess_benchmark_logs.py` repairs malformed benchmark CSV files and reconstructs missing `batch_index` values.

Run it with defaults:

```bash
python3 scripts/postprocess_benchmark_logs.py
```

Or specify input and output explicitly:

```bash
python3 scripts/postprocess_benchmark_logs.py \
  --input data/logs/benchmarks-logs.csv \
  --output data/logs/benchmarks-logs.postprocessed.csv
```

### Plot GEMM spread lines

Plots mean timing curves with spread bands from cleaned GEMM benchmark logs:

```bash
python3 scripts/plot_benchmark_spread_lines.py \
  --input data/logs/benchmarks-logs.postprocessed.csv \
  --outdir scripts/plots/benchmark_spread_lines
```

### Plot validation clusters

Creates per-device validation scatter plots from training benchmark logs:

```bash
python3 scripts/plot_validation_clusters.py \
  --input data/logs/validation-benchmark-logs.csv \
  --outdir scripts/plots/validation_clusters
```

### Other analysis scripts

Additional plotting and reporting scripts are available in `scripts/`, including:

- `plot_benchmarks.py`
- `plot_benchmark_times.py`
- `plot_opencl_cuda_devicewise.py`
- `plot_benchmark_clusters.py`
- `plot_benchmark_clusters_cleaned.py`
- `plot_batch_index_distributions.py`
- `plot_batch_index_variance_std.py`
- `mean_times_by_device_and_input_size.py`
- `mean_full_step_times.py`
- `mean_validation_times.py`
- `batch_index_stats_by_device_and_input_size.py`
- `fix_numeric_decimal_separators.py`

For each script, run `python3 <script> --help` to see the supported arguments.

## Output Files

The main generated files are:

- `data/logs/benchmarks-logs.csv`: default benchmark log output
- `data/logs/benchmarks-logs.postprocessed.csv`: repaired benchmark log output
- `data/logs/validation-benchmark-logs.csv`: training workload validation output
- `data/logs/pytorch-mnist-reference.csv`: PyTorch reference output
- `scripts/plots/...`: generated plots from analysis scripts

If a relative log filename is used internally, it is resolved under `data/logs/`.
