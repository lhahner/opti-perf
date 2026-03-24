# OptiPerf 
> `optiPerf` is a benchmark program to measure and compare different machine
  learning optimizers implementations in OpenCL and CUDA. Currently implementing
  Adam optimizer.

## Install
To install optiPerf simply run the shell script, depending on your system.
Just make sure you are connected to a GPU and have run permission on the file.

```
./install.sh
```

## Run
After install the program is already compiled with the options defined in the 
install.sh file. To run the benchmark first verify which combination
you want to run for benchmark in `config.yaml`. If have setup your configs,
just run: 

```
./build/app
```
The benchmark data is stored inside of `data/logs/benchmarks-logs.csv`.

### Real training validation benchmark
To validate the Adam kernel timings against a real machine learning workload,
set `runtime.workload: "Training"` in `config.yaml`. This runs a mini-batch
MNIST linear classifier training step while reusing the same CPU, OpenCL, and
CUDA Adam implementations.

The workload expects the unzipped MNIST IDX files in `data/mnist/`:

```
train-images-idx3-ubyte
train-labels-idx1-ubyte
```

You can download them with:

```
python3 scripts/load_download_mnist.py -d data/mnist
```

### Run a batch 
If you want to executed a bunch of runs use `run.sh` and specify how
much iterations of the program you want to run. E.g. to run 10 batches:

```
run.sh 10
```
