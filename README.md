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
install.sh file. Just run the following command to gather benchmark data.

```
./build/bench
```
The benchmark data is stored inside of `data/logs/benchmarks-logs.csv`.
