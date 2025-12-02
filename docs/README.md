# Seminar with Practical: Scalable Computing Systems and Applications in AI, Big Data and HPC
## Project's name: *Opti-Perf*

### ToDo

1. Define a workload
2. Define a Benchmark
3. Define the enhancement scope
4. Selection of various optimizers
5. Summary of target and project setups

### 1. Define a workload

To test and compare the performance tweaks of various optimization 
algorithms PyTroch is used to implement a neural network which trains
on the MINST-dataset. PyTorch offers a C++ frontend which makes it easy
to integrate in the project setup and thus easy to integrate 
frameworks like CUDA and OpenCL. The following
guide will help to implement the neural network;

[C++ Frontend Guide](https://docs.pytorch.org/tutorials/advanced/cpp_frontend.html)

Further to integrate custom implementation of various optimization algorithms,
as for example Adam, this guide explains how to integrate them;

[Using a custom optimizer in PyTorch](https://www.geeksforgeeks.org/machine-learning/custom-optimizers-in-pytorch/)

### 2. Define a Benchmark

Based on "Best practices for benchmarks for optimization algorithms" the application is reviewed on 4 categories. 
The workload considered is to train a DCGAN. Measuring the performance of various optimization algorithms running on GPU with OpenCL and CUDA. 
One the following metrics:

- **Efficiency**: 
    - GPU/CPU operation time until solution obtained (computation). 
    - Data Transfer Time from disk to GPU (I/O).
- **Reliability**: 
    - Accuracy / Success rate for every 10th epoch.
- **Quality of Solution**: 
    - Normalized final error and variability.

Best practices to define a benchmark for performance - https://arxiv.org/pdf/1709.08242

### 3. Define the enhancement scope
<!-- Either compare CPU with GPU or only check upon GPU tweaks
     both in terms of I/O and computation. -->
     
- Comparing computation and I/O for CPU and GPU in this workload might not provide 
good insights on the acutal improvement since GPU will always outperform CPU on the given workload.

- What might be a better idea is to have a look at the performance of the two large frameworks
for GPU utilization OpenCL and CUDA. Does using either OpenCL or CUDA bring performance increasements?
There were done studies in the past but they are more than 10 years old, 
e.g.: https://ieeexplore.ieee.org/document/6047190/ and https://arxiv.org/abs/1005.2581.

- Therefore the performance enhancement of different optimizers is done with CUDA and OpenCL to see
whether one of the frameworks and a certain optimizer will cause a performance increasement for 
training the neural network or at least just to give an overview which setup performs best.

### 4. Selection of various optimizers
- Various Survey and Review papers:
    - Techniques and optimization algorithms in machine learning: A review: https://doi.org/10.70593/978-81-981271-4-3_2 
    - A Review of Optimization Algorithms for Training Neural Networks: https://ieeexplore.ieee.org/document/10303287
    - An overview of gradient descent optimization algorithms: https://doi.org/10.48550/arXiv.1609.04747
    - A Survey of Optimization Methods from a Machine Learning Perspective: https://arxiv.org/pdf/1906.06821
    - On Training Neural Networks with Mixed Integer Programming: https://starlab.ewi.tudelft.nl/yorke-smith/papers/w43.pdf

- Most commonly covered optimizers in the given papers:
    - Stochastic gradient descent, mentioned in all
    - AdaGrad, mentioned in all
    - Adam, mentioned in all
    - AdaDelta, mentioned in all
    - Root Mean Square Propagation (RMSprop), mentioned in all
    
- Also interesting optimizers mentioned:
    - Particle Swarm Optimisation (PSO), mentioned in https://arxiv.org/pdf/1906.06821, 
    - Mixed Integer Programming (MIP), mentioned in  https://starlab.ewi.tudelft.nl/yorke-smith/papers/w43.pdf
    
### 5. Summary of the target and project setup

The project compares the usage of various optimizers as of SGD, Adam, RMSprop, PSO and MIP for 
training a neural network to detect handwritten number figures from the MINST dataset. It implements
each optimzer in OpenCL and CUDA and compares the performance on GPU operation time, Data Transfer Time
from disk to GPU, Success rate for every 10th epoch and normalized final error. The report will answer
the question, *"which combination of the commonly used optimizers works better with OpenCL or CUDA?"*.

