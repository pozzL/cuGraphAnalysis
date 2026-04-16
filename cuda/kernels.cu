#include "cudaWrapper.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

#define cudaCheck(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
      << " - " << cudaGetErrorString(err) << std::endl; \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

__global__ 
void forwardBFS_kernel() {
}

__global__
void backwardPropagation_kernel();


extern "C" int forwardBFS() {
}


extern "C" void backwardPropagation() {
}
