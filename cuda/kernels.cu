#include "cudaWrapper.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(call) \
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


extern "C" int forwardBFS(GraphCsr graph) { 
  int* d_numNodes;
  int* d_numEdges;
  int* d_rowPtr;
  int* d_colInd;

  CHECK_CUDA( cudaMalloc(&d_numNodes, sizeof(int)) );
  CHECK_CUDA( cudaMalloc(&d_numEdges, sizeof(int)) );
  CHECK_CUDA( cudaMalloc(&d_rowPtr, graph.numNodes * sizeof(int)) );
  CHECK_CUDA( cudaMalloc(&d_colInd, graph.numEdges * sizeof(int)) );

  CHECK_CUDA( cudaMemcpy(d_numNodes, grap.numNodes , sizeof(int) , 
                        cudaMemcpyHostToDevice );
  

  CHECK_CUDA( cudaMemcpy(d_numEdges, grap.numEdges , sizeof(int), 
                        cudaMemcpyHostToDevice );

  CHECK_CUDA( cudaMemcpy(d_rowPtr, graph.rowPtr , 
                        numNodes * sizeof(int), 
                        cudaMemcpyHostToDevice );

  CHECK_CUDA( cudaMemcpy(d_colInd, graph.colInd, 
                        graph.numEdges * sizeof(int), 
                        cudaMemcpyHostToDevice );
  
  //here you shold launch the algorith for every source 
  //for now it will be limitated to one
  

  //for the first levels it is better to use the CPU rather than the GPU
  //because the overhead will be more than the execution 



}


extern "C" void backwardPropagation() {
}
