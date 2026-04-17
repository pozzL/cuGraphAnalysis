#include "cudaWrapper.h"
#include <driver_types.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#define BLOCK_SIZE 256

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


void BFSCPU(GraphCsr graph, const std::vector<int>& frontierCurrent,
            const std::vector<int>& frontierNext) {
  //implement BFS on CPU

}

__global__
void backwardPropagation_kernel();


extern "C" int forwardBFS(GraphCsr graph) { 
  int* d_numNodes;
  int* d_numEdges;
  int* d_rowPtr;
  int* d_colInd;
  int* d_frontierCurrent;
  int* d_frontierNext;
  int* d_nextSize;

  CHECK_CUDA( cudaMalloc(&d_numNodes, sizeof(int)) );
  CHECK_CUDA( cudaMalloc(&d_numEdges, sizeof(int)) );
  CHECK_CUDA( cudaMalloc(&d_rowPtr, graph.numNodes * sizeof(int)) );
  CHECK_CUDA( cudaMalloc(&d_colInd, graph.numEdges * sizeof(int)) );


  CHECK_CUDA( 
    cudaMemcpy(d_numNodes, graph.numNodes , sizeof(int) ,
                         cudaMemcpyHostToDevice )
  );

  CHECK_CUDA( 
    cudaMemcpy(d_numEdges, graph.numEdges , sizeof(int), 
                        cudaMemcpyHostToDevice)
  );

  CHECK_CUDA( 
    cudaMemcpy(d_rowPtr, graph.rowPtr , 
                        graph.numNodes * sizeof(int), 
                        cudaMemcpyHostToDevice) 
  );

  CHECK_CUDA( 
    cudaMemcpy(d_colInd, graph.colInd, 
                        graph.numEdges * sizeof(int), 
                        cudaMemcpyHostToDevice)
  );
  
  //variables for CPU computation of the graph
  std::vector<int> h_frontierCurrent;
  std::vector<int> h_frontierNext;

  h_frontierCurrent.push(0);
  h_frontierNext.assign(graph.numNodes, 0);
  int numCurrNodes;
  numCurrNodes = h_frontierCurrent.size();

  while(!h_frontierCurrent.empty()) {

    if(numCurrNodes < 50000) //CPU EXECUTION
      BFSCPU(graph,h_frontierCurrent,h_frontierNext);
    else {  //GPU EXECUTION

      CHECK_CUDA(
        cudaMalloc(&d_frontierCurrent, h_frontierCurrent.size() * sizeof(int))
      );

      CHECK_CUDA(
        cudaMalloc(&d_frontierNext, graph.numNodes * sizeof(int)) 
        //i dont know how many will be in next level so i set full lenght
      );

      CHECK_CUDA( 
        cudaMalloc(&d_nextSize, sizeof(int)) 
        //glocal counter for next level (CAS add)
      );
      
      CHECK_CUDA(cudaMemcpy(d_frontierCurrent ,
                            h_frontierCurrent.data(),
                            h_frontierCurrent.size()*sizeof(int),
                            cudaMemcpyHostToDevice)
      );
      
      CHECK_CUDA(cudaMemcpy(d_frontierNext,
                            NULL,
                            h_frontierCurrent.size()*sizeof(int),
                            cudaMemcpyHostToDevice)
      );
      CHECK_CUDA(cudaMemcpy(d_nextSize, 0, sizeof(int), cudaMemcpyHostToDevice)); 
      
      
      int numBlocks = (h_frontierCurrent.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;

      forwardBFS_kernel<<<numBlocks, BLOCK_SIZE>>>(d_numNodes,
                                                   d_numEdges,d_rowPtr,d_colInd); 
      cudaDeviceSynchronize():


      CHECK_CUDA(
        cudaMemcpy(numCurrNodes,d_nextSize,sizeof(int),
                   cudaMemcpyDeviceToHost)
      );
      CHECK_CUDA(
        cudaMemcpy(h_frontierCurrent.data(), d_frontierNext, 
                   numCurrNodes*sizeof*(int), cudaMemcpyDeviceToHost)
      );
      h_frontierNext.assign(numNodes, 0);

      CHECK_CUDA(cudaFree(d_frontierCurrent));
      CHECK_CUDA(cudaFree(d_frontierNext));
      CHECK_CUDA(cudaFree(d_nextSize));
    }
  }

  CHECK_CUDA(cudaFree(d_numNodes));
  CHECK_CUDA(cudaFree(d_numEdges));
  CHECK_CUDA(cudaFree(d_rowPtr));
  CHECK_CUDA(cudaFree(d_colInd));
}


extern "C" void backwardPropagation() {
}
