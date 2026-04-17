#include "cudaWrapper.h"
#include <driver_types.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "graph.h"
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

void BFSCPU(const GraphCsr graph, std::vector<int>& frontierCurrent,
            std::vector<int>& frontierNext, std::vector<int>& visited) {
  //keeps ausiliary structure for visited nodes should be modified later on 
  //to utilize red and black tree for frontierCurrent and frontierNext,
  //the problem is passing the vector calculated CUDA to the tree
  for(int node : frontierCurrent) {
    int start = graph.rowPtr[node];
    int end = graph.rowPtr[node + 1];
    for(int i = start; i < end; i++) {
      int neighbor = graph.colInd[i];
      if(!visited[neighbor]) {
        visited[neighbor] = 1;
        frontierNext.push_back(neighbor);
      }
    }
  }

}

__global__ 
void forwardBFS_kernel(int numCurrNodes, int* d_rowPtr, int* d_colInd, int* d_frontierCurrent, 
                       int* d_frontierNext, int* d_nextSize, int* d_visited) {
  __shared__ int currFrontier_s[BLOCK_SIZE];
  __shared__ int numCurrFrontier_s;
  __shared__ int global_offset_s;

  if (threadIdx.x == 0) {
    numCurrFrontier_s = 0;
  }
  __syncthreads();

  int t = blockIdx.x * blockDim.x + threadIdx.x;

  if (t < numCurrNodes) {
    int v = d_frontierCurrent[t];
    int start = d_rowPtr[v];
    int end = d_rowPtr[v + 1];

    for (int i = start; i < end; i++) {
      int w = d_colInd[i];
      if (atomicCAS(&d_visited[w], 0, 1) == 0) {
        int winner = atomicAdd(&numCurrFrontier_s, 1);

        if (winner < BLOCK_SIZE) {
          currFrontier_s[winner] = w;
        } else { //bypass shared frontier and add directly in global
          int glob_idx = atomicAdd(d_nextSize, 1);
          d_frontierNext[glob_idx] = w;
        }
      }
    }
  }

  __syncthreads();

  int validElements = numCurrFrontier_s < BLOCK_SIZE ? numCurrFrontier_s : BLOCK_SIZE;
  if (threadIdx.x == 0 && validElements> 0) {
    global_offset_s = atomicAdd(d_nextSize, validElements);
  }

  __syncthreads();

  if(threadIdx.x < validElements) {
    d_frontierNext[global_offset_s + threadIdx.x] = currFrontier_s[threadIdx.x];
  }

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
  int* d_visited;
  CHECK_CUDA( cudaMalloc(&d_numNodes, sizeof(int)) );
  CHECK_CUDA( cudaMalloc(&d_numEdges, sizeof(int)) );
  CHECK_CUDA( cudaMalloc(&d_rowPtr, graph.numNodes * sizeof(int)) );
  CHECK_CUDA( cudaMalloc(&d_colInd, graph.numEdges * sizeof(int)) );
  CHECK_CUDA( cudaMalloc(&d_visited, graph.numNodes * sizeof(int)));

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
  std::vector<int> h_visited(graph.numNodes, 0);

  CHECK_CUDA(cudaMemcpy(d_visited, h_visited.data(), graph.numNodes * sizeof(int), cudaMemcpyHostToDevice));

  h_frontierCurrent.push_back(0); // Assuming start node is 0
  h_visited[0] = 1;
  CHECK_CUDA(cudaMemcpy(&d_visited[0], &h_visited[0], sizeof(int), cudaMemcpyHostToDevice));

  int numCurrNodes;
  bool visitedStateOnDevice = false;

  while(!h_frontierCurrent.empty()) {
    numCurrNodes = h_frontierCurrent.size();

    if(numCurrNodes < 50000) { //CPU EXECUTION
      if (visitedStateOnDevice) {
        CHECK_CUDA(cudaMemcpy(h_visited.data(), d_visited, graph.numNodes * sizeof(int), cudaMemcpyDeviceToHost));
        visitedStateOnDevice = false;
      }
      BFSCPU(graph, h_frontierCurrent, h_frontierNext, h_visited);

      h_frontierCurrent = h_frontierNext;
      h_frontierNext.clear();
    }
    else {  //GPU EXECUTION
      if (!visitedStateOnDevice) {
        CHECK_CUDA(cudaMemcpy(d_visited, h_visited.data(), graph.numNodes * sizeof(int), cudaMemcpyHostToDevice));
        visitedStateOnDevice = true;
      }

      CHECK_CUDA(cudaMalloc(&d_frontierCurrent, h_frontierCurrent.size() * sizeof(int)));
      CHECK_CUDA(cudaMalloc(&d_frontierNext, graph.numNodes * sizeof(int)));
      CHECK_CUDA(cudaMalloc(&d_nextSize, sizeof(int)));

      CHECK_CUDA(cudaMemcpy(d_frontierCurrent ,
                            h_frontierCurrent.data(),
                            h_frontierCurrent.size()*sizeof(int),
                            cudaMemcpyHostToDevice)
                 );

      CHECK_CUDA(cudaMemset(d_frontierNext, 0, h_frontierCurrent.size()*sizeof(int)));
      CHECK_CUDA(cudaMemset(d_nextSize, 0, sizeof(int))); 


      int numBlocks = (h_frontierCurrent.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;

      forwardBFS_kernel<<<numBlocks, BLOCK_SIZE>>>(d_numNodes,
                                                   d_numEdges,d_rowPtr,d_colInd); 
      cudaDeviceSynchronize():


        CHECK_CUDA(cudaMemcpy(&numCurrNodes, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));

      h_frontierCurrent.resize(numCurrNodes);
      CHECK_CUDA(cudaMemcpy(h_frontierCurrent.data(), d_frontierNext, numCurrNodes * sizeof(int), cudaMemcpyDeviceToHost));

      CHECK_CUDA(cudaFree(d_frontierCurrent));
      CHECK_CUDA(cudaFree(d_frontierNext));
      CHECK_CUDA(cudaFree(d_nextSize));
    }
  }

  CHECK_CUDA(cudaFree(d_visited));
  CHECK_CUDA(cudaFree(d_numNodes));
  CHECK_CUDA(cudaFree(d_numEdges));
  CHECK_CUDA(cudaFree(d_rowPtr));
  CHECK_CUDA(cudaFree(d_colInd));

  return 0;
}


extern "C" void backwardPropagation() {
}
