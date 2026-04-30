#include "cudaWrapper.h"
#include <driver_types.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
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
            std::vector<int>& frontierNext, 
            std::vector<int>& distances, std::vector<double>& sigmas, int current_wave) {
  //keeps ausiliary structure for visited nodes should be modified later on 
  //to utilize red and black tree for frontierCurrent and frontierNext,
  //the problem is passing the vector calculated CUDA to the tree

  for(int node : frontierCurrent) {
    int start = graph.rowPtr[node];
    int end = graph.rowPtr[node + 1];
    for(int i = start; i < end; i++) {
      int neighbor = graph.colInd[i];
      if(distances[neighbor] == -1) {
        distances[neighbor] = current_wave;
        sigmas[neighbor] += sigmas[node];
        frontierNext.push_back(neighbor);
      }else if(distances[neighbor] == current_wave) {
        sigmas[neighbor] += sigmas[node];
      }
    }
  }

}

__global__ 
void forwardBFS_kernel(int numCurrNodes, const int* d_rowPtr, const int* d_colInd, 
                       int* d_frontierCurrent, int* d_frontierNext, 
                       int* d_nextSize, int* d_distances, 
                       double* d_sigmas, int current_wave) {
  
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
      int expected = -1;
      int oldDist = atomicCAS(&d_distances[w], expected, current_wave);

      if (oldDist == expected) {
        atomicAdd(&d_sigmas[w], d_sigmas[v]);
        int winner = atomicAdd(&numCurrFrontier_s, 1);
        if (winner < BLOCK_SIZE) {
          currFrontier_s[winner] = w;
        } else { //bypass shared frontier and add directly in global
          int glob_idx = atomicAdd(d_nextSize, 1);
          d_frontierNext[glob_idx] = w;
        }
      } else if (oldDist == current_wave) {
        atomicAdd(&d_sigmas[w], d_sigmas[v]);
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
void pullMerits_kernel(int numNodes, int* d_distances, double* d_sigmas, double* d_deltas, int current_wave, 
                                const int* d_rowPtr, const int* d_colInd) {

  int t = blockIdx.x * blockDim.x + threadIdx.x;

  if(t < numNodes) {
    if(d_distances[t] == current_wave){
      
      int start = d_rowPtr[t];
      int end = d_rowPtr[t+1];

      double sum = 0.0;
      for(int i = start; i < end; i++) { //credit redistribution 
        int w = d_colInd[i];
        if (d_distances[w] == current_wave + 1) {
          sum += (d_sigmas[t] / d_sigmas[w]) * (1.0 + d_deltas[w]);
        }
      }
      d_deltas[t] += sum;
    }
  }
}


extern "C" int Brandes(GraphCsr graph) { 

  //allocating CPU variables
  std::vector<int> h_frontierCurrent;
  std::vector<int> h_frontierNext;
  std::vector<double> h_sigmas(graph.numNodes, 0.0);
  std::vector<int> h_distances(graph.numNodes, -1);
  std::vector<double> h_deltas(graph.numNodes, 0.0);

  int maxWaves = 0;
  int numCurrNodes;
  bool visitedStateOnDevice = false;
  int h_wave = 1;

  //initializing CPU variables
  h_sigmas[0] = 1.0;
  h_distances[0] = 0;
  h_frontierCurrent.push_back(0); 

  //allocating GPU variables
  int* d_rowPtr;
  int* d_colInd;
  int* d_frontierCurrent;
  int* d_frontierNext;
  int* d_nextSize;
  int* d_distances;
  double* d_sigmas;
  double* d_deltas;

  CHECK_CUDA(cudaMalloc(&d_rowPtr, graph.numNodes * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_colInd, graph.numEdges * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_distances, graph.numNodes * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_sigmas, graph.numNodes * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_deltas, graph.numNodes * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_frontierCurrent, graph.numNodes * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_frontierNext, graph.numNodes * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_nextSize, sizeof(int)));

  //initialising GPU variables
  CHECK_CUDA(cudaMemcpy(d_rowPtr, graph.rowPtr, graph.numNodes * sizeof(int), cudaMemcpyHostToDevice)); 
  CHECK_CUDA(cudaMemcpy(d_colInd, graph.colInd, graph.numEdges * sizeof(int), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(d_distances, h_distances.data(), graph.numNodes * sizeof(int), cudaMemcpyHostToDevice));
  
  CHECK_CUDA(cudaMemcpy(d_sigmas, h_sigmas.data(), graph.numNodes * sizeof(double), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_deltas, h_deltas.data(), graph.numNodes * sizeof(double), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(d_frontierCurrent, h_frontierCurrent.data(), h_frontierCurrent.size() * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_frontierNext, 0, graph.numNodes * sizeof(int)));
  CHECK_CUDA(cudaMemset(d_nextSize, 0, sizeof(int))); 

  //main loop for BFS
  while(!h_frontierCurrent.empty()) {

    numCurrNodes = h_frontierCurrent.size();
    
    std::cout << "\n wave number " << h_wave << " Current frontier size: " << numCurrNodes << std::endl;

    bool FORCE_GPU_TEST = false;

    //CPU EXECUTION
    if(numCurrNodes < 50000 && !FORCE_GPU_TEST) {
      std::cout << "Executing on CPU" << std::endl;
      if (visitedStateOnDevice) {

        //Transfer the states from GPU to CPU
        CHECK_CUDA(cudaMemcpy(h_frontierCurrent.data(), d_frontierCurrent, numCurrNodes * sizeof(int), cudaMemcpyDeviceToHost));
        visitedStateOnDevice = false;
        CHECK_CUDA(cudaMemcpy(h_distances.data(), d_distances, graph.numNodes * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_sigmas.data(), d_sigmas, graph.numNodes * sizeof(double), cudaMemcpyDeviceToHost));

      }
      BFSCPU(graph, h_frontierCurrent, h_frontierNext, h_distances, h_sigmas, h_wave);
      
      visitedStateOnDevice = false;
      h_frontierCurrent = h_frontierNext;
      h_frontierNext.clear();
      h_wave++;
    }
    //GPU EXECUTION
    else {  
      std::cout << "Executing on GPU" << std::endl;
      if (!visitedStateOnDevice) {
        visitedStateOnDevice = true;
        //Transfer the states from CPU to GPU
        CHECK_CUDA(cudaMemcpy(d_distances, h_distances.data(), graph.numNodes * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_sigmas, h_sigmas.data(), graph.numNodes * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_frontierCurrent, h_frontierCurrent.data(), numCurrNodes * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_frontierNext, 0, graph.numNodes * sizeof(int)));
      }

      int numBlocks = (numCurrNodes + BLOCK_SIZE - 1) / BLOCK_SIZE;

      forwardBFS_kernel<<<numBlocks, BLOCK_SIZE>>>(numCurrNodes, d_rowPtr, d_colInd,
                                                   d_frontierCurrent, d_frontierNext,
                                                   d_nextSize, d_distances,
                                                   d_sigmas, h_wave); 
      cudaDeviceSynchronize();

      CHECK_CUDA(cudaMemcpy(&numCurrNodes, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));

      h_frontierCurrent.resize(numCurrNodes);
      

      std::swap(d_frontierCurrent, d_frontierNext);

      // Clearing variables for next call.
      CHECK_CUDA(cudaMemset(d_nextSize, 0, sizeof(int))); 
      h_wave++;
    }
  }

  //sync states based on last to execute
  if (!visitedStateOnDevice) {
    CHECK_CUDA(cudaMemcpy(d_distances, h_distances.data(), graph.numNodes * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sigmas, h_sigmas.data(), graph.numNodes * sizeof(double), cudaMemcpyHostToDevice));
  } else {
    CHECK_CUDA(cudaMemcpy(h_distances.data(), d_distances, graph.numNodes * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_sigmas.data(), d_sigmas, graph.numNodes * sizeof(double), cudaMemcpyDeviceToHost));
  }

  maxWaves = h_wave;

  std::cout << "\nForward BFS completed. Max waves: " << maxWaves << std::endl;
  std::cout << "Starting Pull Merits" << std::endl;

  for(int i = 0; i < maxWaves; i++) {
    int current_wave = maxWaves - 1 - i;
    int numBlocks = ( graph.numNodes + BLOCK_SIZE -1 )  / BLOCK_SIZE;
    
    pullMerits_kernel<<<numBlocks, BLOCK_SIZE>>>(graph.numNodes, 
                                                          d_distances, 
                                                          d_sigmas, d_deltas, current_wave, 
                                                          d_rowPtr, 
                                                          d_colInd);
  }
  cudaDeviceSynchronize();
  std::cout << "Pull Merits completed." << std::endl;

  CHECK_CUDA(cudaMemcpy(h_deltas.data(), d_deltas, 
             graph.numNodes * sizeof(double), 
             cudaMemcpyDeviceToHost ));


  //Finally calculating final BC
  std::vector<std::pair<double, int>> sorted_deltas;
  for(int i = 0; i < graph.numNodes; i++) {
    sorted_deltas.push_back({h_deltas[i], i});
  }
  std::sort(sorted_deltas.begin(), sorted_deltas.end(), std::greater<std::pair<double, int>>());
  
  int numPrint = (graph.numNodes < 15) ? graph.numNodes : 15;
  for(int i = 0; i < numPrint; i++) {
    std::cout << "Node " << sorted_deltas[i].second << " Centrality: " << sorted_deltas[i].first << std::endl;
  }
  std::cout << "---------------------\n" << std::endl;

  CHECK_CUDA(cudaFree(d_frontierCurrent));
  CHECK_CUDA(cudaFree(d_frontierNext));
  CHECK_CUDA(cudaFree(d_nextSize));
  CHECK_CUDA(cudaFree(d_rowPtr));
  CHECK_CUDA(cudaFree(d_colInd));
  CHECK_CUDA(cudaFree(d_distances));
  CHECK_CUDA(cudaFree(d_sigmas));
  CHECK_CUDA(cudaFree(d_deltas));

  return 0;
}
