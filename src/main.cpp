#include <iostream>
#include <vector>
#include <utility>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include "graph.h"
#include "brandes.h"
#include "cudaWrapper.h"

const std::string filename = "../graphs/dataset.txt"; 

//should be parallelized later on 
void buildRowPtr(const std::vector<std::pair<int, int>>& edges, 
    int numNodes, std::vector<int>& rowPtr) {
  rowPtr.assign(numNodes + 1, 0);
  for (const auto& edge : edges) { //initialize the array for 
                                   //a[node] = number of connected
    rowPtr[edge.first + 1]++;
  }
  for (int i = 0; i < numNodes; i++) {
    rowPtr[i + 1] += rowPtr[i]; //sum the number of connected of the nodes 
                                //before himself to calculete start index
  }
}

//should be parellelized later on
void buildColInd(const std::vector<std::pair<int, int>>& edges, 
    const std::vector<int>& rowPtr, std::vector<int>& colInd) {
  colInd.resize(edges.size());
  std::vector<int> currentPos = rowPtr;
  for (const auto& edge : edges) {
    int u = edge.first;
    int v = edge.second;
    colInd[currentPos[u]] = v;
    currentPos[u]++;
  }
}

void readSNAPFile(const std::string& filename, std::vector<std::pair<int, 
    int>>& edges, int& numNodes) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error opening the file" << filename << std::endl;
    return;
  }

  std::string line;
  numNodes = 0;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream iss(line);
    int u, v;
    if (iss >> u >> v) {
      edges.push_back({u, v});
      numNodes = std::max({numNodes, u + 1, v + 1});
    }
  }
  file.close();
}

int main() {

  std::vector<std::pair<int, int>> edges;
  int numNodes = 0;

  std::cout << "Reading the graph" << filename << std::endl;
  readSNAPFile(filename, edges, numNodes);

  if (edges.empty()) {
    std::cerr << "No edges readed" << std::endl;
    return 1;
  }

  std::cout << "Nodes: " << numNodes << " Edges: " << edges.size() << std::endl;

  std::vector<int> h_rowPtr;
  std::vector<int> h_colInd;

  buildRowPtr(edges, numNodes, h_rowPtr);
  buildColInd(edges, h_rowPtr, h_colInd);

  GraphCsr h_graph;
  h_graph.numNodes = numNodes;
  h_graph.numEdges = edges.size();
  h_graph.rowPtr = h_rowPtr.data();
  h_graph.colInd = h_colInd.data();

  std::cout << "CSR format generated" << std::endl;

  //calls to all the kernels ecc

  return 0;
}
