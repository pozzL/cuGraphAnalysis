#ifndef GRAPH_H
#define GRAPH_H

struct GraphCsr {
  int numNodes;
  int numEdges;
  int* rowPtr; 
  int* colInd;
};

#endif
