#ifndef GRAPH_H
#define GRAPH_H

struct GraphCsr {
  int numNodes;
  int numEdges;
  int* rowPtr; 
  int* colInd;
  int* label; //will be used to assign the label for a node with BFS
};

#endif
