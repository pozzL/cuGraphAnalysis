# cuGraphAnalysis: Hybrid CPU/GPU Graph Analytics

cuGraphAnalysis is a C++ and CUDA tool designed to calculate the Betweenness Centrality using Brandes' Algorithm, applied to a single source node.

## Hybrid Frontier-Based Execution

The framework uses a frontier-based Breadth-First Search (BFS) rather than a vertex-centric approach. Based on the number of nodes in the current frontier, the algorithm dynamically decides whether to execute the current wave on the GPU or the CPU to maximize efficiency. (if trasporting the data to GPU is more expensive than iterating through the CPU nodes, then execute it on the CPU).

## Algorithm Phases & Formulas

### 1. Forward Phase (Shortest Paths)
Computes shortest path distances ($d$) and the number of shortest paths ($\sigma$) from the source node $s$.

For each unexplored neighbor $w$ of node $v$:
*   If $w$ is visited for the first time ($d(w) = -1$): 
    $$d(w) = d(v) + 1$$
    $$\sigma(w) = \sigma(v)$$
*   If $w$ is visited again in the same wave ($d(w) = d(v) + 1$):
    $$\sigma(w) = \sigma(w) + \sigma(v)$$

### 2. Backward Phase (Credit Redistribution)
Iterates backwards through the BFS waves to accumulate the dependency credits ($\delta$) of the source node on all other nodes. This phase is executed entirely on the GPU and does not benefit from the frontier-based execution (no hybrid approach in this phase, we already have the distances array that divide the levels).

$$\delta(v) = \sum_{w: d(w) = d(v) + 1} \frac{\sigma(v)}{\sigma(w)} \cdot (1 + \delta(w))$$

## Graph Format & Configuration

The graph must be an unweighted edge list (e.g., standard `.txt` files from SNAP datasets (https://snap.stanford.edu/data/)).
1. Place your graph dataset file in the `graphs/` directory.
2. Update the `filename` variable in `src/main.cpp` to point to your dataset (e.g., `const std::string filename = "../graphs/dataset.txt";`).

**Important Node Initialization Warning:** 
Pay strict attention to whether your graph is 0-indexed or 1-indexed. Before running, you must manually update the source node initialization in `cuda/kernels.cu` to match the first existing node in your graph (usually `0` or `1`):
```cpp
h_sigmas[source] = 1.0;
h_distances[source] = 0;
h_frontierCurrent.push_back(source); 
```

## Build & Run

```bash
mkdir build
cd build
cmake ..
make
./cuGraphAnalysis
```
