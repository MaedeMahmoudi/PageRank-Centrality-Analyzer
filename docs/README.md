# PageRank Algorithm Implementation
## Linear Algebra Course Project

### Overview
This project implements the PageRank algorithm for analyzing social networks and web graphs. The algorithm computes node centrality using linear algebra concepts including matrices, eigenvectors, and iterative methods.

### Mathematical Foundation
The PageRank algorithm is based on the formula:

r = d * M^T * r + (1-d) * v / n

where:
- r = rank vector
- M = transition matrix
- d = damping factor (typically 0.85)
- v = personalization vector
- n = number of nodes

### Implementation Features
1. **Simple Transition Matrix**: Standard implementation
2. **Weighted Graph Support**: Handles edge weights
3. **Smart Transition**: Personalization vector support
4. **Damping Analysis**: Sensitivity to damping factor


### Results
The algorithm successfully computes node rankings for various graph types including:
- Simple graphs
- Social networks
- Web graphs
- Weighted graphs

