# PageRank Algorithm Description

## Core Concept
PageRank assigns importance scores to nodes in a graph based on the structure of incoming links. A node is important if other important nodes link to it.

## Algorithm Steps

### 1. Input
- Adjacency matrix A (n x n)
- Damping factor d (default: 0.85)
- Tolerance ε (default: 1e-8)

### 2. Build Transition Matrix
For each node i:
- If out-degree > 0: M[i,j] = A[i,j] / out-degree[i]
- If out-degree = 0: M[i,j] = 1/n (dangling node)

### 3. Initialize Rank Vector
r = [1/n, 1/n, ..., 1/n]

### 4. Iterate Until Convergence
r_new = d * M^T * r + (1-d) * v / n

Stop when: ||r_new - r|| < ε

### 5. Output
Sorted nodes by rank value (descending)

## Mathematical Properties

### Stochastic Matrix
Each row of M sums to 1, making it a stochastic matrix.

### Eigenvector Interpretation
The converged rank vector r is the dominant eigenvector of:
M' = d * M^T + (1-d) * E / n
where E is the all-ones matrix.

### Convergence Guarantee
The algorithm always converges for d < 1 due to the Perron-Frobenius theorem.

## Variants

### Weighted PageRank
Edge weights represent link importance or user interaction frequency.

### Personalized PageRank
Uses custom personalization vector v to bias rankings toward specific nodes.

### Adaptive PageRank
Transition matrix updates based on current rank values during iteration.