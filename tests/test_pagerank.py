import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.pagerank_core import PageRank
from src import transition_matrix

def test_simple_graph():
    adj = np.array([
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 0]
    ])
    pr = PageRank(adj)
    ranks = pr.compute()
    assert len(ranks) == 3
    assert abs(np.sum(ranks) - 1.0) < 1e-10
    print("test_simple_graph passed")

def test_transition_matrix():
    adj = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    M = transition_matrix.build_simple(adj)
    assert M.shape == (3, 3)
    assert np.allclose(np.sum(M, axis=1), 1.0)
    print("test_transition_matrix passed")

def test_dangling_nodes():
    adj = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [1, 0, 0]
    ])
    pr = PageRank(adj)
    ranks = pr.compute()
    assert len(ranks) == 3
    assert np.sum(ranks) > 0
    print("test_dangling_nodes passed")

if __name__ == "__main__":
    test_simple_graph()
    test_transition_matrix()
    test_dangling_nodes()