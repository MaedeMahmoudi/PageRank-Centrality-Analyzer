import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.pagerank_core import PageRank

def test_convergence_simple():
    adj = np.array([
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 0]
    ])
    pr = PageRank(adj, tol=1e-6)
    ranks = pr.compute()
    assert len(pr.history) > 1
    assert np.linalg.norm(pr.history[-1] - pr.history[-2]) < 1e-6
    print("test_convergence_simple passed")

def test_convergence_speed():
    adj = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0]
    ])
    pr = PageRank(adj, tol=1e-8)
    ranks = pr.compute()
    assert pr.iterations < 50
    print("test_convergence_speed passed")

def test_convergence_damping_effect():
    adj = np.array([
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 0]
    ])
    pr1 = PageRank(adj, damping=0.85)
    pr2 = PageRank(adj, damping=0.5)
    ranks1 = pr1.compute()
    ranks2 = pr2.compute()
    assert not np.allclose(ranks1, ranks2)
    print("test_convergence_damping_effect passed")

if __name__ == "__main__":
    test_convergence_simple()
    test_convergence_speed()
    test_convergence_damping_effect()