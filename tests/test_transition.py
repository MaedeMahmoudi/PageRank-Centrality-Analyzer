import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src import transition_matrix

def test_build_simple():
    adj = np.array([
        [0, 1, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])
    M = transition_matrix.build_simple(adj)
    expected = np.array([
        [0, 0.5, 0.5],
        [1, 0, 0],
        [0, 1, 0]
    ])
    assert np.allclose(M, expected)
    print("test_build_simple passed")

def test_build_weighted():
    adj = np.array([
        [0, 0.7, 0.3],
        [0.5, 0, 0.5],
        [0.8, 0.2, 0]
    ])
    M = transition_matrix.build_weighted(adj)
    assert M.shape == (3, 3)
    assert np.allclose(np.sum(M, axis=1), 1.0)
    print("test_build_weighted passed")

def test_build_smart():
    adj = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [1, 0, 0]
    ])
    personalization = np.array([0.5, 0.3, 0.2])
    M = transition_matrix.build_smart(adj, personalization)
    assert M.shape == (3, 3)
    assert np.allclose(np.sum(M, axis=1), 1.0)
    print("test_build_smart passed")

if __name__ == "__main__":
    test_build_simple()
    test_build_weighted()
    test_build_smart()