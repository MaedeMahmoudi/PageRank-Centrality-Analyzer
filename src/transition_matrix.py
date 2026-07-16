import numpy as np

def build_simple(adj):
    n = len(adj)
    out_deg = np.sum(adj, axis=1)
    M = np.zeros((n, n))
    for i in range(n):
        if out_deg[i] > 0:
            M[i, :] = adj[i, :] / out_deg[i]
        else:
            M[i, :] = np.ones(n) / n
    return M

def build_weighted(adj):
    n = len(adj)
    out_sum = np.sum(adj, axis=1)
    M = np.zeros((n, n))
    for i in range(n):
        if out_sum[i] > 0:
            M[i, :] = adj[i, :] / out_sum[i]
        else:
            M[i, :] = np.ones(n) / n
    return M

def build_smart(adj, personalization=None):
    n = len(adj)
    out_deg = np.sum(adj, axis=1)
    M = np.zeros((n, n))
    
    if personalization is None:
        personalization = np.ones(n) / n
    else:
        personalization = np.array(personalization, dtype=float)
        personalization = personalization / np.sum(personalization)
    
    for i in range(n):
        if out_deg[i] > 0:
            M[i, :] = adj[i, :] / out_deg[i]
        else:
            M[i, :] = personalization
    return M