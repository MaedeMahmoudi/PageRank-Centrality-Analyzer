import numpy as np

class PageRank:
    def __init__(self, adj_matrix, damping=0.85, tol=1e-8, max_iter=100):
        self.adj = np.array(adj_matrix, dtype=float)
        self.n = len(self.adj)
        self.d = damping
        self.tol = tol
        self.max_iter = max_iter
        self.ranks = None
        self.history = []
        
    def _build_transition(self):
        out_degree = np.sum(self.adj, axis=1)
        M = np.zeros((self.n, self.n))
        for i in range(self.n):
            if out_degree[i] > 0:
                M[i, :] = self.adj[i, :] / out_degree[i]
            else:
                M[i, :] = np.ones(self.n) / self.n
        return M
    
    def compute(self):
        M = self._build_transition()
        r = np.ones(self.n) / self.n
        self.history = [r.copy()]
        
        for _ in range(self.max_iter):
            r_new = self.d * M.T @ r + (1 - self.d) * np.ones(self.n) / self.n
            self.history.append(r_new.copy())
            if np.linalg.norm(r_new - r) < self.tol:
                break
            r = r_new
            
        self.ranks = r
        return r
    
    def get_ranking(self):
        if self.ranks is None:
            self.compute()
        indices = np.argsort(self.ranks)[::-1]
        return [(i, self.ranks[i]) for i in indices]