import numpy as np

def load_graph_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    adj = np.zeros((n, n))
    for i in range(1, n+1):
        row = list(map(float, lines[i].strip().split()))
        adj[i-1, :] = row
    return adj

def save_ranking_to_file(ranking, filename):
    with open(filename, 'w') as f:
        f.write("Node,Rank,Score\n")
        for node, score in ranking:
            f.write(f"{node},{score:.6f},{score*100:.2f}%\n")

def compute_statistics(adj):
    n = len(adj)
    edges = np.sum(adj > 0)
    density = edges / (n * (n-1)) if n > 1 else 0
    out_deg = np.sum(adj > 0, axis=1)
    in_deg = np.sum(adj > 0, axis=0)
    return {
        'nodes': n,
        'edges': int(edges),
        'density': density,
        'avg_out': np.mean(out_deg),
        'avg_in': np.mean(in_deg),
        'max_out': int(np.max(out_deg)),
        'max_in': int(np.max(in_deg))
    }