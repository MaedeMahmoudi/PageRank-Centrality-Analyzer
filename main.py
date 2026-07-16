import numpy as np
from src.pagerank_core import PageRank
from src import transition_matrix
from src import visualization
from src import utils
from src import config

def run_pagerank(adj_matrix, method='simple', damping=config.DEFAULT_DAMPING, tol=config.DEFAULT_TOL, max_iter=config.DEFAULT_MAX_ITER):
    if method == 'simple':
        M = transition_matrix.build_simple(adj_matrix)
    elif method == 'weighted':
        M = transition_matrix.build_weighted(adj_matrix)
    elif method == 'smart':
        M = transition_matrix.build_smart(adj_matrix)
    else:
        M = transition_matrix.build_simple(adj_matrix)
    
    pr = PageRank(adj_matrix, damping, tol, max_iter)
    ranks = pr.compute()
    ranking = pr.get_ranking()
    
    return ranks, ranking, pr.history

def load_and_run(filename, method='simple'):
    adj = utils.load_graph_from_file(filename)
    ranks, ranking, history = run_pagerank(adj, method)
    return ranks, ranking, history

def analyze_damping(adj_matrix, d_values=[0.1, 0.3, 0.5, 0.7, 0.85, 0.95]):
    results = []
    for d in d_values:
        pr = PageRank(adj_matrix, damping=d)
        ranks = pr.compute()
        results.append(ranks)
    return d_values, results

if __name__ == "__main__":
    print("="*50)
    print("PageRank Algorithm - Linear Algebra Project")
    print("="*50)
    
    adj = np.array([
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 0]
    ])
    
    pr = PageRank(adj)
    ranks = pr.compute()
    ranking = pr.get_ranking()
    
    print("\nNode Rankings:")
    print("-"*30)
    for node, score in ranking:
        print(f"Node {node}: {score:.6f} ({score*100:.2f}%)")
    
    stats = utils.compute_statistics(adj)
    print("\nGraph Statistics:")
    print("-"*30)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    fig = visualization.plot_convergence(pr.history)
    visualization.save_fig(fig, 'results/convergence.png')
    print("\nConvergence plot saved to: results/convergence.png")
    
    d_values, results = analyze_damping(adj)
    fig = visualization.plot_sensitivity(d_values, results)
    visualization.save_fig(fig, 'results/sensitivity.png')
    print("Sensitivity plot saved to: results/sensitivity.png")
    
    print("\n")
    print("Done! Check the 'results' folder for outputs.")
 