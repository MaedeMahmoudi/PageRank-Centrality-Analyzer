import matplotlib.pyplot as plt
import numpy as np

def plot_convergence(history, title="Convergence of PageRank"):
    history = np.array(history)
    n_nodes = history.shape[1]
    plt.figure(figsize=(10, 6))
    
    # استفاده از مارکرها و خطوط متفاوت
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    colors = plt.cm.tab10(np.linspace(0, 1, n_nodes))
    
    for i in range(n_nodes):
        plt.plot(history[:, i], 
                label=f'Node {i}', 
                linewidth=2,
                marker=markers[i % len(markers)],
                markevery=max(1, len(history)//20),
                color=colors[i])
    
    plt.xlabel('Iterations')
    plt.ylabel('PageRank Score')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_sensitivity(d_values, results, title="Sensitivity to Damping Factor"):
    plt.figure(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    colors = plt.cm.tab10(np.linspace(0, 1, len(results[0])))
    
    for node_idx in range(len(results[0])):
        scores = [r[node_idx] for r in results]
        plt.plot(d_values, scores, 
                marker=markers[node_idx % len(markers)],
                linewidth=2,
                markersize=8,
                label=f'Node {node_idx}',
                color=colors[node_idx])
    
    plt.xlabel('Damping Factor (d)')
    plt.ylabel('PageRank Score')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def save_fig(fig, filename, dpi=300):
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)