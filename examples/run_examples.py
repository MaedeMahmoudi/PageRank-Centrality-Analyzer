import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pagerank_core import PageRank
from src import utils
from src import visualization
from src import config

def run_all_examples():
    example_files = {
        'Simple Graph': 'examples/simple_graph.txt',
        'Social Network': 'examples/social_network.txt',
        'Web Graph': 'examples/web_graph.txt',
        'Weighted Graph': 'examples/weighted_graph.txt'
    }
    
    results = {}
    for name, path in example_files.items():
        adj = utils.load_graph_from_file(path)
        pr = PageRank(adj)
        ranks = pr.compute()
        results[name] = {
            'adj': adj,
            'ranks': ranks,
            'history': pr.history,
            'ranking': pr.get_ranking()
        }
    
    return results

if __name__ == "__main__":
    run_all_examples()