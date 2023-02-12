"""
Bellman-Ford and Floyd-Warshall's algorithms

Here we shall analyse one of the given
algorithm and use them on random generated graphs.
Then we'll compare them to pre-built algorithms and
calculate effectiveness for both interpretations.

Held by:
"""

import random
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, groupby


def gnp_random_connected_graph(num_of_nodes: int,
                               completeness: int,
                               directed: bool = False,
                               draw: bool = False):

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    edges = combinations(range(num_of_nodes), 2)
    G.add_nodes_from(range(num_of_nodes))

    for _, node_edges in groupby(edges, key = lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        if random.random() < 0.5:
            random_edge = random_edge[::-1]
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < completeness:
                G.add_edge(*e)

    for (u,v,w) in G.edges(data=True):
        w['weight'] = random.randint(-5, 20)

    if draw:
        plt.figure(figsize=(10,6))
        if directed:
            # draw with edge weights
            pos = nx.spring_layout(G)
            nx.draw(G,pos, node_color='lightblue',
                    with_labels=True,
                    node_size=500,
                    arrowsize=20,
                    arrows=True)
            labels = nx.get_edge_attributes(G,'weight')
            nx.draw_networkx_edge_labels(G, pos,edge_labels=labels)

        else:
            nx.draw(G, node_color='lightblue',
                with_labels=True,
                node_size=500)

    return G


def bellman_ford(edges: list) -> list:
    """
    Here we made our own interpretation of Bellman-Ford algorithm
    for directed weighted graphs.
    :param edges:
    :return:
    """
    pass


def floyd_warshall(edges: list) -> list:
    """

    :param edges:
    :return:
    """
    pass


if __name__ == '__main__':
    import doctest

    num_of_nodes = random.randint(5, 20)
    completeness = 1
    G = gnp_random_connected_graph(num_of_nodes, completeness, True, True)
    edges = [((u, v), G.get_edge_data(u, v)['weight']) for u, v in G.edges()]
    print()

    print(doctest.testmod())
