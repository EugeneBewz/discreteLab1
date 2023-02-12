"""
Kruskal and Prim's algorithms

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


def kruskal_algorithm(edges: list) -> list:
    """
    Here we tried to make Kruskal's algorithm for random
    undirected weighted graphs, where we build a minimum spanning tree (MST)
    for given graphs.

    :param edges: list of tuples, where elem1 is tuple with two integers
    (graph edge) and elem2 is edge's weight.
    :return: list of tuples, where each tuple represents an edge presented
    in graph frame.
    """
    pass


def prim_algorithm(edges: list) -> list:
    """
    Here we tried to make Prim's algorithm a.k.a Prim-Jarník's algorithm
    for random undirected weighted graphs, where we build a minimum spanning tree (MST)
    for given graphs.

    :param edges: list of tuples, where elem1 is tuple with two integers
    (graph edge) and elem2 is edge's weight.
    :return: list of tuples, where each tuple represents an edge presented
    in graph frame.
    """
    pass


if __name__ == "__main__":
    import doctest

    num_of_nodes = random.randint(5, 20)
    completeness = 1
    # directed = True
    # draw = True
    G = gnp_random_connected_graph(num_of_nodes, completeness)
    list_of_edges = [((u, v), G.get_edge_data(u, v)['weight']) for u, v in G.edges()]
    print(kruskal_algorithm(list_of_edges))

    print(doctest.testmod())
