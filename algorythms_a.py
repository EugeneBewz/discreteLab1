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
from typing import List, Tuple
from networkx.algorithms import tree

import time
from tqdm import tqdm


#* Making a graph class to keep all useful things in it ==============================================
class Graph(object): # <- General graph containing every single node and edge
    def __init__(self, nodes, edges, edges_weight, sorted_weight):
        self.set_of_nodes = {x for x in range(nodes)}
        self.edges = edges
        self.edges_weight = {u: v for u, v in edges_weight}
        self.sorted_weight = {u_1: v_1 for u_1, v_1 in sorted_weight}

        self.adjacency_list = [[] for _ in range(nodes)]
        for (source, destination) in edges:
            self.adjacency_list[source].append(destination)
            self.adjacency_list[destination].append(source)


#* Random graph generation ===================================================================
def gnp_random_connected_graph(num_of_nodes: int,
                               completeness: float,
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


#* Kruskal's algorithm for random undirected graphs ================================================
def kruskal(graph: object) -> List[Tuple[int]]:
    """
    Here we tried to make Kruskal's algorithm for random
    undirected weighted graphs, where we build a minimum spanning tree (MST)
    for given graphs.

    :param graph: a class with a bunch of useful things
    :return: a list with edges that make an MST
    """
    kruskal_list = [list(graph.sorted_weight)[0]]
    sorted_edges = [x for x in graph.sorted_weight]
    the_nodes = set(list(graph.sorted_weight)[0])

    while the_nodes != graph.set_of_nodes:
        for ed in sorted_edges[1:]:
            if len(the_nodes.intersection(set(ed))) == 1:
                the_nodes = the_nodes.union(set(ed))
                kruskal_list.append(ed)
    return kruskal_list


#? Not sure whether it will be done ============================================================
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

    #* Graph preparations =====================================================================
    nodes = random.randint(5, 20)
    complete = 0.2
    G = gnp_random_connected_graph(nodes, complete)
    mstk = tree.minimum_spanning_tree(G, algorithm="kruskal") # built-in algorithm

    graph_nodes = nodes
    graph_edges = [x for x in G.edges()]
    graph_edges_weight = [((u, v), G.get_edge_data(u, v)['weight']) for u, v in G.edges()]
    sorted_edges_weight = sorted(graph_edges_weight, key=lambda x: x[1])

    graph = Graph(graph_nodes,
                  graph_edges,
                  graph_edges_weight,
                  sorted_edges_weight)

    # Uncomment lines below, if you want to see what graph is built of
    # print("Graph nodes: ", graph.set_of_nodes)
    # print()
    # print("Graph edges: ", graph.edges)
    # print()
    # print("Graph edges' weight: ", graph.edges_weight)
    # print()
    # print("Graph sorted edges' weight: ", graph.sorted_weight)

    #* Results ==============================================================================
    print("Pre-built algorithm: ", mstk.edges())
    print("Our algorithm: ", kruskal(graph))

    #* Check time for pre-built algorithm =========================================================
    NUM_OF_ITERATIONS = 1000
    time_taken = 0
    for i in tqdm(range(NUM_OF_ITERATIONS)):
        # note that we should not measure time of graph creation
        G = gnp_random_connected_graph(100, 0.4, False)

        start = time.time()
        tree.minimum_spanning_tree(G, algorithm="prim")
        end = time.time()

        time_taken += end - start
    pre_result = time_taken / NUM_OF_ITERATIONS
    print("Pre-built algorithm:", pre_result)

    #* Check time for our algorithm ============================================================
    for i in tqdm(range(NUM_OF_ITERATIONS)):
        # note that we should not measure time of graph creation
        G = gnp_random_connected_graph(100, 0.4, False)

        start = time.time()
        kruskal(graph)
        end = time.time()

        time_taken += end - start
    result = time_taken / NUM_OF_ITERATIONS

    print("Our algorithm:", result)
    print("Difference: ", result - pre_result)
