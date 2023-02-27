"""
Bellman-Ford and Floyd-Warshall's algorithms

Here we shall analyse one of the given
algorithm and use them on random generated graphs.
Then we'll compare them to pre-built algorithms and
calculate effectiveness for both interpretations.

Held by: Yevhenii Bevz / Khrystyna Mysak
"""

import random
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, groupby
from typing import List
from networkx.algorithms import floyd_warshall_predecessor_and_distance
from networkx.algorithms import bellman_ford_predecessor_and_distance


import time
from tqdm import tqdm


#* Making a graph class to keep all useful things in it ================================================================
class Graph(object): # <- General graph containing every single node and edge
    def __init__(self, nodes, edges):
        self.set_of_nodes = {x for x in range(nodes)}
        self.edges = edges

        self.matrix = []
        for x in range(num_of_nodes):
            matrix_row = []
            for y in range(num_of_nodes):
                matrix_row += [float("inf")]
            self.matrix += [matrix_row]

        for element in edges:
            self.matrix[element[0][0]][element[0][1]] = element[1]


#* Random graph generation =============================================================================================
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


#* Bellman-Ford's algorithm for random directed graphs =================================================================
def bellman_ford(graph: object, starting_node: int = 0) -> list or str:
    """
    Here we made our own implementation of Bellman-Ford algorithm
    for directed weighted graphs. The function also detects negative weight cycles.
    :param graph: a class with a bunch of useful things
    :return: a list of shortest distances from src (0 as default) to all other vertices
    """
    # assign all distances to infinity, except source vertex
    dist = [float("Inf")] * len(graph.set_of_nodes)
    dist[starting_node] = 0
    # main iterations (relaxing all edges) to find shortest distances from src to all other vertices
    for _ in range(len(graph.set_of_nodes) - 1):
        for edge, weight in graph.edges:
            # reassigning distance values and parent index of the adjacent vertices of the picked vertex
            if dist[edge[0]] != float("Inf") and dist[edge[0]] + weight < dist[edge[1]]:
                dist[edge[1]] = dist[edge[0]] + weight
    # the last, len(graph.set_of_nodes)-th iteration to check for negative cycles
    for edge, weight in graph.edges:
        if dist[edge[0]] != float("Inf") and dist[edge[0]] + weight < dist[edge[1]]:
            return "Graph contains negative weight cycle!"
    return dist


def built_in_bellman_ford():
    """
    This is a built-in Floyd-Warshall algorythm, turned
    into a function for comfortable usage.
    :return: list of dictionaries, where key is source point and values
    are paths to every other node
    """
    # pred is a dictionary of predecessors, dist is a dictionary of distances
    try:
        pred, dist = bellman_ford_predecessor_and_distance(G, 0)
        for k, v in dist.items():
            print(f"Distance to {k}:", v)
    except:
        return "Negative cycle detected"


#* Floyd-Warshall's algorithm for random directed graphs ===============================================================
def floyd_warshall(graph: object) -> List[list] or str:
    """
    Here we tried to make our own interpretation of Floyd-Warshall
    algorythm for weighted directed graphs. The algorythm uses
    matrix of weights and iterates through it N times, where N is the
    number of nodes.
    :param graph: graph objects with a bunch of useful things
    :return: nested list with minimum path for each pair of nodes.
    """
    my_matrix = graph.matrix
    for k in range(len(my_matrix)):
        for i in range(len(my_matrix)):
            for j in range(len(my_matrix)):
                my_matrix[i][j] = min(my_matrix[i][j], my_matrix[i][k] + my_matrix[k][j])

    for x in range(num_of_nodes):
        if my_matrix[x][x] < 0:
            return "Negative cycle detected"

    return my_matrix


def built_in_floyd_warshall() -> List[List[dict]] or str:
    """
    This is a built-in Floyd-Warshall algorythm, turned
    into a function for comfortable usage.
    :return: list of dictionaries, where key is source point and values
    are paths to every other node
    """
    try:
        the_list = []
        pred, dist = floyd_warshall_predecessor_and_distance(G)
        for k, v in dist.items():
            # print(f"Distances with {k} source:", dict(v))
            node_list = [{k: dict(v)}]
            the_list.append(node_list)
        return the_list
    except:
        return "Negative cycle detected"


#* Calculate time ======================================================================================================
def time_diff(func1, func2) -> float:
    """
    Calculate difference in time between two algorithms.
    :param func1: built-in function
    :param func2: our own function
    :return: float
    """

    NUM_OF_ITERATIONS = 1000
    time_taken_1, time_taken_2 = 0, 0
    for i in tqdm(range(NUM_OF_ITERATIONS)):
        # note that we should not measure time of graph creation
        G = gnp_random_connected_graph(100, 0.4, False)

        # Measure time of built-in algorithm
        start1 = time.time()
        func1
        end1 = time.time()

        time_taken_1 += end1 - start1

        # Measure time of our algorithm
        start2 = time.time()
        func2
        end2 = time.time()

        time_taken_2 += end2 - start2

    return time_taken_2 - time_taken_1


if __name__ == '__main__':

    # * Graph preparations =============================================================================================
    num_of_nodes = random.randint(5, 20)
    completeness = 1
    G = gnp_random_connected_graph(num_of_nodes, completeness, True, True)
    edges = [((u, v), G.get_edge_data(u, v)['weight']) for u, v in G.edges()]

    graph = Graph(
        num_of_nodes,
        edges
    )

    #? Uncomment lines below, if you want to see what graph is built of
    # print("Graph nodes: ", graph.set_of_nodes)
    # print()
    # print("Graph edges: ", graph.edges)
    # print()
    # print("Graph matrix of weight: ", graph.matrix)

    # * Results 1 ======================================================================================================

    f1 = built_in_floyd_warshall()
    f2 = floyd_warshall(graph)

    print("Pre-built algorithm: ", f1)
    print("Our algorithm: ", f2)

    print("Time difference: ", time_diff(f1, f2))

    # * Results 2 ======================================================================================================

    f1_2 = built_in_bellman_ford()
    f2_2 = bellman_ford(graph)

    print("Pre-built algorithm: ", f1_2)
    print("Our algorithm: ", f2_2)

    print("Time difference: ", time_diff(f1_2, f2_2))
