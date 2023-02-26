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
from typing import List
from networkx.algorithms import floyd_warshall_predecessor_and_distance

import time
from tqdm import tqdm


#* Making a graph class to keep all useful things in it ==============================================
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


#* Random graph generation ===================================================================
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


def floyd_warshall(graph: object) -> List[list]:
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
    return my_matrix


def built_in_floyd_warshall(graph: object):
    """
    This is a built-in Floyd-Warshall algorythm, turned
    into a function for comfortable usage.
    :param graph: graph object
    :return: list of dictionaries, where key is source point and values
    are paths to every other node
    """
    try:
        the_list = []
        pred, dist = floyd_warshall_predecessor_and_distance(graph)
        for k, v in dist.items():
            # print(f"Distances with {k} source:", dict(v))
            node_list = [{k: dict(v)}]
            the_list.append(node_list)
        return the_list
    except:
        return "Negative cycle detected"


if __name__ == '__main__':

    # * Graph preparations =====================================================================
    num_of_nodes = random.randint(5, 20)
    # num_of_nodes = 5
    completeness = 1
    G = gnp_random_connected_graph(num_of_nodes, completeness, True, True)
    edges = [((u, v), G.get_edge_data(u, v)['weight']) for u, v in G.edges()]

    graph = Graph(
        num_of_nodes,
        edges
    )

    # * Check time for pre-built algorithm =========================================================
    NUM_OF_ITERATIONS = 700
    time_taken = 0
    for i in tqdm(range(NUM_OF_ITERATIONS)):

        start = time.time()
        built_in_floyd_warshall(graph)
        end = time.time()

        time_taken += end - start
    pre_result = time_taken / NUM_OF_ITERATIONS
    print("Pre-built Floyd-Warshall's algorithm:", pre_result, '\n')

    # * Check time for our algorithm ============================================================
    for i in tqdm(range(NUM_OF_ITERATIONS)):

        start = time.time()
        floyd_warshall(graph)
        end = time.time()

        time_taken += end - start
    result = time_taken / NUM_OF_ITERATIONS

    print("Our Floyd-Warshall's algorithm:", result, '\n')
    print("Difference: ", result - pre_result)

