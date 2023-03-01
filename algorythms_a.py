"""
Kruskal and Prim's algorithms

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
from typing import List, Tuple
from networkx.algorithms import tree
import time
from tqdm import tqdm


#* Making a graph class to keep all useful things in it ================================================================
class Graph: # <- General graph containing every single node and edge
    def __init__(self, nodes, edges, edges_weight, sorted_weight):
        self.set_of_nodes = {x for x in range(nodes)}
        self.edges = edges
        self.edges_weight = {u: v for u, v in edges_weight}
        self.sorted_weight = {u_1: v_1 for u_1, v_1 in sorted_weight}

        self.adjacency_list = [[] for _ in range(nodes)]
        for (source, destination) in edges:
            self.adjacency_list[source].append(destination)
            self.adjacency_list[destination].append(source)


#* Random graph generation =============================================================================================
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


#* Kruskal's algorithm for random undirected graphs ====================================================================
def kruskal(graph: Graph) -> List[Tuple[int]]:
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


#* Prim's algorithm ====================================================================================================
def cost(graph: Graph, edge):
    """
    Function returns the the weight of an edge.
    """
    return graph.edges_weight[edge]


def min_prims_edge(graph: Graph, visited_nodes) -> tuple:
    """
    Function looks for valid incident edges to each node in visited nodes,
    and then returns the minimum edge.
    """
    incident_edges=[edge for node in visited_nodes for edge in graph.edges if node in edge]
    valid_edges=[]

    for edge in incident_edges:
        if edge[0] not in visited_nodes or edge[1] not in visited_nodes:
            valid_edges.append(edge)

    min_edge = valid_edges[0]
    for edge in valid_edges:
        if cost(graph, edge) < cost(graph, min_edge):
            min_edge = edge

    return min_edge


def prim(graph: Graph) -> str:
    """
    Here we tried to make Prim's algorithm a.k.a Prim-Jarník's algorithm
    for random undirected weighted graphs, where we build a minimum spanning tree (MST)
    for given graphs.

    :param graph: class Graph with a bunch of useful things.
    :return: text with minimum path weight and list of tuples, where each tuple represents an edge presented
    in graph frame.
    """
    visited_nodes = {0}
    minimum_spanning_tree = []

    while len(set(visited_nodes)) != len(set(graph.set_of_nodes)):
        edge = min_prims_edge(graph, visited_nodes)
        visited_nodes.add(edge[0])
        visited_nodes.add(edge[1])
        minimum_spanning_tree.append(edge)
    total_cost = sum(cost(graph, edge) for edge in minimum_spanning_tree)

    return f"Total cost: {total_cost}, MST: {minimum_spanning_tree}"


#* Calculate time ======================================================================================================
def time_diff(func1, func2) -> float:
    """
    Calculate difference in time between two algorithms.
    :param func1: built-in function
    :param func2: our own function
    :return: float
    """

    NUM_OF_ITERATIONS = 500
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


if __name__ == "__main__":

    #* Graph preparations ==============================================================================================
    nodes = random.randint(5, 20)
    complete = 0.2
    G = gnp_random_connected_graph(nodes, complete)

    mstk_kruskal = tree.minimum_spanning_tree(G, algorithm="kruskal") # built-in algorithm
    mstk_prim = tree.minimum_spanning_tree(G, algorithm="prim")  # built-in algorithm

    graph_nodes = nodes
    graph_edges = [x for x in G.edges()]
    graph_edges_weight = [((u, v), G.get_edge_data(u, v)['weight']) for u, v in G.edges()]
    sorted_edges_weight = sorted(graph_edges_weight, key=lambda x: x[1])

    graph = Graph(graph_nodes,
                  graph_edges,
                  graph_edges_weight,
                  sorted_edges_weight)

    #? Uncomment lines below, if you want to see what graph is built of
    # print("Graph nodes: ", graph.set_of_nodes)
    # print()
    # print("Graph edges: ", graph.edges)
    # print()
    # print("Graph edges' weight: ", graph.edges_weight)
    # print()
    # print("Graph sorted edges' weight: ", graph.sorted_weight)

    #* Results 1 =======================================================================================================

    f1 = tree.minimum_spanning_tree(G, algorithm="kruskal")
    f2 = kruskal(graph)

    print("Pre-built Kruskal's algorithm: ", mstk_kruskal.edges())
    print("Our Kruskal's algorithm: ", kruskal(graph))

    print("Time difference: ", time_diff(f1, f2))

    #* Results 2 =======================================================================================================

    f1_2 = tree.minimum_spanning_tree(G, algorithm="prim")
    f2_2 = kruskal(graph)

    print("Pre-built Prim's algorithm: ", mstk_prim.edges())
    print("Our Prim's algorithm: ", prim(graph))

    print("Time difference: ", time_diff(f1_2, f2_2))
