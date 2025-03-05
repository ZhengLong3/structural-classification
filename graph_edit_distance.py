from __future__ import annotations

import json

from collections.abc import Callable
from functools import total_ordering
from queue import PriorityQueue

import networkx as nx
import numpy as np
import torch

from structure_graph import StructureGraph
import utils

@total_ordering
class PriorityData:
    def __init__(self, gn: float, hn: float, representation) -> PriorityData:
        self.gn = gn
        self.hn = hn
        self.repr = representation

    def __eq__(self, other: PriorityData):
        return self.gn + self.hn == other.gn + other.hn

    def __lt__(self, other: PriorityData):
        return self.gn + self.hn < other.gn + other.hn

class SimpleGraph:
    def __init__(self, node_list: tuple[float], edge_list: dict[tuple[int, int], float]) -> SimpleGraph:
        """
        Creates a simple graph object which contains the state representation of the graph for A* search.

        Parameters:
            node_list: A tuple of floats dictating the weights on the nodes.
            edge_list: A dictionary where the key is the two nodes the edge connects and the value is the weight of the edge.
        """
        self.node_list: tuple[float] = node_list
        self.edge_list: dict[tuple[int, int], float] = edge_list

    def __eq__(self, other: SimpleGraph) -> bool:
        """
        Two SimpleGraphs are equal if the nodes have the same weight in the same order and contain the same edges with the same weights.
        """
        if self.get_num_nodes() != other.get_num_nodes():
            return False
        if self.get_num_edges() != other.get_num_edges():
            return False
        for a, b in zip(sorted(self.node_list), sorted(other.node_list)):
            if abs(a - b) > 0.001:
                return False
        for key in self.edge_list:
            if key not in other.edge_list:
                return False
            if abs(self.edge_list[key] - other.edge_list[key]) > 0.001:
                return False
        return True

    def insert_node(self, value: float) -> SimpleGraph:
        """
        Creates a new SimpleGraph with a new node added to the back of the node_list.

        Parameters:
            value: A float for the weight of the newly added node.

        Returns:
            A new SimpleGraph with the new node added.
        """
        return SimpleGraph(self.node_list + (value,), self.edge_list.copy())

    def check_deletable(self, index: int) -> bool:
        """
        Checks whether a node can be deleted. A node can be deleted only if it has no edges connected to it.

        Parameters:
            index: The position of the node to check.
        """
        for key in self.edge_list:
            if key[0] == index or key[1] == index:
                return False
        return True

    def delete_node(self, index: int) -> SimpleGraph:
        """
        Deletes the node at with position at index. Can only be done if no edges are connected to the deleted node.
        """
        if not self.check_deletable(index):
            return self
        last_index = len(self.node_list) - 1
        new_graph = self.swap_labels(index, last_index)
        new_graph.node_list = new_graph.node_list[:-1]
        return new_graph

    def insert_edge(self, node1: int, node2: int, value: float) -> SimpleGraph:
        """
        Creates a new SimpleGraph with a new edge added.

        Parameters:
            node1: The position of the node in the node_list to add the edge to.
            node2: The position of the second node in the node list to add the edge to.
            value: The weight of the edge to be added.

        Returns:
            A new SimpleGraph with the new edge added.
        """
        new_dict = self.edge_list.copy()
        new_dict[sorted((node1, node2))] = value
        return SimpleGraph(self.node_list, new_dict)

    def delete_edge(self, node1: int, node2: int) -> SimpleGraph:
        """
        Creates a new SimpleGraph with a deleted edge.

        Parameters:
            node1: The position of the first node of the edge in node_list.
            node2: The position of the second node of the edge in node_list.

        Returns:
            A new SimpleGraph with the edge deleted.
        """
        sorted_key = sorted((node1, node2))
        new_set = self.edge_list.copy()
        if sorted_key in new_set:
            del new_set[sorted_key]
        return SimpleGraph(self.node_list, new_set)

    def swap_labels(self, node1: int, node2: int) -> SimpleGraph:
        """
        Creates a new SimpleGraph with the positions of two nodes swapped in the node_list.

        Parameters:
            node1: The position of the first node in the node_list.
            node2: The position of the second node in the node_list.

        Returns:
            A new SimpleGraph with the node positions swapped (including the edge_list).
        """
        list_node_list = list(self.node_list)
        list_node_list[node1], list_node_list[node2] = list_node_list[node2], list_node_list[node1]
        new_dict = {}
        for key in self.edge_list:
            key_copy = list(key)
            for i in range(2):
                if key_copy[i] == node1:
                    key_copy[i] = node2
                elif key_copy[i] == node2:
                    key_copy[i] = node1
            new_dict[sorted(tuple(key_copy))] = self.edge_list[key]
        return SimpleGraph(tuple(list_node_list), new_dict)

    def get_num_nodes(self) -> int:
        return len(self.node_list)

    def get_num_edges(self) -> int:
        return len(self.edge_list)


def dumb_heuristic(current: SimpleGraph, goal: SimpleGraph):
    pass

def transition_function(current: SimpleGraph, goal: SimpleGraph) -> list[tuple[float, SimpleGraph]]:
    """
    Returns a list of (cost, result) tuples of possible edit actions given the current graph. The list of actions and their costs are as follows:
    1. Adding an edge (costs the weight of the edge)
    2. Removing an edge (costs the weight of the edge)
    3. Modifying the weight of an edge (costs the absolute difference in weights)
    4. Adding a vertex (costs the weight of the vertex)
    5. Removing a vertex (costs the weight of the vertex and the vertex must be disconnected)
    6. Modifying the weight of a vertex (costs the absolute difference in weights)
    7. Moving the position of a node in the graph (costs 0) -> Might end up not allowing this

    Parameters:
        current: The current SimpleGraph to generate children of.
        goal: The final SimpleGraph to reach.

    Returns:
        A list of (cost, result) tuples.
    """

    actions = []

    # # Shuffling the order of the nodes
    # for i in range(current.get_num_nodes() - 1):
    #     for j in range(i + 1, current.get_num_nodes()):
    #         actions.append((0, current.swap_labels(i, j)))

    # Consider adding nodes if there are more nodes in the goal graph compared to the current graph.
    if current.get_num_nodes() < goal.get_num_nodes():
        for value in goal.node_list:
            actions.append((value, current.insert_node(value)))
    
    # Consider deleting nodes if there are more nodes in the current graph compared to the goal graph.
    if current.get_num_nodes() > goal.get_num_nodes():
        for i in range(current.get_num_nodes()):
            actions.append((current.node_list[i], current.delete_node(i)))


    actions = []

    # # Shuffling the order of the nodes
    # for i in range(current.get_num_nodes() - 1):
    #     for j in range(i + 1, current.get_num_nodes()):
    #         actions.append((0, current.swap_labels(i, j)))

    # Consider adding nodes if there are more nodes in the goal graph compared to the current graph.
    if current.get_num_nodes() < goal.get_num_nodes():
        for value in goal.node_list:
            actions.append((value, current.insert_node(value)))
    
    # Consider deleting nodes if there are more nodes in the current graph compared to the goal graph.
    if current.get_num_nodes() > goal.get_num_nodes():
        for i in range(current.get_num_nodes()):
            actions.append((current.node_list[i], current.delete_node(i)))


def a_star_graph_edit_distance(start: np.ndarray, goal: np.ndarray, heuristic: Callable[[SimpleGraph, SimpleGraph], float]) -> float:
    queue = PriorityQueue()

def node_subst_cost(node1, node2):
    return abs(node1["area"] - node2["area"])

def node_ins_del_cost(node):
    return node["area"]

def edge_subst_cost(edge1, edge2):
    return abs(edge1["angle"] - edge2["angle"])

def edge_ins_del_cost(edge):
    return edge["angle"]


if __name__ == "__main__":
    structures = {}
    with open("data/structures.json", "r") as f:
        structure_list = json.load(f)
    for structure in structure_list:
        structures[structure["name"]] = structure["rects"]
    # graph1 = StructureGraph.create_from_node_list(map(lambda x: PlaneNode.create_from_vectors(*x), structures["House"]))
    # graph2 = StructureGraph.create_from_node_list(map(lambda x: PlaneNode.create_from_vectors(*x), structures["cube"]))
    # print(nx.graph_edit_distance(graph1.get_simple_graph(), graph1.get_simple_graph(), node_subst_cost=node_subst_cost, node_del_cost=node_ins_del_cost, node_ins_cost=node_ins_del_cost, edge_subst_cost=edge_subst_cost, edge_ins_cost=edge_ins_del_cost, edge_del_cost=edge_ins_del_cost))
    name_list = ["big_square", "small_square", "rectangle", "slanted_big_square", "prism", "3plane_atom", "cleaned_tetrahedron", "square_pyramid", "grid_atom", "skewed_grid_atom", "pentagon", "house", "kite", "triangular_bipyramid", "tent", "kite_prism"]
    graph_list = list(map(lambda x: StructureGraph.create_from_vector_list(structures[x]), name_list))
    array = torch.zeros((len(name_list), len(name_list)))
    for i in range(len(name_list)):
        for j in range(len(name_list)):
            array[i, j] = nx.graph_edit_distance(graph_list[i].get_simple_graph(), graph_list[j].get_simple_graph(), node_subst_cost=node_subst_cost, node_del_cost=node_ins_del_cost, node_ins_cost=node_ins_del_cost, edge_subst_cost=edge_subst_cost, edge_ins_cost=edge_ins_del_cost, edge_del_cost=edge_ins_del_cost)
    utils.tensor_to_csv(array, "./output/geds.csv")