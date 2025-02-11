from __future__ import annotations

from collections.abc import Callable
from functools import total_ordering
from queue import PriorityQueue

import numpy as np

from structure_graph import StructureGraph

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


def dumb_heuristic(current: SimpleGraph, goal: SimpleGraph):
    pass

def transition_function(current: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """
    Returns a list of (cost, result) tuples of possible edit actions given the current graph. The list of actions and their costs are as follows:
    1. Adding an edge (costs the weight of the edge)
    2. Removing an edge (costs the weight of the edge)
    3. Modifying the weight of an edge (costs the absolute difference in weights)
    4. Adding a vertex (costs the weight of the vertex)
    5. Removing a vertex (costs the weight of the vertex and the vertex must be disconnected)
    6. Modifying the weight of a vertex (costs the absolute difference in weights)
    7. Moving the position of a node in the graph (costs 0)

    Returns:
        A list of (cost, result) tuples.
    """

def a_star_graph_edit_distance(start: np.ndarray, goal: np.ndarray, heuristic: Callable[[SimpleGraph, SimpleGraph], float]) -> float:
    queue = PriorityQueue()
