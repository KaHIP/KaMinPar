from __future__ import annotations

import enum
from typing import overload

def seed() -> int:
    """
    Retrieve the current seed of the random number generator.

    Returns:
        int: The current seed value.
    """

def reseed(seed: int) -> None:
    """
    Reseed the random number generator.

    Args:
        seed (int): The new seed value.
    """

class Context:
    """
    A context containing the settings used by KaMinPar.
    """

def context_names() -> list[str]:
    """
    Retrieve all available context names.

    Returns:
        list[str]: A list of available context names.
    """

def context_by_name(name: str) -> Context:
    """
    Retrieve a context by its name.

    Args:
        name (str): The name of the context to retrieve.

    Returns:
        Context: The corresponding context.
    """

def default_context() -> Context:
    """
    Retrieve the default context.
    """

def fast_context() -> Context:
    """
    Retrieve the fast context.
    """

def strong_context() -> Context:
    """
    Retrieve the higher-quality context.
    """

def terapart_context() -> Context:
    """
    Retrieve the default context for memory-efficient partitioning.
    """

def terapart_strong_context() -> Context:
    """
    Retrieve the higher-quality context for memory-efficient partitioning.
    """

def terapart_largek_context() -> Context:
    """
    Retrieve the context for memory-efficient large-k partitioning.
    """

def largek_context() -> Context:
    """
    Retrieve the default context for large-k partitioning.
    """

def largek_fast_context() -> Context:
    """
    Retrieve the fast context for large-k partitioning.
    """

def largek_strong_context() -> Context:
    """
    Retrieve the higher-quality context for large-k partitioning.
    """

class Graph:
    """
    Represents an undirected graph.
    """

    def n(self) -> int:
        """
        Returns the number of nodes in the graph.

        Returns:
            int: The total number of nodes.
        """

    def m(self) -> int:
        """
        Returns the number of edges in the graph.

        Returns:
            int: The total number of edges.
        """

    def is_node_weighted(self) -> bool:
        """
        Checks if the graph has weighted nodes.

        Returns:
            bool: True if nodes have weights, False otherwise.
        """

    def is_edge_weighted(self) -> bool:
        """
        Checks if the graph has weighted edges.

        Returns:
            bool: True if edges have weights, False otherwise.
        """

    def node_weight(self, u: int) -> float:
        """
        Returns the weight of a given node.

        Args:
            u (int): The node ID.

        Returns:
            float: The weight of the node.
        """

    def degree(self, u: int) -> int:
        """
        Returns the degree (number of neighbors) of a given node.

        Args:
            u (int): The node ID.

        Returns:
            int: The degree of the node.
        """

    def neighbors(self, u: int) -> list[(int, int)]:
        """
        Returns a list of neighbors of a given node along with edge weights.

        Args:
            u (int): The node ID.

        Returns:
            list[(int, int)]: A list of tuples where each tuple contains a neighbor ID
                              and the edge weight.
        """

class GraphFileFormat(enum.Enum):
    """
    The format of a graph file.
    """

    METIS = ...
    PARHIP = ...

def load_graph(filename: str, file_format: GraphFileFormat, compress: bool = False) -> Graph:
    """
    Loads a graph from a file.

    Args:
        filename (str): Path to the graph file.
        file_format (GraphFileFormat): The format of the graph file.
        compress (bool, optional): Whether to compress the graph during IO. Defaults to False.

    Returns:
        Graph: The loaded graph.

    Raises:
        ValueError: If the graph file cannot be read or parsed.
    """

class KaMinPar:
    """
    The KaMinPar algorithm.
    """

    def __init__(self, num_threads: int, context: Context) -> None:
        """
        Initializes the KaMinPar algorithm.

        Args:
            num_threads (int): Number of threads to use.
            context (Context): Configuration to use.
        """

    @overload
    def compute_partition(self, graph: Graph, k: int, eps: float) -> list[int]:
        """
        Computes a partition of the given graph into `k` blocks, allowing an imbalance factor `eps`.

        Args:
            graph (Graph): The input graph to be partitioned.
            k (int): Number of block.
            eps (float): Imbalance factor.

        Returns:
            list[int]: A list where each index represents a node and the value represents its
                       assigned partition.
        """

    @overload
    def compute_partition(self, graph: Graph, max_block_weights: list[int]) -> list[int]:
        """
        Computes a partition of the given graph into `len(max_block_weights)` blocks with specified
        maximum block weights.

        Args:
            graph (Graph): The input graph to be partitioned.
            max_block_weights (list[int]): A list of maximum weights allowed for each block.

        Returns:
            list[int]: A list where each index represents a node and the value represents
                       its assigned partition.
        """

    @overload
    def compute_partition(self, graph: Graph, max_block_weight_factors: list[float]) -> list[int]:
        """
        Computes a partition of the given graph into `len(max_block_weights)` blocks with specified
        block weight factors.

        Args:
            graph (Graph): The input graph to be partitioned.
            max_block_weight_factors (list[float]): A list of factors determining the maximum
                                                    allowed weight for each partition.

        Returns:
            list[int]: A list where each index represents a node and the value represents
                       its assigned partition.
        """

def edge_cut(graph: Graph, partition: list[int]) -> int:
    """
    Computes the edge cut of a given graph partition.

    Args:
        graph (Graph): The input graph.
        partition (list[int]): A list where each index represents a node and the value
                               represents its assigned partition.

    Returns:
        int: The total weight of edges that connect different partitions.
    """
