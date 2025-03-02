from __future__ import annotations

import enum

def seed() -> int:
    """
    Retrieve the current seed of the random number generator.
    """

def reseed(seed: int) -> None:
    """
    Reseed the random number generator.
    """

class Context:
    """
    A context for the KaMinPar algorithm.
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
    A graph for the KaMinPar algorithm.
    """

    def n(self) -> int:
        """
        Retrieve the number of vertices.
        """

    def m(self) -> int:
        """
        Retrieve the number of edges.
        """

    def is_node_weighted(self) -> bool:
        """
        Check if the graph is node-weighted.
        """

    def is_edge_weighted(self) -> bool:
        """
        Check if the graph is edge-weighted.
        """

    def node_weight(self, u: int) -> float:
        """
        Retrieve the weight of a vertex.
        """

    def degree(self, u: int) -> int:
        """
        Retrieve the degree of a vertex.
        """

class GraphFileFormat(enum.Enum):
    """
    The format of a graph file.
    """

    METIS = ...
    PARHIP = ...

def load_graph(filename: str, file_format: GraphFileFormat) -> Graph:
    """
    Load a graph from a file.
    """

class KaMinPar:
    """
    The KaMinPar algorithm.
    """

    def __init__(self, num_threads: int, context: Context) -> None:
        """
        Initialize the algorithm.
        """

    def compute_partition(self, graph: Graph, k: int, eps: float):
        """
        Compute a partition of a graph.
        """
