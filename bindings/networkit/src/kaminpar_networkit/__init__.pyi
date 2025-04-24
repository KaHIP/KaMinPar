from __future__ import annotations

import networkit

class KaMinPar:
    """
    The KaMinPar algorithm.
    """

    def __init__(self, graph: networkit.Graph) -> None:
        """
        Initializes the KaMinPar instance with a given graph.

        Args:
            graph (networkit.Graph): The input graph to be partitioned.

        Raises:
            ValueError: If the provided graph is directed.
        """

    def loadGraph(self, graph: networkit.Graph) -> None:
        """
        Loads a new graph into the KaMinPar instance.

        Args:
            graph (networkit.Graph): The graph to be loaded.

        Raises:
            ValueError: If the provided graph is directed.
        """

    def computePartition(self, k: int) -> networkit.Partition:
        """
        Computes a partition of the graph into `k` blocks, allowing an imbalance factor of 3%.

        Args:
            k (int): The number of blocks to partition the graph into.

        Returns:
            networkit.Partition: The computed partition.
        """

    def computePartitionWithEpsilon(self, k: int, eps: float) -> networkit.Partition:
        """
        Computes a partition of the graph into `k` blocks, allowing an imbalance factor of `eps`.

        Args:
            k (int): The number of blocks to partition the graph into.
            eps (float): The allowed imbalance factor.

        Returns:
            networkit.Partition: The computed partition.
        """

    def computePartitionWithFactors(
        self, max_block_weight_factors: list[float]
    ) -> networkit.Partition:
        """
        Computes a partition of the graph into `len(max_block_weight_factors)` blocks,
        using the specified block weight factors.

        Args:
            max_block_weight_factors (list[float]): A list of factors that determine the allowed
                                                    block weights.

        Returns:
            networkit.Partition: The computed partition.
        """

    def computePartitionWithWeights(self, max_block_weights: list[int]) -> networkit.Partition:
        """
        Computes a partition of the graph into `len(max_block_weights)` blocks,
        with specified maximum block weights.

        Args:
            max_block_weights (list[int]): A list of maximum weights for each block.

        Returns:
            networkit.Partition: The computed partition.
        """
