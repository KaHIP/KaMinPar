/*******************************************************************************
 * Analyzes the moves made by the FM algorithm to compute per-batch statistics.
 * This is only used for debugging and analyzing the algorithm.
 *
 * @file:   fm_batch_stats.h
 * @author: Daniel Seemaier
 * @date:   27.02.2024
 ******************************************************************************/
#pragma once

#include <vector>

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/refinement/fm/fm_definitions.h"

namespace kaminpar::shm::fm {
class BatchStatsComputator {
  struct Stats {
    NodeID size;
    NodeID max_distance;
    std::vector<NodeID> size_by_distance;
    std::vector<EdgeWeight> gain_by_distance;
  };

public:
  using SeedNodesVec = std::vector<NodeID>;
  using MovesVec = std::vector<fm::AppliedMove>;
  using Batches = tbb::concurrent_vector<std::pair<SeedNodesVec, MovesVec>>;

  BatchStatsComputator(const PartitionedGraph &p_graph) : _p_graph(p_graph) {}

  void track(SeedNodesVec seed_nodes, MovesVec moves);
  void next_iteration();
  void print();

private:
  void print_iteration(int iteration);

  [[nodiscard]] std::vector<Stats>
  compute_batch_stats(const PartitionedGraph &graph, Batches next_batches) const;

  [[nodiscard]] std::pair<PartitionedGraph, Batches>
  build_prev_p_graph(const PartitionedGraph &p_graph, Batches next_batches) const;

  [[nodiscard]] Stats compute_single_batch_stats_in_sequence(
      PartitionedGraph &p_graph,
      const std::vector<NodeID> &seeds,
      const std::vector<fm::AppliedMove> &moves,
      const std::vector<NodeID> &distances
  ) const;

  [[nodiscard]] std::vector<NodeID> compute_batch_distances(
      const Graph &graph,
      const std::vector<NodeID> &seeds,
      const std::vector<fm::AppliedMove> &moves
  ) const;

  const PartitionedGraph &_p_graph;

  Batches _current_iteration_batches;
  std::vector<std::vector<Stats>> _per_iteration_per_batch_stats;
};
} // namespace kaminpar::shm::fm
