/*******************************************************************************
 * Analyzes the moves made by the FM algorithm to compute per-batch statistics.
 * This is only used for debugging and analyzing the algorithm.
 *
 * @file:   fm_batch_stats.cc
 * @author: Daniel Seemaier
 * @date:   27.02.2024
 ******************************************************************************/
#include "kaminpar-shm/refinement/fm/fm_batch_stats.h"

#include <queue>
#include <set>

#include "kaminpar-shm/metrics.h"

#include "kaminpar-common/logger.h"

namespace kaminpar::shm::fm {
void BatchStatsComputator::track(SeedNodesVec seed_nodes, MovesVec moves) {
  _current_iteration_batches.emplace_back(std::move(seed_nodes), std::move(moves));
}

void BatchStatsComputator::next_iteration() {
  _per_iteration_per_batch_stats.push_back(
      compute_batch_stats(_p_graph, std::move(_current_iteration_batches))
  );
}

void BatchStatsComputator::print() {
  LOG_STATS << "Batches: [STATS:FM:BATCHES]";
  for (std::size_t i = 0; i < _per_iteration_per_batch_stats.size(); ++i) {
    if (!_per_iteration_per_batch_stats[i].empty()) {
      LOG_STATS << "  * Iteration " << (i + 1) << ":";
      print_iteration(i);
    }
  }
}

void BatchStatsComputator::print_iteration(const int iteration) {
  const auto &stats = _per_iteration_per_batch_stats[iteration];

  const NodeID max_distance =
      std::max_element(stats.begin(), stats.end(), [&](const auto &lhs, const auto &rhs) {
        return lhs.max_distance < rhs.max_distance;
      })->max_distance;

  std::vector<NodeID> total_size_by_distance(max_distance + 1);
  std::vector<EdgeWeight> total_gain_by_distance(max_distance + 1);
  for (NodeID distance = 0; distance <= max_distance; ++distance) { // <=
    for (const auto &batch_stats : stats) {
      if (distance < batch_stats.size_by_distance.size()) {
        KASSERT(distance < batch_stats.gain_by_distance.size());
        total_size_by_distance[distance] += batch_stats.size_by_distance[distance];
        total_gain_by_distance[distance] += batch_stats.gain_by_distance[distance];
      }
    }
  }

  LOG_STATS << "    - Max distance: " << max_distance << " [STATS:FM:BATCHES:" << iteration << "]";
  std::stringstream size_ss, gain_ss;
  size_ss << "      + Size by distance: " << total_size_by_distance[0];
  gain_ss << "      + Gain by distance: " << total_gain_by_distance[0];

  for (NodeID distance = 1; distance <= max_distance; ++distance) { // <=
    size_ss << "," << total_size_by_distance[distance];
    gain_ss << "," << total_gain_by_distance[distance];
  }
  LOG_STATS << size_ss.str() << " [STATS:FM:BATCHES:" << iteration << "]";
  LOG_STATS << gain_ss.str() << " [STATS:FM:BATCHES:" << iteration << "]";
}

auto BatchStatsComputator::compute_batch_stats(
    const PartitionedGraph &next_p_graph, Batches next_batches
) const -> std::vector<Stats> {
  // Rollback the partition to *before* any moves of the batch were applied
  // prev_batches will now contain the *target* block for all nodes instead of their previous block
  auto [prev_p_graph, prev_batches] = build_prev_p_graph(next_p_graph, std::move(next_batches));

  std::vector<std::vector<NodeID>> batch_distances(prev_batches.size());
  tbb::parallel_for<std::size_t>(
      0,
      prev_batches.size(),
      [&, &prev_p_graph = prev_p_graph, &prev_batches = prev_batches](std::size_t i) {
        const auto &[seeds, moves] = prev_batches[i];
        if (!moves.empty()) {
          batch_distances[i] = compute_batch_distances(prev_p_graph.graph(), seeds, moves);
        }
      }
  );

  // In the recorded sequence, re-apply batches batch-by-batch to measure their effect on partition
  // quality
  std::vector<Stats> batch_stats;
  for (std::size_t i = 0; i < prev_batches.size(); ++i) {
    const auto &[seeds, moves] = prev_batches[i];
    const auto &distances = batch_distances[i];

    if (!moves.empty()) {
      batch_stats.push_back(
          compute_single_batch_stats_in_sequence(prev_p_graph, seeds, moves, distances)
      );
    } else {
      batch_stats.emplace_back();
    }
  }

  // If everything went right, we should now have the same partition as next_partition
  KASSERT(
      metrics::edge_cut_seq(prev_p_graph) == metrics::edge_cut(next_p_graph), "", assert::always
  );
  KASSERT(metrics::imbalance(prev_p_graph) == metrics::imbalance(next_p_graph), "", assert::always);

  return batch_stats;
}

// Computes the partition *before* any moves of the given batches where applied to it
// Changes the batches to store the blocks to which the nodes where moved to
auto BatchStatsComputator::build_prev_p_graph(const PartitionedGraph &p_graph, Batches batches)
    const -> std::pair<PartitionedGraph, Batches> {
  StaticArray<BlockID> prev_partition(p_graph.n());
  auto &next_partition = p_graph.raw_partition();
  std::copy(next_partition.begin(), next_partition.end(), prev_partition.begin());

  // Rollback partition to before the moves in the batches where applied
  // Update the batches to store the "new" from block
  for (auto &[seeds, moves] : batches) {
    for (auto &move : moves) {
      std::swap(prev_partition[move.node], move.from);
    }
  }

  return {
      PartitionedGraph(p_graph.graph(), p_graph.k(), std::move(prev_partition)),
      std::move(batches),
  };
}

// Computes the statistics for a single batch
// The given partition should reflect all batches that came before this one, but none of the ones
// that will come afterwards
// This function also applies the moves of the current batch to the given partition
auto BatchStatsComputator::compute_single_batch_stats_in_sequence(
    PartitionedGraph &p_graph,
    const std::vector<NodeID> &seeds,
    const std::vector<fm::AppliedMove> &moves,
    const std::vector<NodeID> &distances
) const -> Stats {
  KASSERT(!seeds.empty());
  KASSERT(!moves.empty());
  Stats stats;

  stats.size = moves.size();
  stats.max_distance = *std::max_element(distances.begin(), distances.end());
  stats.size_by_distance.resize(stats.max_distance + 1);
  stats.gain_by_distance.resize(stats.max_distance + 1);

  NodeID cur_distance = 0;

  EdgeWeight gain_for_next_improvement = 0;
  NodeID size_for_next_improvement = 0;

  for (std::size_t i = 0; i < moves.size(); ++i) {
    const auto &[u, block, improvement] = moves[i];

    // Compute the gain of the move
    EdgeWeight int_degree = 0;
    EdgeWeight ext_degree = 0;
    for (const auto &[e, v] : p_graph.neighbors(u)) {
      if (p_graph.block(v) == p_graph.block(u)) {
        int_degree += p_graph.edge_weight(e);
      } else if (p_graph.block(v) == block) {
        ext_degree += p_graph.edge_weight(e);
      }
    }

    KASSERT(i < distances.size());
    cur_distance = std::max(cur_distance, distances[i]);

    gain_for_next_improvement += ext_degree - int_degree;
    size_for_next_improvement += 1;

    if (improvement) {
      stats.gain_by_distance[cur_distance] += gain_for_next_improvement;
      stats.size_by_distance[cur_distance] += size_for_next_improvement;
      gain_for_next_improvement = 0;
      size_for_next_improvement = 0;
    }

    p_graph.set_block(u, block);
  }

  return stats;
}

std::vector<NodeID> BatchStatsComputator::compute_batch_distances(
    const Graph &graph, const std::vector<NodeID> &seeds, const std::vector<fm::AppliedMove> &moves
) const {
  // Keeps track of moved nodes that we yet have to discover
  std::unordered_map<NodeID, std::size_t> searched;
  for (std::size_t i = 0; i < moves.size(); ++i) {
    searched[moves[i].node] = i;
  }

  // Keep track of nodes that we have already discovered
  std::set<NodeID> visited;

  // Current frontier of the BFS
  std::queue<NodeID> frontier;
  for (const NodeID &seed : seeds) {
    frontier.push(seed);
    visited.insert(seed);
  }

  NodeID current_distance = 0;
  NodeID current_layer_size = frontier.size();
  std::vector<NodeID> distances(moves.size());

  while (!searched.empty()) {
    KASSERT(!frontier.empty());

    if (current_layer_size == 0) {
      ++current_distance;
      current_layer_size = frontier.size();
    }

    const NodeID u = frontier.front();
    frontier.pop();
    --current_layer_size;

    // If the node was moved, record its distance from any seed node
    if (auto it = searched.find(u); it != searched.end()) {
      distances[it->second] = current_distance;
      searched.erase(it);
    }

    // Expand search to its neighbors
    for (const auto &[e, v] : graph.neighbors(u)) {
      if (visited.count(v) == 0) {
        visited.insert(v);
        frontier.push(v);
      }
    }
  }

  return distances;
}
} // namespace kaminpar::shm::fm
