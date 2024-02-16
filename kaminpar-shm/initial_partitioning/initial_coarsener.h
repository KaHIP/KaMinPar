/*******************************************************************************
 * @file:   initial_coarsener.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Sequential coarsener based on label propagation with leader
 * locking.
 ******************************************************************************/
#pragma once

#include <functional>
#include <utility>

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/definitions.h"
#include "kaminpar-shm/initial_partitioning/sequential_graph_hierarchy.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/fast_reset_array.h"
#include "kaminpar-common/random.h"

#define STATIC_MAX_CLUSTER_WEIGHT(x)                                                               \
  [&](const NodeID) {                                                                              \
    return x;                                                                                      \
  }

namespace kaminpar::shm::ip {
class InitialCoarsener {
  static constexpr auto kChunkSize = 256;
  static constexpr auto kNumberOfNodePermutations = 16;

  using ContractionResult = std::pair<Graph, std::vector<NodeID>>;

public:
  struct Cluster {
    bool locked : 1; // use bit from weight so that the struct is 8 bytes wide
                     // instead of 12
    NodeWeight weight : std::numeric_limits<NodeWeight>::digits - 1;
    NodeID leader;
  };
  static_assert(
      sizeof(NodeWeight) != sizeof(NodeID) || sizeof(Cluster) == sizeof(NodeWeight) + sizeof(NodeID)
  );

  struct MemoryContext {
    std::vector<Cluster> clustering{};
    std::vector<NodeID> cluster_sizes{};
    std::vector<NodeID> leader_node_mapping{};
    FastResetArray<EdgeWeight> rating_map{};
    FastResetArray<EdgeWeight> edge_weight_collector{};
    std::vector<NodeID> cluster_nodes{};

    [[nodiscard]] std::size_t memory_in_kb() const {
      return clustering.size() * sizeof(Cluster) / 1000 +         //
             cluster_sizes.size() * sizeof(NodeID) / 1000 +       //
             leader_node_mapping.size() * sizeof(NodeID) / 1000 + //
             rating_map.memory_in_kb() +                          //
             edge_weight_collector.memory_in_kb() +               //
             cluster_nodes.size() * sizeof(NodeID) / 1000;        //
    }
  };

  InitialCoarsener(
      const Graph *graph, const InitialCoarseningContext &c_ctx, MemoryContext &&m_ctx
  );

  InitialCoarsener(const Graph *graph, const InitialCoarseningContext &c_ctx);

  InitialCoarsener(const InitialCoarsener &) = delete;
  InitialCoarsener &operator=(const InitialCoarsener &) = delete;

  InitialCoarsener(InitialCoarsener &&) noexcept = delete;
  InitialCoarsener &operator=(InitialCoarsener &&) = delete;

  [[nodiscard]] inline std::size_t size() const {
    return _hierarchy.size();
  }

  [[nodiscard]] inline bool empty() const {
    return size() == 0;
  }

  [[nodiscard]] inline const Graph *coarsest_graph() const {
    return &_hierarchy.coarsest_graph();
  }

  const Graph *coarsen(const std::function<NodeWeight(NodeID)> &cb_max_cluster_weight);

  PartitionedGraph uncoarsen(PartitionedGraph &&c_p_graph);

  MemoryContext free();

  void reset_current_clustering() {
    if (_current_graph->node_weighted()) {
      reset_current_clustering(_current_graph->n(), _current_graph->raw_node_weights());
    } else {
      // this is robust if _current_graph is empty (then we can't use
      // node_weight(0))
      reset_current_clustering_unweighted(
          _current_graph->n(), _current_graph->total_node_weight() / _current_graph->n()
      );
    }
  }

  template <typename Weights>
  void reset_current_clustering(const NodeID n, const Weights &node_weights) {
    KASSERT(n <= _clustering.size());
    KASSERT(n <= node_weights.size());

    _current_num_moves = 0;
    for (NodeID u = 0; u < n; ++u) {
      _clustering[u].locked = false;
      _clustering[u].leader = u;
      _clustering[u].weight = node_weights[u];
    }
  }

  void reset_current_clustering_unweighted(const NodeID n, const NodeWeight unit_node_weight) {
    _current_num_moves = 0;
    for (NodeID u = 0; u < n; ++u) {
      _clustering[u].locked = false;
      _clustering[u].leader = u;
      _clustering[u].weight = unit_node_weight;
    }
  }

  void handle_node(NodeID u, NodeWeight max_cluster_weight);
  NodeID pick_cluster(NodeID u, NodeWeight u_weight, NodeWeight max_cluster_weight);
  NodeID pick_cluster_from_rating_map(NodeID u, NodeWeight u_weight, NodeWeight max_cluster_weight);

private:
  ContractionResult contract_current_clustering();

  void perform_label_propagation(NodeWeight max_cluster_weight);

  //
  // Interleaved label propagation: compute for the next coarse graph while
  // constructing it
  //
  inline void interleaved_handle_node(const NodeID c_u, const NodeWeight c_u_weight) {
    if (!_interleaved_locked) {
      const NodeID best_cluster{
          pick_cluster_from_rating_map(c_u, c_u_weight, _interleaved_max_cluster_weight)};
      const bool changed_cluster{best_cluster != c_u};

      if (changed_cluster) {
        ++_current_num_moves;
        _clustering[c_u].leader = best_cluster;
        _clustering[best_cluster].weight += c_u_weight;
        _clustering[best_cluster].locked = true;
      }
    }

    _interleaved_locked = _clustering[c_u + 1].locked;
  }

  inline void interleaved_visit_neighbor(const NodeID, const NodeID c_v, const EdgeWeight weight) {
    if (!_interleaved_locked) {
      _rating_map[_clustering[c_v].leader] += weight;
    }
  }

  const Graph *_input_graph;
  const Graph *_current_graph;
  SequentialGraphHierarchy _hierarchy;

  const InitialCoarseningContext &_c_ctx;

  std::vector<Cluster> _clustering{};
  FastResetArray<EdgeWeight> _rating_map{};
  std::vector<NodeID> _cluster_sizes{};
  std::vector<NodeID> _leader_node_mapping{};
  FastResetArray<EdgeWeight> _edge_weight_collector{};
  std::vector<NodeID> _cluster_nodes{};

  NodeID _current_num_moves = 0;
  bool _precomputed_clustering = false;
  NodeWeight _interleaved_max_cluster_weight = 0;
  bool _interleaved_locked = false;

  Random &_rand = Random::instance();
  RandomPermutations<NodeID, kChunkSize, kNumberOfNodePermutations> _random_permutations{_rand};
};
} // namespace kaminpar::shm::ip
