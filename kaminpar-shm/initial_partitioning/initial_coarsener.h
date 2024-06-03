/*******************************************************************************
 * Sequential label propagation coarsening used during initial bipartitionign.
 *
 * @file:   initial_coarsener.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <utility>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/initial_partitioning/sequential_graph_hierarchy.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/fast_reset_array.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm::ip {
struct InitialCoarsenerTimings {
  std::uint64_t contract_ms = 0;
  std::uint64_t alloc_ms = 0;
  std::uint64_t interleaved1_ms = 0;
  std::uint64_t interleaved2_ms = 0;
  std::uint64_t lp_ms = 0;
  std::uint64_t total_ms = 0;

  InitialCoarsenerTimings &operator+=(const InitialCoarsenerTimings &other) {
    contract_ms += other.contract_ms;
    alloc_ms += other.alloc_ms;
    interleaved1_ms += other.interleaved1_ms;
    interleaved2_ms += other.interleaved2_ms;
    lp_ms += other.lp_ms;
    total_ms += other.total_ms;
    return *this;
  }
};

class InitialCoarsener {
  static constexpr auto kChunkSize = 256;
  static constexpr auto kNumberOfNodePermutations = 16;

  using ContractionResult = std::pair<CSRGraph, ScalableVector<NodeID>>;

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

  InitialCoarsener(const InitialCoarseningContext &c_ctx);

  InitialCoarsener(const InitialCoarsener &) = delete;
  InitialCoarsener &operator=(const InitialCoarsener &) = delete;

  InitialCoarsener(InitialCoarsener &&) noexcept = delete;
  InitialCoarsener &operator=(InitialCoarsener &&) = delete;

  [[nodiscard]] inline std::size_t level() const {
    return _hierarchy.level();
  }

  [[nodiscard]] inline bool empty() const {
    return _hierarchy.empty();
  }

  [[nodiscard]] inline const CSRGraph *current() const {
    return &_hierarchy.current();
  }

  void init(const CSRGraph &graph);

  const CSRGraph *coarsen(NodeWeight max_cluster_weight);
  PartitionedCSRGraph uncoarsen(PartitionedCSRGraph &&c_p_graph);

  void reset_current_clustering() {
    if (_current_graph->is_node_weighted()) {
      reset_current_clustering(_current_graph->n(), _current_graph->raw_node_weights());
    } else {
      // This is robust if _current_graph is empty
      // (in this case, we cannot use node_weight(0))
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

  InitialCoarsenerTimings timings() {
    auto timings = _timings;
    _timings = InitialCoarsenerTimings{};
    return timings;
  }

private:
  ContractionResult contract_current_clustering();

  void perform_label_propagation(NodeWeight max_cluster_weight);

  // Interleaved label propagation: compute for the next coarse graph while
  // constructing it
  inline void interleaved_handle_node(const NodeID c_u, const NodeWeight c_u_weight) {
    if (!_interleaved_locked) {
      const NodeID best_cluster{
          pick_cluster_from_rating_map(c_u, c_u_weight, _interleaved_max_cluster_weight)
      };
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

  const CSRGraph *_input_graph;
  const CSRGraph *_current_graph;
  SequentialGraphHierarchy _hierarchy;

  const InitialCoarseningContext &_c_ctx;

  ScalableVector<Cluster> _clustering;
  FastResetArray<EdgeWeight> _rating_map;
  ScalableVector<NodeID> _cluster_sizes;
  ScalableVector<NodeID> _leader_node_mapping;
  FastResetArray<EdgeWeight> _edge_weight_collector;
  ScalableVector<NodeID> _cluster_nodes;

  NodeID _current_num_moves = 0;
  bool _precomputed_clustering = false;
  NodeWeight _interleaved_max_cluster_weight = 0;
  bool _interleaved_locked = false;

  Random &_rand = Random::instance();
  RandomPermutations<NodeID, kChunkSize, kNumberOfNodePermutations> _random_permutations{_rand};

  InitialCoarsenerTimings _timings{};
};
} // namespace kaminpar::shm::ip
