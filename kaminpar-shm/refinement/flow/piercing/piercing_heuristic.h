#pragma once

#include <algorithm>
#include <cstdint>
#include <span>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/util/breadth_first_search.h"
#include "kaminpar-shm/refinement/flow/util/node_status.h"

#include "kaminpar-common/datastructures/binary_heap.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

class PiercingHeuristic {
  static constexpr std::uint8_t kUnknown = 0;
  static constexpr std::uint8_t kInitialSourceSide = 1;
  static constexpr std::uint8_t kInitialSinkSide = 2;

  class PiercingNodeCandidatesBuckets {
  public:
    void initialize(const NodeID max_distance) {
      _candidates_buckets.resize(max_distance + 1);
    }

    void reset() {
      for (ScalableVector<NodeID> &candidates : _candidates_buckets) {
        candidates.clear();
      }
    }

    void add_candidate(const NodeID u, const NodeID distance) {
      KASSERT(distance < _candidates_buckets.size());
      _candidates_buckets[distance].push_back(u);
    }

    void sort(const NodeID bucket) {
      std::sort(_candidates_buckets[bucket].begin(), _candidates_buckets[bucket].end());
    }

    [[nodiscard]] NodeID min_occupied_bucket() const {
      return 0;
    }

    [[nodiscard]] NodeID max_occupied_bucket() const {
      return _candidates_buckets.size() - 1;
    }

    [[nodiscard]] std::span<const NodeID> candidates(const NodeID bucket) const {
      KASSERT(bucket < _candidates_buckets.size());

      return _candidates_buckets[bucket];
    }

  private:
    ScalableVector<ScalableVector<NodeID>> _candidates_buckets;
  };

  struct BulkPiercingContext {
    std::size_t num_rounds;
    std::size_t total_bulk_piercing_nodes;

    NodeWeight initial_side_weight;
    NodeWeight weight_added_so_far;
    NodeWeight current_weight_goal;
    NodeWeight current_weight_goal_remaining;

    void initialize(
        const NodeWeight side_weight,
        const NodeWeight total_weight,
        const NodeWeight max_side_weight,
        const NodeWeight max_total_weight
    ) {
      num_rounds = 0;
      total_bulk_piercing_nodes = 0;

      initial_side_weight = side_weight;
      weight_added_so_far = 0;

      const double ratio = max_side_weight / static_cast<double>(max_total_weight);
      current_weight_goal = std::max(0.0, ratio * total_weight - side_weight);
      current_weight_goal_remaining = 0;
    }
  };

public:
  PiercingHeuristic(const PiercingHeuristicContext &_ctx);

  void initialize(
      const CSRGraph &graph,
      NodeWeight source_side_weight,
      NodeWeight sink_side_weight,
      NodeWeight total_weight,
      NodeWeight max_source_side_weight,
      NodeWeight max_sink_side_weight
  );

  void set_initial_node_side(NodeID node, bool source_side);

  void compute_distances();

  void reset(bool source_side);

  void add_piercing_node_candidate(NodeID node, bool reachable);

  std::span<const NodeID> find_piercing_nodes(
      const NodeStatus &cut_status,
      const NodeStatus &terminal_status,
      NodeWeight side_weight,
      NodeWeight max_weight
  );

private:
  std::size_t compute_max_num_piercing_nodes(NodeWeight side_weight);

  BulkPiercingContext &bulk_piercing_context();

private:
  const PiercingHeuristicContext &_ph_ctx;

  const CSRGraph *_graph;
  ScalableVector<NodeID> _piercing_nodes;

  StaticArray<std::uint8_t> _initial_side_status;
  StaticArray<NodeID> _distance;

  BFSRunner _source_side_bfs_runner;
  BFSRunner _sink_side_bfs_runner;

  bool _source_side;
  PiercingNodeCandidatesBuckets _reachable_candidates_buckets;
  PiercingNodeCandidatesBuckets _unreachable_candidates_buckets;

  BulkPiercingContext _source_side_bulk_piercing_ctx;
  BulkPiercingContext _sink_side_bulk_piercing_ctx;
};

} // namespace kaminpar::shm
