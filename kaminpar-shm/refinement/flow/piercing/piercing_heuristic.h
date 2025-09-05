#pragma once

#include <algorithm>
#include <cstddef>
#include <span>

#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/flow_network/border_region.h"
#include "kaminpar-shm/refinement/flow/flow_network/flow_network.h"
#include "kaminpar-shm/refinement/flow/util/breadth_first_search.h"
#include "kaminpar-shm/refinement/flow/util/node_status.h"

#include "kaminpar-common/datastructures/marker.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm {

class PiercingHeuristic {
  static constexpr bool kReachableTag = true;
  static constexpr bool kUnreachableTag = false;

  class PiercingNodeCandidatesBuckets {
  public:
    [[nodiscard]] NodeID min_occupied_bucket() const {
      return 0;
    }

    [[nodiscard]] NodeID max_occupied_bucket() const {
      return _candidates_buckets.size() - 1;
    }

    [[nodiscard]] ScalableVector<NodeID> &candidates(const NodeID bucket) {
      KASSERT(bucket < _candidates_buckets.size());

      return _candidates_buckets[bucket];
    }

    void initialize(const NodeID max_distance) {
      _candidates_buckets.resize(max_distance + 1);
      reset();
    }

    void add_candidate(const NodeID u, const NodeID distance) {
      KASSERT(distance < _candidates_buckets.size());
      _candidates_buckets[distance].push_back(u);
    }

    void reset() {
      for (ScalableVector<NodeID> &candidates : _candidates_buckets) {
        candidates.clear();
      }
    }

    void free() {
      _candidates_buckets.clear();
      _candidates_buckets.shrink_to_fit();
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
  PiercingHeuristic(const PiercingHeuristicContext &_ctx, BlockID num_blocks);

  void initialize(
      const BorderRegion &border_region,
      const FlowNetwork &flow_network,
      NodeWeight max_source_side_weight,
      NodeWeight max_sink_side_weight
  );

  void add_piercing_node_candidate(bool source_side, NodeID node, bool reachable);

  std::span<const NodeID> find_piercing_nodes(
      bool source_side,
      const NodeStatus &cut_status,
      const Marker<> &reachable_oracle,
      NodeWeight side_weight,
      NodeWeight max_weight
  );

  void free();

private:
  void compute_distances();

  std::size_t compute_max_num_piercing_nodes(bool source_side, NodeWeight side_weight);

private:
  const PiercingHeuristicContext &_ph_ctx;
  const BlockID _num_blocks;

  const BorderRegion *_border_region;
  const FlowNetwork *_flow_network;

  ScalableVector<NodeID> _piercing_nodes;

  BFSRunner _bfs_runner;
  StaticArray<NodeWeight> _distance;

  PiercingNodeCandidatesBuckets _source_reachable_candidates_buckets;
  PiercingNodeCandidatesBuckets _source_unreachable_candidates_buckets;

  PiercingNodeCandidatesBuckets _sink_reachable_candidates_buckets;
  PiercingNodeCandidatesBuckets _sink_unreachable_candidates_buckets;

  BulkPiercingContext _source_side_bulk_piercing_ctx;
  BulkPiercingContext _sink_side_bulk_piercing_ctx;

  Random _random;
};

} // namespace kaminpar::shm
