/*******************************************************************************
 * @file:   greedy_balancer.h
 * @author: Daniel Seemaier
 * @date:   12.04.2022
 * @brief:  Distributed balancing refinement algorithm.
 ******************************************************************************/
#pragma once

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/dkaminpar.h"
#include "dkaminpar/refinement/refiner.h"

#include "common/datastructures/binary_heap.h"
#include "common/datastructures/marker.h"
#include "common/datastructures/rating_map.h"

namespace kaminpar::dist {
class GreedyBalancerFactory : public GlobalRefinerFactory {
public:
  GreedyBalancerFactory(const Context &ctx);

  GreedyBalancerFactory(const GreedyBalancerFactory &) = delete;
  GreedyBalancerFactory &operator=(const GreedyBalancerFactory &) = delete;

  GreedyBalancerFactory(GreedyBalancerFactory &&) noexcept = default;
  GreedyBalancerFactory &operator=(GreedyBalancerFactory &&) = delete;

  std::unique_ptr<GlobalRefiner>
  create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  const Context &_ctx;
};

class GreedyBalancer : public GlobalRefiner {
  SET_STATISTICS_FROM_GLOBAL();
  SET_DEBUG(false);

  constexpr static std::size_t kPrintStatsEveryNRounds = 100'000;
  constexpr static std::size_t kBucketsPerBlock = 32;

  struct Statistics {
    bool initial_feasible = false;
    bool final_feasible = false;
    BlockID initial_num_imbalanced_blocks = 0;
    BlockID final_num_imbalanced_blocks = 0;
    double initial_imbalance = 0;
    double final_imbalance = 0;
    BlockWeight initial_total_overload = 0;
    BlockWeight final_total_overload = 0;
    GlobalNodeID num_adjacent_moves = 0;
    GlobalNodeID num_nonadjacent_moves = 0;
    GlobalNodeID local_num_conflicts = 0;
    GlobalNodeID local_num_nonconflicts = 0;
    int num_reduction_rounds = 0;

    GlobalEdgeWeight initial_cut = 0;
    GlobalEdgeWeight final_cut = 0;
  };

public:
  GreedyBalancer(
      const Context &ctx, DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
  );

  GreedyBalancer(const GreedyBalancer &) = delete;
  GreedyBalancer &operator=(const GreedyBalancer &) = delete;

  GreedyBalancer(GreedyBalancer &&) noexcept = default;
  GreedyBalancer &operator=(GreedyBalancer &&) = delete;

  void initialize() final;
  bool refine() final;

private:
  void init_buckets();

  inline NodeWeight &get_bucket_value(const BlockID block, const std::size_t bucket) {
    const std::size_t index = block * kBucketsPerBlock + bucket;
    KASSERT(index < _buckets.size());
    return _buckets[index];
  }

  inline const NodeWeight &get_bucket_value(const BlockID block, const std::size_t bucket) const {
    const std::size_t index = block * kBucketsPerBlock + bucket;
    KASSERT(index < _buckets.size());
    return _buckets[index];
  }

  inline int get_bucket(const double rel_gain) const {
    const int bucket = rel_gain < 0 ? std::ceil(std::log2(1 - rel_gain)) : 0;
    KASSERT(bucket >= 0 && bucket < kBucketsPerBlock);
    return bucket;
  }

  inline bool strong_balancing_enabled() const {
    return _ctx.refinement.greedy_balancer.enable_strong_balancing;
  }

  inline bool fast_balancing_enabled() const {
    return _ctx.refinement.greedy_balancer.enable_fast_balancing &&
           _ctx.refinement.greedy_balancer.fast_balancing_threshold > 0;
  }

  struct MoveCandidate {
    GlobalNodeID node;
    BlockID from;
    BlockID to;
    NodeWeight weight;
    double rel_gain;
  };

  std::vector<MoveCandidate> pick_move_candidates();

  template <typename Elements> Elements reduce_buckets_or_move_candidates(Elements &&elements);

  std::vector<MoveCandidate>
  reduce_move_candidates(std::vector<MoveCandidate> &&a, std::vector<MoveCandidate> &&b);

  NoinitVector<NodeWeight>
  reduce_buckets(NoinitVector<NodeWeight> &&a, NoinitVector<NodeWeight> &&b);

  void perform_moves(const std::vector<MoveCandidate> &moves);

  void perform_move(const MoveCandidate &move);

  void
  print_candidates(const std::vector<MoveCandidate> &moves, const std::string &desc = "") const;

  void print_overloads() const;

  void init_pq();

  std::pair<BlockID, double> compute_gain(NodeID u, BlockID u_block) const;

  BlockWeight block_overload(BlockID b) const;

  double compute_relative_gain(EdgeWeight absolute_gain, NodeWeight weight) const;

  bool add_to_pq(BlockID b, NodeID u);

  bool add_to_pq(BlockID b, NodeID u, NodeWeight u_weight, double rel_gain);

  void reset_statistics();

  void print_statistics() const;

  NoinitVector<NodeWeight> compactify_buckets() const;

  const Context &_ctx;

  DistributedPartitionedGraph &_p_graph;
  const PartitionContext &_p_ctx;

  DynamicBinaryMinMaxForest<NodeID, double> _pq;
  mutable tbb::enumerable_thread_specific<RatingMap<EdgeWeight, BlockID>> _rating_map{[&] {
    return RatingMap<EdgeWeight, BlockID>{_ctx.partition.k};
  }};
  std::vector<BlockWeight> _pq_weight;
  Marker<> _marker;

  Statistics _stats;

  NoinitVector<NodeWeight> _buckets;
};
}; // namespace kaminpar::dist
