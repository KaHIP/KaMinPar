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
#include "dkaminpar/refinement/balancer/weight_buckets.h"
#include "dkaminpar/refinement/gain_calculator.h"
#include "dkaminpar/refinement/refiner.h"

#include "common/datastructures/binary_heap.h"
#include "common/datastructures/marker.h"

namespace kaminpar::dist {
class NodeBalancerFactory : public GlobalRefinerFactory {
public:
  NodeBalancerFactory(const Context &ctx);

  NodeBalancerFactory(const NodeBalancerFactory &) = delete;
  NodeBalancerFactory &operator=(const NodeBalancerFactory &) = delete;

  NodeBalancerFactory(NodeBalancerFactory &&) noexcept = default;
  NodeBalancerFactory &operator=(NodeBalancerFactory &&) = delete;

  std::unique_ptr<GlobalRefiner>
  create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  const Context &_ctx;
};

class NodeBalancer : public GlobalRefiner {
  SET_STATISTICS_FROM_GLOBAL();
  SET_DEBUG(false);

  struct Candidate {
    GlobalNodeID id;
    BlockID from;
    BlockID to;
    NodeWeight weight;
    double gain;
  };

public:
  NodeBalancer(
      const Context &ctx, DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
  );

  NodeBalancer(const NodeBalancer &) = delete;
  NodeBalancer &operator=(const NodeBalancer &) = delete;

  NodeBalancer(NodeBalancer &&) noexcept = default;
  NodeBalancer &operator=(NodeBalancer &&) = delete;

  void initialize() final;
  bool refine() final;

private:
  bool is_sequential_balancing_enabled() const;
  bool is_parallel_balancing_enabled() const;

  bool perform_sequential_round();
  std::vector<Candidate> pick_sequential_candidates();

  void perform_moves(const std::vector<Candidate> &moves, bool update_block_weights);
  void perform_move(const Candidate &move, bool update_block_weights);

  BlockWeight block_overload(BlockID b) const;
  BlockWeight block_underload(BlockID b) const;

  bool try_pq_insertion(BlockID b, NodeID u);
  bool try_pq_insertion(BlockID b, NodeID u, NodeWeight u_weight, double rel_gain);

  bool perform_parallel_round();

  bool
  assign_feasible_target_block(Candidate &candidate, const std::vector<BlockWeight> &deltas) const;

  DistributedPartitionedGraph &_p_graph;

  const Context &_ctx;
  const NodeBalancerContext &_nb_ctx;
  const PartitionContext &_p_ctx;

  DynamicBinaryMinMaxForest<NodeID, double> _pq;
  std::vector<BlockWeight> _pq_weight;
  Marker<> _marker;

  Buckets _buckets;
  GainCalculator _gain_calculator;

  bool _stalled = false;

  std::vector<std::size_t> _cached_cutoff_buckets;
};
}; // namespace kaminpar::dist
