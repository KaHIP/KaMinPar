/*******************************************************************************
 * Greedy balancing algorithm that moves sets of nodes at a time.
 *
 * @file:   move_set_balancer.h
 * @author: Daniel Seemaier
 * @date:   19.07.2023
 ******************************************************************************/
#pragma once

#include <memory>

#include "dkaminpar/refinement/balancer/move_sets.h"
#include "dkaminpar/refinement/balancer/weight_buckets.h"
#include "dkaminpar/refinement/refiner.h"

#include "common/datastructures/binary_heap.h"
#include "common/datastructures/marker.h"
#include "common/logger.h"
#include "common/random.h"
#include "common/timer.h"

namespace kaminpar::dist {
struct MoveSetBalancerMemoryContext;

class MoveSetBalancerFactory : public GlobalRefinerFactory {
public:
  MoveSetBalancerFactory(const Context &ctx);

  ~MoveSetBalancerFactory();

  std::unique_ptr<GlobalRefiner>
  create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

  void take_m_ctx(MoveSetBalancerMemoryContext m_ctx);

private:
  const Context &_ctx;

  std::unique_ptr<MoveSetBalancerMemoryContext> _m_ctx;
};

class MoveSetBalancer : public GlobalRefiner {
public:
  MoveSetBalancer(
      MoveSetBalancerFactory &factory,
      const Context &ctx,
      DistributedPartitionedGraph &p_graph,
      const PartitionContext &p_ctx,
      MoveSetBalancerMemoryContext m_ctx
  );

  MoveSetBalancer(const MoveSetBalancer &) = delete;
  MoveSetBalancer &operator=(const MoveSetBalancer &) = delete;

  MoveSetBalancer(MoveSetBalancer &&) = delete;
  MoveSetBalancer &operator=(MoveSetBalancer &&) = delete;

  ~MoveSetBalancer();

  operator MoveSetBalancerMemoryContext() &&;

  void initialize() final;
  bool refine() final;

private:
  void try_pq_insertion(NodeID set);
  void try_pq_update(NodeID set);

  void perform_parallel_round();

  struct MoveCandidate {
    PEID owner;
    NodeID set;
    NodeWeight weight;
    double gain;
    BlockID from;
    BlockID to;
  };

  void perform_sequential_round();
  std::vector<MoveCandidate> pick_sequential_candidates();
  std::vector<MoveCandidate> reduce_sequential_candidates(std::vector<MoveCandidate> candidates);
  void perform_moves(const std::vector<MoveCandidate> &candidates, bool update_block_weights);

  BlockWeight overload(BlockID block) const;
  BlockWeight underload(BlockID block) const;
  bool is_overloaded(BlockID block) const;
  BlockID count_overloaded_blocks() const;

  bool assign_feasible_target_block(
      MoveCandidate &candidate, const std::vector<BlockWeight> &deltas
  ) const;

  Random &_rand = Random::instance();

  MoveSetBalancerFactory &_factory;
  const Context &_ctx;
  DistributedPartitionedGraph &_p_graph;
  const PartitionContext &_p_ctx;

  DynamicBinaryMinMaxForest<NodeID, double> _pqs;
  NoinitVector<BlockWeight> _pq_weights;
  Marker<> _moved_marker;

  Buckets _weight_buckets;
  MoveSets _move_sets;
};
} // namespace kaminpar::dist
