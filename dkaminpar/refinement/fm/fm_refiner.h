/*******************************************************************************
 * Distributed FM refiner.
 *
 * @file:   fm_refiner.h
 * @author: Daniel Seemaier
 * @date:   11.09.2022
 ******************************************************************************/
#pragma once

#include <tbb/concurrent_vector.h>

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/datastructures/growt.h"
#include "dkaminpar/refinement/refiner.h"

#include "kaminpar/refinement/fm_refiner.h"

#include "common/logger.h"
#include "common/parallel/atomic.h"

namespace kaminpar::dist {
namespace fm {
class NodeMapper {
public:
  NodeMapper(NoinitVector<GlobalNodeID> batch_to_graph);

  NodeMapper(const NodeMapper &) = delete;
  NodeMapper &operator=(const NodeMapper &) = delete;

  NodeMapper(NodeMapper &&) noexcept = default;
  NodeMapper &operator=(NodeMapper &&) = delete;

  GlobalNodeID to_graph(NodeID bnode) const;
  NodeID to_batch(GlobalNodeID gnode) const;

private:
  void construct();

  NoinitVector<GlobalNodeID> _batch_to_graph;
  growt::StaticGhostNodeMapping _graph_to_batch;
};

class PartitionRollbacker {
public:
  virtual ~PartitionRollbacker() = default;

  virtual void update() = 0;
  virtual void rollback() = 0;
};

class EnabledPartitionRollbacker : public PartitionRollbacker {
public:
  EnabledPartitionRollbacker(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx);

  void update() final;
  void rollback() final;

private:
  void copy_partition();

  DistributedPartitionedGraph &_p_graph;
  const PartitionContext &_p_ctx;

  bool _last_is_best = true;
  EdgeWeight _best_cut;
  double _best_l1 = 0.0;
  NoinitVector<BlockID> _best_partition;
  NoinitVector<NodeWeight> _best_block_weights;
};

class DisabledPartitionRollbacker : public PartitionRollbacker {
public:
  void update() final {}
  void rollback() final {}
};
} // namespace fm

class FMRefinerFactory : public GlobalRefinerFactory {
public:
  FMRefinerFactory(const Context &ctx);

  FMRefinerFactory(const FMRefinerFactory &) = delete;
  FMRefinerFactory &operator=(const FMRefinerFactory &) = delete;

  FMRefinerFactory(FMRefinerFactory &&) noexcept = default;
  FMRefinerFactory &operator=(FMRefinerFactory &&) = delete;

  std::unique_ptr<GlobalRefiner>
  create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  const Context &_ctx;
};

class FMRefiner : public GlobalRefiner {
public:
  FMRefiner(
      const Context &ctx, DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
  );

  FMRefiner(const FMRefiner &) = delete;
  FMRefiner &operator=(const FMRefiner &) = delete;
  FMRefiner(FMRefiner &&) noexcept = default;
  FMRefiner &operator=(FMRefiner &&) = delete;

  void initialize() final;
  bool refine() final;

private:
  shm::PartitionContext setup_shm_p_ctx(const shm::Graph &b_graph) const;
  shm::fm::SharedData setup_fm_data(
      const shm::PartitionedGraph &bp_graph,
      const std::vector<NodeID> &seeds,
      const fm::NodeMapper &mapper
  ) const;
  shm::KwayFMRefinementContext setup_fm_ctx() const;

  const Context &_ctx;
  const FMRefinementContext &_fm_ctx;

  DistributedPartitionedGraph &_p_graph;
  const PartitionContext &_p_ctx;

  std::unique_ptr<GlobalRefinerFactory> _balancer_factory;
};
} // namespace kaminpar::dist
