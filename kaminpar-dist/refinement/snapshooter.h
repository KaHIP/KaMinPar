/*******************************************************************************
 * Utility functions to take copies of partitions and re-apply them if desired.
 *
 * @file:   snapshooter.h
 * @author: Daniel Seemaier
 * @date:   21.09.2023
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/dkaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::dist {
class PartitionSnapshooter {
public:
  virtual ~PartitionSnapshooter() = default;

  virtual void init(const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) = 0;

  virtual void
  update(const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) = 0;

  virtual void update(
      const DistributedPartitionedGraph &p_graph,
      const PartitionContext &p_ctx,
      EdgeWeight cut,
      double l1
  ) = 0;

  virtual void rollback(DistributedPartitionedGraph &p_graph) = 0;
};

class BestPartitionSnapshooter : public PartitionSnapshooter {
public:
  BestPartitionSnapshooter(NodeID max_total_n, BlockID max_k);

  void init(const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

  void update(const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

  void update(
      const DistributedPartitionedGraph &p_graph,
      const PartitionContext &p_ctx,
      EdgeWeight cut,
      double l1
  ) final;

  void rollback(DistributedPartitionedGraph &p_graph) final;

private:
  void copy_partition(const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx);

  double _best_l1;
  EdgeWeight _best_cut;
  StaticArray<BlockID> _best_partition;
  StaticArray<BlockWeight> _best_block_weights;
  bool _last_is_best;
};

class DummyPartitionSnapshooter : public PartitionSnapshooter {
public:
  void init(const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

  void update(const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

  void update(const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx, EdgeWeight cut, double l1) final;

  void rollback(DistributedPartitionedGraph &p_graph) final;
};
} // namespace kaminpar::dist
