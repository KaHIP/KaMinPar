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

  virtual void update() = 0;
  virtual void rollback() = 0;
};

class BestPartitionSnapshooter : public PartitionSnapshooter {
public:
  BestPartitionSnapshooter(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx);

  void update() final;
  void rollback() final;

private:
  void copy_partition();

  DistributedPartitionedGraph &_p_graph;
  const PartitionContext &_p_ctx;

  double _best_l1;
  EdgeWeight _best_cut;
  StaticArray<BlockID> _best_partition;
  StaticArray<BlockWeight> _best_block_weights;

  bool _last_is_best;
};

class DummyPartitionSnapshooter : public PartitionSnapshooter {
public:
  void update() final;
  void rollback() final;
};
} // namespace kaminpar::dist
