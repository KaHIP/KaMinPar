/*******************************************************************************
 * Interface for graph partitioning schemes.
 *
 * @file:   partitioner.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"

namespace kaminpar::dist {

class Partitioner {
public:
  virtual ~Partitioner() = default;
  virtual DistributedPartitionedGraph partition() = 0;

  void enable_graph_stats_output() {
    _print_graph_stats = true;
  }

protected:
  bool _print_graph_stats = false;
};

} // namespace kaminpar::dist
