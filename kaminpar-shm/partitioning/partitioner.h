/*******************************************************************************
 * Interface for partitioning schemes.
 *
 * @file:   partitioner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/partitioned_graph.h"

namespace kaminpar::shm {

class Partitioner {
public:
  virtual ~Partitioner() = default;
  virtual PartitionedGraph partition() = 0;

  void enable_metrics_output() {
    _print_metrics = true;
  }

protected:
  bool _print_metrics = false;
};

} // namespace kaminpar::shm
