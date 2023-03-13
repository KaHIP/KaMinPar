/*******************************************************************************
 * @file:   random_initial_partitioner.h
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 * @brief:  Initial partitioner that assigns nodes to blocks randomly.
 ******************************************************************************/
#pragma once

#include "dkaminpar/context.h"
#include "dkaminpar/initial_partitioning/initial_partitioner.h"

namespace kaminpar::dist {
class RandomInitialPartitioner : public InitialPartitioner {
public:
  RandomInitialPartitioner() {}

  RandomInitialPartitioner(const RandomInitialPartitioner &) = delete;
  RandomInitialPartitioner &
  operator=(const RandomInitialPartitioner &) = delete;
  RandomInitialPartitioner(RandomInitialPartitioner &&) noexcept = default;
  RandomInitialPartitioner &operator=(RandomInitialPartitioner &&) = delete;

  shm::PartitionedGraph
  initial_partition(const shm::Graph &graph,
                    const PartitionContext &p_ctx) override;
};
} // namespace kaminpar::dist
