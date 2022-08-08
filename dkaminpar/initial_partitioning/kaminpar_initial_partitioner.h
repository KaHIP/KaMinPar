/*******************************************************************************
 * @file:   kaminpar_initial_partitioner.h
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 * @brief:  Initial partitioner invoking KaMinPar.
 ******************************************************************************/
#pragma once

#include "dkaminpar/context.h"
#include "dkaminpar/initial_partitioning/i_initial_partitioner.h"

namespace kaminpar::dist {
class KaMinParInitialPartitioner : public IInitialPartitioner {
public:
    KaMinParInitialPartitioner(const Context& ctx) : _ctx{ctx} {}

    KaMinParInitialPartitioner(const KaMinParInitialPartitioner&)            = delete;
    KaMinParInitialPartitioner& operator=(const KaMinParInitialPartitioner&) = delete;
    KaMinParInitialPartitioner(KaMinParInitialPartitioner&&) noexcept        = default;
    KaMinParInitialPartitioner& operator=(KaMinParInitialPartitioner&&)      = delete;

    shm::PartitionedGraph initial_partition(const shm::Graph& graph) override;

private:
    const Context& _ctx;
};
} // namespace dkaminpar
