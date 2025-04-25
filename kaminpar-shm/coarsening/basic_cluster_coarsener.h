/*******************************************************************************
 * Baseline coarsener that is optimized to contract clusterings.
 *
 * @file:   basic_cluster_coarsener.h
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/coarsening/abstract_cluster_coarsener.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {

class BasicClusterCoarsener : public AbstractClusterCoarsener {
public:
  BasicClusterCoarsener(const Context &ctx, const PartitionContext &p_ctx);

  BasicClusterCoarsener(const BasicClusterCoarsener &) = delete;
  BasicClusterCoarsener &operator=(const BasicClusterCoarsener) = delete;

  BasicClusterCoarsener(BasicClusterCoarsener &&) = delete;
  BasicClusterCoarsener &operator=(BasicClusterCoarsener &&) = delete;

  bool coarsen() final;
};

} // namespace kaminpar::shm
