/*******************************************************************************
 * Thread-local pool of initial bipartitioner workers.
 *
 * @file:   initial_worker_pool.h
 * @author: Daniel Seemaier
 * @date:   05.06.2024
 ******************************************************************************/
#pragma once

#include <vector>

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-shm/initial_partitioning/initial_multilevel_bipartitioner.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {
class InitialBipartitionerWorkerPool {
public:
  explicit InitialBipartitionerWorkerPool(const Context &ctx) : _ctx(ctx) {}

  InitialMultilevelBipartitioner get() {
    auto &pool = _pool_ets.local();

    if (!pool.empty()) {
      auto initial_partitioner = std::move(pool.back());
      pool.pop_back();
      return initial_partitioner;
    }

    return InitialMultilevelBipartitioner(_ctx);
  }

  void put(InitialMultilevelBipartitioner initial_partitioner) {
    auto &pool = _pool_ets.local();
    pool.push_back(std::move(initial_partitioner));
  }

private:
  const Context &_ctx;
  tbb::enumerable_thread_specific<std::vector<InitialMultilevelBipartitioner>> _pool_ets;
};
} // namespace kaminpar::shm
