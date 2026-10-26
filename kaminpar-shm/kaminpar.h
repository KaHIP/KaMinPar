/*******************************************************************************
 * Public library interface of KaMinPar.
 *
 * @file:   kaminpar.h
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 ******************************************************************************/
#pragma once

// Part of this header is defined in the public library header:
#include "include/kaminpar-shm/kaminpar.h" // IWYU pragma: export

#include "kaminpar-common/runtime_context.h"

namespace kaminpar::shm {

// Copies fork problem state while retaining the same configuration and services.
struct ExecutionContext {
  const Context &config;
  RuntimeContext &runtime;
  PartitionContext partition;
  ParallelContext parallel;

  ExecutionContext(const Context &config, RuntimeContext &runtime)
      : config(config),
        runtime(runtime),
        parallel(config.parallel) {}
};

} // namespace kaminpar::shm
