/*******************************************************************************
 * @file:   apps.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:
 ******************************************************************************/
#pragma once

#include "apps/environment.h"
#include "kaminpar/context.h"
#include "kaminpar/definitions.h"
#include "kaminpar/utility/logger.h"

#if __has_include(<numa.h>)
#include <numa.h>
#endif // __has_include(<numa.h>)

#include <tbb/global_control.h>

#ifdef USE_BACKWARD
#include "backward.hpp"
#endif // USE_BACKWARD

namespace kaminpar {
void print_identifier(int argc, char *argv[]) {
  LLOG << "BUILD ";
  LLOG << "commit=" << Environment::GIT_SHA1 << " ";
  LLOG << "date='" << __DATE__ << "' ";
  LLOG << "time=" << __TIME__ << " ";
  LLOG << "hostname='" << Environment::HOSTNAME << "' ";
  LOG;

  LLOG << "MACROS ";
  LLOG << "KAMINPAR_ENABLE_HEAVY_ASSERTIONS=" << DETECT_EXIST(KAMINPAR_ENABLE_HEAVY_ASSERTIONS) << " ";
  LLOG << "KAMINPAR_ENABLE_ASSERTIONS=" << DETECT_EXIST(KAMINPAR_ENABLE_ASSERTIONS) << " ";
  LLOG << "KAMINPAR_ENABLE_LIGHT_ASSERTIONS=" << DETECT_EXIST(KAMINPAR_ENABLE_LIGHT_ASSERTIONS) << " ";
  LLOG << "KAMINPAR_ENABLE_TIMERS=" << DETECT_EXIST(KAMINPAR_ENABLE_TIMERS) << " ";
  LLOG << "KAMINPAR_ENABLE_STATISTICS=" << DETECT_EXIST(KAMINPAR_ENABLE_STATISTICS) << " ";
  LLOG << "KAMINPAR_64BIT_EDGE_IDS=" << DETECT_EXIST(KAMINPAR_64BIT_EDGE_IDS) << " ";
  LLOG << "KAMINPAR_USE_BACKWARD_CPP=" << DETECT_EXIST(KAMINPAR_USE_BACKWARD_CPP) << " ";
  LOG;

  LOG << "MODIFIED files={" << Environment::GIT_MODIFIED_FILES << "}";

  LLOG << "ARGS ";
  for (int i = 0; i < argc; ++i) { LLOG << "argv[" << i << "]='" << argv[i] << "' "; }
  LOG;

  if (DETECT_EXIST(KAMINPAR_ENABLE_ASSERTIONS) || DETECT_EXIST(KAMINPAR_ENABLE_HEAVY_ASSERTIONS)) {
    LOG << std::string(80, '*');
    LOG << "!!! RUNNING WITH ASSERTIONS !!!";
    LOG << std::string(80, '*');
  }
}

tbb::global_control init_parallelism(const std::size_t num_threads) {
  return tbb::global_control{tbb::global_control::max_allowed_parallelism, num_threads};
}

void init_numa() {
#if __has_include(<numa.h>)
  if (numa_available() >= 0) {
    numa_set_interleave_mask(numa_all_nodes_ptr);
    LOG << "NUMA using round-robin allocations";
    return;
  }
#endif // __has_include(<numa.h>)
  LOG << "NUMA not available";
}

auto init_backward() {
#ifdef USE_BACKWARD
  return backward::SignalHandling{};
#else  // USE_BACKWARD
  return 0;
#endif // USE_BACKWARD
}
} // namespace kaminpar
