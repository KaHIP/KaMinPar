/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2020 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
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
  LLOG << "KAMINPAR_EXPERIMENTS_MODE=" << DETECT_EXIST(KAMINPAR_EXPERIMENTS_MODE) << " ";
  LLOG << "KAMINPAR_ENABLE_DEBUG_FEATURES=" << DETECT_EXIST(KAMINPAR_ENABLE_DEBUG_FEATURES) << " ";
  LLOG << "KAMINPAR_ENABLE_TIMERS=" << DETECT_EXIST(KAMINPAR_ENABLE_TIMERS) << " ";
  LLOG << "KAMINPAR_LINK_TCMALLOC=" << DETECT_EXIST(KAMINPAR_LINK_TCMALLOC) << " ";
  LLOG << "KAMINPAR_64BIT_EDGE_IDS=" << DETECT_EXIST(KAMINPAR_64BIT_EDGE_IDS) << " ";
  LLOG << "KAMINPAR_ENABLE_ASSERTIONS=" << DETECT_EXIST(KAMINPAR_ENABLE_ASSERTIONS) << " ";
  LOG;

  LOG << "MODIFIED files={" << Environment::GIT_MODIFIED_FILES << "}";

  LLOG << "ARGS ";
  for (int i = 0; i < argc; ++i) { LLOG << "argv[" << i << "]='" << argv[i] << "' "; }
  LOG;

  if (DETECT_EXIST(KAMINPAR_ENABLE_ASSERTIONS)) {
    LOG << std::string(80, '*');
    LOG << "!!! RUNNING IN DEBUG MODE !!!";
    LOG << std::string(80, '*');
  }
}

void force_clean_build() {
  ALWAYS_ASSERT(Environment::GIT_MODIFIED_FILES != "<none>" && Environment::GIT_MODIFIED_FILES != "<unavailable>")
      << "Please commit your changes before running experiments.";
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
