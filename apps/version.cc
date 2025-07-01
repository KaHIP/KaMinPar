/*******************************************************************************
 * Prints a detailed version message including the build configuration.
 *
 * @file:   version.cc
 * @author: Daniel Seemaier
 * @date:   18.09.2024
 ******************************************************************************/
#include "apps/version.h"

#include <iostream>

#include <kaminpar-common/assert.h>
#include <kaminpar-common/environment.h>
#include <kaminpar-shm/kaminpar.h>
#include <tbb/version.h>

namespace kaminpar {

void print_version() {
  std::cout << "KaMinPar v" << KAMINPAR_VERSION_MAJOR << "." << KAMINPAR_VERSION_MINOR << "."
            << KAMINPAR_VERSION_PATCH << "\n";
  std::cout << "Git version: " << Environment::GIT_SHA1
            << ", built on host: " << Environment::HOSTNAME << "\n";
  std::cout << "Build configuration:\n";

#ifdef KAMINPAR_EXPERIMENTAL
  std::cout << "  Experimental features: enabled\n";
#else
  std::cout << "  Experimental features: disabled\n";
#endif
#ifdef KAMINPAR_ENABLE_TIMERS
  std::cout << "  Timers: enabled\n";
#else
  std::cout << "  Timers: disabled\n";
#endif
#ifdef KAMINPAR_ENABLE_THP
  std::cout << "  Transparent huge pages: enabled\n";
#else
  std::cout << "  Transparent huge pages: disabled\n";
#endif
#ifdef KAMINPAR_ENABLE_STATISTICS
  std::cout << "  Statistics: enabled\n";
#else
  std::cout << "  Statistics: disabled\n";
#endif
#ifdef KAMINPAR_ENABLE_HEAP_PROFILING
  std::cout << "  Heap profiling: enabled\n";
#else
  std::cout << "  Heap profiling: disabled\n";
#endif
#ifdef KAMINPAR_ENABLE_PAGE_PROFILING
  std::cout << "  Page profiling: enabled\n";
#else
  std::cout << "  Page profiling: disabled\n";
#endif
  std::cout << "  Graph Compression:\n";
#ifdef KAMINPAR_COMPRESSION_EDGE_WEIGHT
  std::cout << "    Edge weight compression: enabled\n";
#else
  std::cout << "    Edge weight compression: disabled\n";
#endif
#ifdef KAMINPAR_COMPRESSION_HIGH_DEGREE_ENCODING
  std::cout << "    High degree encoding: enabled\n";
#else
  std::cout << "    High degree encoding: disabled\n";
#endif
#ifdef KAMINPAR_COMPRESSION_INTERVAL_ENCODING
  std::cout << "    Interval encoding: enabled\n";
#else
  std::cout << "    Interval encoding: disabled\n";
#endif
#ifdef KAMINPAR_COMPRESSION_RUN_LENGTH_ENCODING
  std::cout << "    Run-length encoding: enabled\n";
#else
  std::cout << "    Run-length encoding: disabled\n";
#endif
#ifdef KAMINPAR_COMPRESSION_STREAMVBYTE_ENCODING
  std::cout << "    StreamVByte encoding: enabled\n";
#else
  std::cout << "    StreamVByte encoding: disabled\n";
#endif
#ifdef KAMINPAR_COMPRESSION_FAST_DECODING
  std::cout << "    Fast decoding: enabled\n";
#else
  std::cout << "    Fast decoding: disabled\n";
#endif
  std::cout << "  Data types:\n";
#ifdef KAMINPAR_64BIT_NODE_IDS
  std::cout << "    64-bit node IDs: enabled\n";
#else
  std::cout << "    64-bit node IDs: disabled\n";
#endif
#ifdef KAMINPAR_64BIT_EDGE_IDS
  std::cout << "    64-bit edge IDs: enabled\n";
#else
  std::cout << "    64-bit edge IDs: disabled\n";
#endif
#ifdef KAMINPAR_64BIT_WEIGHTS
  std::cout << "    64-bit weights: enabled\n";
#else
  std::cout << "    64-bit weights: disabled\n";
#endif
#ifdef KAMINPAR_64BIT_LOCAL_WEIGHTS
  std::cout << "    64-bit local weights: enabled\n";
#else
  std::cout << "    64-bit local weights: disabled\n";
#endif
  std::cout << "  Dependencies:\n";
#ifdef KAMINPAR_SPARSEHASH_FOUND
  std::cout << "    Sparsehash: found and enabled\n";
#else
  std::cout << "    Sparsehash: not found or disabled\n";
#endif
#ifdef KAMINPAR_USES_GROWT
  std::cout << "    Growt: found and enabled\n";
#else
  std::cout << "    Growt: not found or disabled\n";
#endif
#ifdef KAMINPAR_HAVE_BACKWARD
  std::cout << "    Backward: found and enabled\n";
#else
  std::cout << "    Backward: not found or disabled\n";
#endif
#ifdef KAMINPAR_MTKAHYPAR_FOUND
  std::cout << "    Mt-KaHyPar: found and enabled\n";
#else
  std::cout << "    Mt-KaHyPar: not found or disabled\n";
#endif
  std::cout << "    TBB: " << TBB_VERSION_STRING << "\n";
  std::cout << "  Assertion levels: always";
#if KASSERT_ENABLED(ASSERTION_LEVEL_LIGHT)
  std::cout << "+light";
#endif
#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
  std::cout << "+normal";
#endif
#if KASSERT_ENABLED(ASSERTION_LEVEL_HEAVY)
  std::cout << "+heavy";
#endif
  std::cout << "\n";
  std::cout << std::flush;
}

} // namespace kaminpar
