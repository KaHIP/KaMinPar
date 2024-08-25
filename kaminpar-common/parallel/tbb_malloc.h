/*******************************************************************************
 * @file:   tbb_malloc.h
 * @author: Daniel Seemaier
 * @date:   30.03.2022
 * @brief:  Memory allocation functions that use the TBB scalable allocator.
 ******************************************************************************/
#pragma once

#include <memory>

#include <tbb/scalable_allocator.h>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/heap_profiler.h"

#if defined(__linux__) && defined(KAMINPAR_ENABLE_THP)
#include "sys/mman.h"
#endif

namespace kaminpar::parallel {
template <typename T> struct tbb_deleter {
  void operator()(T *p) {
    scalable_free(p);

    if constexpr (kHeapProfiling && !kPageProfiling) {
      heap_profiler::HeapProfiler::global().record_free(p);
    }
  }
};

template <typename T> using tbb_unique_ptr = std::unique_ptr<T, tbb_deleter<T>>;
// template <typename T> using tbb_unique_ptr = std::unique_ptr<T>;

template <typename T> tbb_unique_ptr<T> make_unique(const std::size_t size, const bool thp) {
  auto nbytes = sizeof(T) * size;
  T *ptr = nullptr;

#if defined(__linux__) && defined(KAMINPAR_ENABLE_THP)
  if (thp) {
    scalable_posix_memalign(reinterpret_cast<void **>(&ptr), 1 << 21, nbytes);
    madvise(ptr, nbytes, MADV_HUGEPAGE);
  } else {
#endif
    ptr = static_cast<T *>(scalable_malloc(nbytes));
#if defined(__linux__) && defined(KAMINPAR_ENABLE_THP)
  }
#endif

  KASSERT(
      ptr != nullptr, "out of memory: could not allocate " << nbytes << " bytes", assert::light
  );

  if constexpr (kHeapProfiling && !kPageProfiling) {
    heap_profiler::HeapProfiler::global().record_alloc(ptr, sizeof(T) * size);
  }

  return tbb_unique_ptr<T>(ptr, tbb_deleter<T>{});
}
} // namespace kaminpar::parallel
