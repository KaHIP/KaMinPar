/*******************************************************************************
 * @file:   tbb_malloc.h
 * @author: Daniel Seemaier
 * @date:   30.03.2022
 * @brief:  Memory allocation functions that use the TBB scalable allocator.
 ******************************************************************************/
#pragma once

#include <memory>

#include <tbb/scalable_allocator.h>

#include "kaminpar-common/heap_profiler.h"

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

template <typename T> tbb_unique_ptr<T> make_unique(const std::size_t size) {
  T *ptr = static_cast<T *>(scalable_malloc(sizeof(T) * size));

  if constexpr (kHeapProfiling && !kPageProfiling) {
    heap_profiler::HeapProfiler::global().record_alloc(ptr, sizeof(T) * size);
  }

  return tbb_unique_ptr<T>(ptr, tbb_deleter<T>{});
}
} // namespace kaminpar::parallel
