/*******************************************************************************
 * @file:   tbb_malloc.h
 * @author: Daniel Seemaier
 * @date:   30.03.2022
 * @brief:  Memory allocation functions that use the TBB scalable allocator.
 ******************************************************************************/
#pragma once

#include <memory>

#include <tbb/scalable_allocator.h>

namespace kaminpar::parallel {
template <typename T> struct tbb_deleter {
  void operator()(T *p) {
#ifdef KAMINPAR_ENABLE_HEAP_PROFILING
    std::free(p);
#else
    scalable_free(p);
#endif
  }
};

template <typename T> using tbb_unique_ptr = std::unique_ptr<T, tbb_deleter<T>>;

template <typename T> tbb_unique_ptr<T> make_unique(const std::size_t size) {
#ifdef KAMINPAR_ENABLE_HEAP_PROFILING
  T *ptr = static_cast<T *>(std::malloc(sizeof(T) * size));
#else
  T *ptr = static_cast<T *>(scalable_malloc(sizeof(T) * size));
#endif
  return tbb_unique_ptr<T>(ptr, tbb_deleter<T>{});
}

template <typename T, typename... Args> tbb_unique_ptr<T> make_unique(Args &&...args) {
#ifdef KAMINPAR_ENABLE_HEAP_PROFILING
  void *memory = static_cast<T *>(std::malloc(sizeof(T)));
#else
  void *memory = static_cast<T *>(scalable_malloc(sizeof(T)));
#endif
  T *ptr = new (memory) T(std::forward<Args...>(args)...);
  return tbb_unique_ptr<T>(ptr, tbb_deleter<T>{});
}
} // namespace kaminpar::parallel
