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

namespace kaminpar::parallel {
template <typename T> struct tbb_deleter {
  void operator()(T *p) {
    scalable_free(p);
  }
};

template <typename T> using tbb_unique_ptr = std::unique_ptr<T, tbb_deleter<T>>;

template <typename T> tbb_unique_ptr<T> make_unique(const std::size_t size) {
  auto nbytes = sizeof(T) * size;
  T *ptr = static_cast<T *>(scalable_malloc(nbytes));
  KASSERT(
      ptr != nullptr,
      "probably out of memory after attemping to allocate " << nbytes << " bytes",
      assert::light
  );
  return tbb_unique_ptr<T>(ptr, tbb_deleter<T>{});
}

template <typename T, typename... Args> tbb_unique_ptr<T> make_unique(Args &&...args) {
  void *memory = static_cast<T *>(scalable_malloc(sizeof(T)));
  T *ptr = new (memory) T(std::forward<Args...>(args)...);
  return tbb_unique_ptr<T>(ptr, tbb_deleter<T>{});
}
} // namespace kaminpar::parallel
