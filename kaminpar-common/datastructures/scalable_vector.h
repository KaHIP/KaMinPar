#include <vector>

#include <tbb/scalable_allocator.h>

#include "kaminpar-common/datastructures/noinit_vector.h"

namespace kaminpar {
// @deprecated
template <typename T> using scalable_vector = std::vector<T, tbb::scalable_allocator<T>>;

// @deprecated
template <typename T>
using scalable_noinit_vector = std::vector<T, NoinitAllocator<T, tbb::scalable_allocator<T>>>;

template <typename T> using ScalableVector = scalable_vector<T>;

template <typename T> using ScalableNoinitVector = scalable_noinit_vector<T>;
} // namespace kaminpar
