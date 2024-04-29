#include <vector>

#include <tbb/scalable_allocator.h>

namespace kaminpar {
#ifdef KAMINPAR_ENABLE_HEAP_PROFILING
// @deprecated
template <typename T> using scalable_vector = std::vector<T>;
#else
// @deprecated
template <typename T> using scalable_vector = std::vector<T, tbb::scalable_allocator<T>>;
#endif

template <typename T> using ScalableVector = scalable_vector<T>;
} // namespace kaminpar
