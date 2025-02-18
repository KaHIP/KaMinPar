#include <vector>

#include <tbb/scalable_allocator.h>

namespace kaminpar {

#ifdef KAMINPAR_ENABLE_HEAP_PROFILING
template <typename T> using ScalableVector = std::vector<T>;
#else
template <typename T> using ScalableVector = std::vector<T, tbb::scalable_allocator<T>>;
#endif

} // namespace kaminpar
