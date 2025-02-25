#include <vector>

#ifdef KAMINPAR_ENABLE_TBB_MALLOC
#include <tbb/scalable_allocator.h>
#endif // KAMINPAR_ENABLE_TBB_MALLOC

namespace kaminpar {

#ifdef KAMINPAR_ENABLE_TBB_MALLOC
#ifdef KAMINPAR_ENABLE_HEAP_PROFILING
template <typename T> using ScalableVector = std::vector<T>;
#else  // KAMINPAR_ENABLE_HEAP_PROFILING
template <typename T> using ScalableVector = std::vector<T, tbb::scalable_allocator<T>>;
#endif // KAMINPAR_ENABLE_HEAP_PROFILING
#else  // KAMINPAR_ENABLE_TBB_MALLOC
template <typename T> using ScalableVector = std::vector<T>;
#endif // KAMINPAR_ENABLE_TBB_MALLOC

} // namespace kaminpar
