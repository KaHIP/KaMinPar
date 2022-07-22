#include <vector>

#include <tbb/scalable_allocator.h>

namespace kaminpar {
template <typename T>
using scalable_vector = std::vector<T, tbb::scalable_allocator<T>>;
}
