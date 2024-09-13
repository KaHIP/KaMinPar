/*******************************************************************************
 * Vector that aligns to cache lines.
 *
 * @file:   cache_aligned_vector.h
 * @author: Daniel Seemaier
 * @date:   15.07.2022
 ******************************************************************************/
#include <vector>

#include <tbb/cache_aligned_allocator.h>

namespace kaminpar {
template <typename T> using cache_aligned_vector = std::vector<T, tbb::cache_aligned_allocator<T>>;
}
