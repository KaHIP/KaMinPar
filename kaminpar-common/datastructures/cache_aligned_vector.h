/*******************************************************************************
 * Vector that aligns to cache lines.
 *
 * @file:   cache_aligned_vector.h
 * @author: Daniel Seemaier
 * @date:   15.07.2022
 ******************************************************************************/
#pragma once

#include <vector>

#include <tbb/cache_aligned_allocator.h>

namespace kaminpar {
#ifdef KAMINPAR_ENABLE_HEAP_PROFILING
template <typename T> using CacheAlignedVector = std::vector<T>;
#else
template <typename T> using CacheAlignedVector = std::vector<T, tbb::cache_aligned_allocator<T>>;
#endif
} // namespace kaminpar
