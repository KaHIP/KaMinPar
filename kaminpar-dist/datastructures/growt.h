/*******************************************************************************
 * Include growt and suppress -Wpedantic warnings.
 *
 * @file:   growt.h
 * @author: Daniel Seemaier
 * @date:   30.09.21
 ******************************************************************************/
#pragma once

#include <type_traits>

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-dist/dkaminpar.h"

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpedantic"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wuninitialized"
#endif

#include <allocator/alignedallocator.hpp>
#include <data-structures/table_config.hpp>
#include <utils/hash/murmur2_hash.hpp>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif

namespace kaminpar::dist::growt {
using DefaultHasherType = utils_tm::hash_tm::murmur2_hash;
using DefaultAllocatorType = ::growt::AlignedAllocator<>;

namespace internal {
// workaround 32 bit value bug in growt
template <typename Type>
using Ensure64BitType =
    std::conditional_t<std::numeric_limits<Type>::is_signed, GlobalNodeWeight, GlobalNodeID>;
} // namespace internal

template <typename Value>
using GlobalNodeIDMap = typename ::growt::table_config<
    GlobalNodeID,
    internal::Ensure64BitType<Value>,
    DefaultHasherType,
    DefaultAllocatorType,
    hmod::growable,
    hmod::deletion>::table_type;

using StaticGhostNodeMapping = typename ::growt::
    table_config<GlobalNodeID, GlobalNodeID, DefaultHasherType, DefaultAllocatorType>::table_type;

template <typename Map, typename Lambda> void pfor_map(Map &map, Lambda &&lambda) {
  std::atomic_size_t counter = 0;

#pragma omp parallel default(none) shared(map, counter, lambda)
  {
    const std::size_t capacity = map.capacity();
    std::size_t cur_block = counter.fetch_add(4096);

    while (cur_block < capacity) {
      auto it = map.range(cur_block, cur_block + 4096);
      for (; it != map.range_end(); ++it) {
        lambda((*it).first, (*it).second);
      }
      cur_block = counter.fetch_add(4096);
    }
  }
}

template <typename Handles, typename Lambda> void pfor_handles(Handles &handles, Lambda &&lambda) {
  std::atomic_size_t counter = 0;

#pragma omp parallel default(none) shared(handles, counter, lambda)
  {
    auto &handle = handles.local();
    const std::size_t capacity = handle.capacity();
    std::size_t cur_block = counter.fetch_add(4096);

    while (cur_block < capacity) {
      auto it = handle.range(cur_block, cur_block + 4096);
      for (; it != handle.range_end(); ++it) {
        lambda((*it).first, (*it).second);
      }
      cur_block = counter.fetch_add(4096);
    }
  }
}
} // namespace kaminpar::dist::growt
