/*******************************************************************************
 * Include growt and suppress -Wpedantic warnings.
 *
 * @file:   growt.h
 * @author: Daniel Seemaier
 * @date:   30.09.21
 ******************************************************************************/
#pragma once

#include <optional>
#include <type_traits>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include "kaminpar-dist/dkaminpar.h"

#include "kaminpar-common/assert.h"

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

// Workaround 32 bit value bug in growt
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

namespace internal {
using GhostMap = typename ::growt::
    table_config<GlobalNodeID, GlobalNodeID, DefaultHasherType, DefaultAllocatorType>::table_type;
}

class MutableGhostNodeMapping;

// The only public reads return values. No iterator, mapped reference or mutable handle escapes.
class StaticGhostNodeMapping {
public:
  static constexpr std::size_t kSlotSize = sizeof(internal::GhostMap::atomic_slot_type);

  StaticGhostNodeMapping() : _map(0) {}
  StaticGhostNodeMapping(const StaticGhostNodeMapping &) = delete;
  StaticGhostNodeMapping &operator=(const StaticGhostNodeMapping &) = delete;
  StaticGhostNodeMapping(StaticGhostNodeMapping &&) noexcept = default;
  StaticGhostNodeMapping &operator=(StaticGhostNodeMapping &&) noexcept = default;

  [[nodiscard]] std::optional<GlobalNodeID> lookup(const GlobalNodeID key) const {
    return _map.find_quiescent(key);
  }

  [[nodiscard]] bool contains(const GlobalNodeID key) const {
    return lookup(key).has_value();
  }

  [[nodiscard]] GlobalNodeID at(const GlobalNodeID key) const {
    const auto value = lookup(key);
    KASSERT(value.has_value(), "Missing frozen ghost node mapping");
    return *value;
  }

  [[nodiscard]] std::size_t capacity() const {
    return _map.capacity();
  }

private:
  friend class MutableGhostNodeMapping;
  explicit StaticGhostNodeMapping(internal::GhostMap map) : _map(std::move(map)) {}
  internal::GhostMap _map;
};

class MutableGhostNodeMapping {
public:
  explicit MutableGhostNodeMapping(const std::size_t capacity) : _map(std::in_place, capacity) {}
  MutableGhostNodeMapping(const MutableGhostNodeMapping &) = delete;
  MutableGhostNodeMapping &operator=(const MutableGhostNodeMapping &) = delete;
  MutableGhostNodeMapping(MutableGhostNodeMapping &&) = delete;
  MutableGhostNodeMapping &operator=(MutableGhostNodeMapping &&) = delete;

  // Concurrent insertions use Growt's original atomic loads/CAS. No handle or iterator escapes.
  bool insert(const GlobalNodeID key, const GlobalNodeID value) {
    KASSERT(_map.has_value(), "Ghost map builder was consumed");
    return _map->insert(key, value).second;
  }

  // Precondition: join every inserting task before freezing. Moving transfers sole ownership;
  // resetting the optional makes accidental post-freeze reuse fail before touching the table.
  [[nodiscard]] StaticGhostNodeMapping freeze() && {
    KASSERT(_map.has_value(), "Ghost map builder was consumed");
    StaticGhostNodeMapping frozen(std::move(*_map));
    _map.reset();
    return frozen;
  }

private:
  std::optional<internal::GhostMap> _map;
};

template <typename Map, typename Lambda> void pfor_map(Map &map, Lambda &&lambda) {
  std::atomic_size_t counter = 0;

  tbb::parallel_for<int>(0, tbb::this_task_arena::max_concurrency(), [&](const int) {
    const std::size_t capacity = map.capacity();
    std::size_t cur_block = counter.fetch_add(4096);

    while (cur_block < capacity) {
      auto it = map.range(cur_block, cur_block + 4096);
      for (; it != map.range_end(); ++it) {
        lambda((*it).first, (*it).second);
      }
      cur_block = counter.fetch_add(4096);
    }
  });
}

template <typename Handles, typename Lambda> void pfor_handles(Handles &handles, Lambda &&lambda) {
  std::atomic_size_t counter = 0;

  tbb::parallel_for<int>(0, tbb::this_task_arena::max_concurrency(), [&](const int) {
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
  });
}

} // namespace kaminpar::dist::growt
