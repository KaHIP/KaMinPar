/*******************************************************************************
 * Static array that can reset used elements in O(# of used elements), similar to FastResetArray.
 * But instead of marking an entry as used when it is accessed, it is marked by the user, otherwise
 * multiple concurrent accesses to the same value would mark the value as used multiple times.
 *
 * @file:   concurrent_fast_reset_array.h
 * @author: Daniel Salwasser
 * @date:   29.10.2023
 ******************************************************************************/
#pragma once

#include <vector>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include "kaminpar-common/datastructures/cache_aligned_vector.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/ranges.h"

namespace kaminpar {

/*!
 * A static array that can reset used elements in O(# of used elements).
 *
 * @tparam Value The type of value to store.
 * @tparam Size The type of index to use to access and store values.
 */
template <typename Value, typename Size = std::size_t> class ConcurrentFastResetArray {
public:
  using value_type = Value;
  using reference = Value &;
  using size_type = Size;

  /*!
   * Constructs a new ConcurrentFastResetArray.
   *
   * @param capacity The capacity of the map, i.e., the number of values that can be stored.
   */
  explicit ConcurrentFastResetArray(const std::size_t capacity = 0) : _data(capacity) {
    RECORD_DATA_STRUCT(capacity * sizeof(value_type), _struct);
    _used_entries_tls.resize(tbb::this_task_arena::max_concurrency());
  }

  /*!
   * Returns the capacity of this array.
   *
   * @return The capacity of this array.
   */
  [[nodiscard]] std::size_t capacity() const {
    return _data.size();
  }

  /*!
   * Returns the thread-local vector of used entries.
   *
   * @return The thread-local vector of used entries.
   */
  [[nodiscard]] std::vector<size_type> &local_used_entries() {
    return _used_entries_tls[tbb::this_task_arena::current_thread_index()];
  }

  /*!
   * Accesses a value at a position.
   *
   * @param pos The position of the value in the map to return. It should be greater or equal then
   * zero and less then the set capacity.
   * @return A reference to the value at the position.
   */
  [[nodiscard]] reference operator[](const size_type pos) {
    KASSERT(pos < _data.size());
    return _data[pos];
  }

  /*!
   * Resizes the array.
   *
   * @param capacity The new capacity of the map, i.e., the number of values that can be stored.
   */
  void resize(const size_type capacity) {
    IF_HEAP_PROFILING(_struct->size = std::max(_struct->size, capacity * sizeof(value_type)));
    _data.resize(capacity);
    _used_entries_tls.resize(tbb::this_task_arena::max_concurrency());
  }

  /*!
   * Frees the memory used by this data structure.
   */
  void free() {
    _data.free();

    _used_entries_tls.clear();
    _used_entries_tls.shrink_to_fit();
  }

  /*!
   * Iterates over all thread-local vectors of used entries and clears them afterwards.
   *
   * @param l The function object that is invoked with a thread-local vector of used entries before
   * its cleared.
   */
  template <typename Lambda> void iterate_and_reset(Lambda &&l) {
    tbb::parallel_for<std::size_t>(0, _used_entries_tls.size(), [&](const auto i) {
      auto &local_used_entries = _used_entries_tls[i];
      if (local_used_entries.empty()) {
        return;
      }

      auto local_entries = TransformedIotaRange(
          static_cast<std::size_t>(0),
          local_used_entries.size(),
          [this, &local_used_entries](const std::size_t j) {
            const std::size_t pos = local_used_entries[j];
            return std::make_pair(pos, _data[pos]);
          }
      );
      l(i, local_entries);

      for (const size_type pos : local_used_entries) {
        _data[pos] = Value();
      }
      local_used_entries.clear();
    });
  }

private:
  StaticArray<value_type> _data;
  CacheAlignedVector<std::vector<size_type>> _used_entries_tls;

  IF_HEAP_PROFILING(heap_profiler::DataStructure *_struct);
};

} // namespace kaminpar
