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

#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/parallel/aligned_element.h"

namespace kaminpar {

/*!
 * A static array that can reset used elements in O(# of used elements).
 *
 * @tparam Value The type of value to store.
 * @tparam Size The type of index to use to access and save values.
 */
template <typename Value, typename Size = std::size_t> class ConcurrentFastResetArray {
public:
  using value_type = Value;
  using reference = Value &;
  using size_type = Size;

  /*!
   * Constructs a new ConcurrentFastResetArray.
   *
   * @param capacity The capacity of the map, i.e. the amount of values to possibly save.
   */
  explicit ConcurrentFastResetArray(const std::size_t capacity = 0) : _data(capacity) {
    RECORD_DATA_STRUCT(capacity * sizeof(value_type), _struct);
    _used_entries_tls.resize(tbb::this_task_arena::max_concurrency());
  }

  /*!
   * Returns the thread-local vector of used entries.
   *
   * @return The thread-local vector of used entries.
   */
  [[nodiscard]] std::vector<size_type> &local_used_entries() {
    return _used_entries_tls[tbb::this_task_arena::current_thread_index()].vec;
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
   * Resized the array.
   *
   * @param capacity The new capacity of the map, i.e. the amount of values to possibly save.
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
    _data.clear();
    _data.shrink_to_fit();

    _used_entries_tls.clear();
    _used_entries_tls.shrink_to_fit();
  }

  /*!
   * Iterates over all thread-local vector of used entries and clears them afterwards.
   *
   * @param l The function object that is invoked with a thread-local vector of used entries before
   * they are cleared.
   */
  template <typename Lambda> void iterate_and_reset(Lambda &&l) {
    tbb::parallel_for<std::size_t>(0, _used_entries_tls.size(), [&](const auto i) {
      l(i, _used_entries_tls[i]);

      for (const size_type pos : _used_entries_tls[i]) {
        _data[pos] = Value();
      }

      _used_entries_tls[i].clear();
    });
  }

private:
  std::vector<value_type> _data;
  std::vector<parallel::AlignedVec<std::vector<size_type>>> _used_entries_tls;

  IF_HEAP_PROFILING(heap_profiler::DataStructure *_struct);
};

} // namespace kaminpar
