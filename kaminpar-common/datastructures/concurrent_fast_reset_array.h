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

#include <tbb/concurrent_vector.h>

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
  explicit ConcurrentFastResetArray(const std::size_t capacity = 0) : _data(capacity) {}

  /*!
   * Accesses a value at a position.
   *
   * @param pos The position of the value in the map to return. It should be greater or equal then
   * zero and less then the set capacity.
   * @return A reference to the value at the position.
   */
  reference operator[](const size_type pos) {
    KASSERT(pos < _data.size());
    return _data[pos];
  }

  /*!
   * Marks a value as used.
   *
   * @param pos The position of the value in the map to mark as used.
   */
  void mark_as_used(const size_type pos) {
    _used_entries.push_back(pos);
  }

  /*!
   * Returns the pairs of position and value that have been marked as used.
   *
   * @returns Returns the pairs of position and value that have been marked as used.
   */
  [[nodiscard]] auto entries() {
    return TransformedRange(
        _used_entries.begin(),
        _used_entries.end(),
        [this](const size_type entry) -> std::pair<Size, value_type> {
          return std::make_pair(entry, _data[entry]);
        }
    );
  }

  /*!
   * Resets all values in the map and marks all values as unused.
   */
  void clear() {
    for (const size_type pos : _used_entries) {
      _data[pos] = Value();
    }

    _used_entries.clear();
  }

private:
  std::vector<value_type> _data;
  tbb::concurrent_vector<size_type> _used_entries{};
};

} // namespace kaminpar
