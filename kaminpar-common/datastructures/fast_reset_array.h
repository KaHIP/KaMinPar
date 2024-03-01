/*******************************************************************************
 * @file:   fast_reset_array.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Static array that can reset used elements in O(# of used elements),
 * where # of used elements might be much smaller than the array's capacity.
 ******************************************************************************/
#pragma once

#include <utility>
#include <vector>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/ranges.h"

namespace kaminpar {
template <typename Value, typename Size = std::size_t> class FastResetArray {
public:
  using value_type = Value;
  using reference = Value &;
  using const_reference = const Value &;
  using size_type = Size;

  explicit FastResetArray(const std::size_t capacity = 0) : _data(capacity) {}

  FastResetArray(const FastResetArray &) = delete;
  FastResetArray &operator=(const FastResetArray &) = delete;
  FastResetArray(FastResetArray &&) noexcept = default;
  FastResetArray &operator=(FastResetArray &&) noexcept = default;

  reference operator[](const size_type pos) {
    KASSERT(pos < _data.size());
    if (_data[pos] == Value()) {
      _used_entries.push_back(pos);
    }
    return _data[pos];
  }
  const_reference operator[](const size_type pos) const {
    return _data[pos];
  }

  const_reference get(const size_type pos) const {
    return _data[pos];
  }
  void set(const size_type pos, const_reference new_value) {
    (*this)[pos] = new_value;
  }

  [[nodiscard]] bool exists(const size_type pos) const {
    return _data[pos] != Value();
  }

  [[nodiscard]] std::vector<size_type> &used_entry_ids() {
    return _used_entries;
  }

  [[nodiscard]] auto used_entry_values() {
    return TransformedRange(
        used_entry_ids().begin(),
        used_entry_ids().end(),
        [this](const std::size_t entry) -> value_type { return _data[entry]; }
    );
  }

  [[nodiscard]] auto entries() {
    return TransformedRange(
        used_entry_ids().begin(),
        used_entry_ids().end(),
        [this](const std::size_t entry) -> std::pair<Size, value_type> {
          return std::make_pair(static_cast<Size>(entry), _data[entry]);
        }
    );
  }

  void clear() {
    for (const std::size_t pos : _used_entries) {
      _data[pos] = Value();
    }
    _used_entries.clear();
  }

  [[nodiscard]] bool empty() const {
    return _used_entries.empty();
  }
  [[nodiscard]] std::size_t size() const {
    return _used_entries.size();
  }
  [[nodiscard]] std::size_t capacity() const {
    return _data.size();
  }
  void resize(const std::size_t capacity) {
    _data.resize(capacity);
  }

  [[nodiscard]] std::size_t memory_in_kb() const {
    return _data.size() * sizeof(value_type) / 1000;
  }

private:
  std::vector<value_type> _data;
  std::vector<size_type> _used_entries{};
};
} // namespace kaminpar
