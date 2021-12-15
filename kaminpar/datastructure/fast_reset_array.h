#pragma once

#include "definitions.h"
#include "utility/ranges.h"

#include <utility>
#include <vector>

namespace kaminpar {
template<typename T>
class FastResetArray {
public:
  using value_type = T;
  using size_type = std::size_t;
  using reference = T &;
  using const_reference = const T &;

  explicit FastResetArray(const std::size_t capacity = 0) : _data(capacity) {}

  FastResetArray(const FastResetArray &) = delete;
  FastResetArray &operator=(const FastResetArray &) = delete;
  FastResetArray(FastResetArray &&) noexcept = default;
  FastResetArray &operator=(FastResetArray &&) noexcept = default;

  reference operator[](const size_type pos) {
    ASSERT(pos < _data.size());
    if (_data[pos] == value_type()) { _used_entries.push_back(pos); }
    return _data[pos];
  }
  const_reference operator[](const size_type pos) const { return _data[pos]; }

  const_reference get(const size_type pos) const { return _data[pos]; }
  void set(const size_type pos, const_reference new_value) { (*this)[pos] = new_value; }

  [[nodiscard]] std::vector<size_type> &used_entry_ids() { return _used_entries; }

  [[nodiscard]] auto used_entry_values() {
    return TransformedRange(used_entry_ids().begin(), used_entry_ids().end(),
                            [this](const std::size_t entry) -> T { return _data[entry]; });
  }

  [[nodiscard]] auto entries() {
    return TransformedRange(used_entry_ids().begin(), used_entry_ids().end(),
                            [this](const std::size_t entry) -> std::pair<std::size_t, T> {
                              return std::make_pair(entry, _data[entry]);
                            });
  }

  void clear() {
    for (const std::size_t pos : _used_entries) { _data[pos] = value_type(); }
    _used_entries.clear();
  }

  [[nodiscard]] bool empty() const { return _used_entries.empty(); }
  [[nodiscard]] std::size_t size() const { return _used_entries.size(); }
  [[nodiscard]] std::size_t capacity() const { return _data.size(); }
  void resize(const std::size_t capacity) { _data.resize(capacity); }

  [[nodiscard]] std::size_t memory_in_kb() const { return _data.size() * sizeof(T) / 1000; }

private:
  std::vector<T> _data;
  std::vector<size_type> _used_entries;
};
} // namespace kaminpar