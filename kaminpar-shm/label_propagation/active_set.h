/*******************************************************************************
 * Standalone active set for label propagation.
 *
 * Tracks which nodes need to be processed in the next iteration of label
 * propagation. Nodes are marked active when a neighbor changes cluster, and
 * marked inactive when they are processed.
 *
 * The template parameter `kEnabled` controls whether the active set actually
 * does anything: when set to `false`, all operations become no-ops and the
 * compiler removes them entirely.
 *
 * @file:   active_set.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar {

template <bool kEnabled> class ActiveSet {
public:
  void allocate(const std::size_t n) {
    if (_data.size() < n) {
      _data.resize(n);
    }
  }

  void free() {
    _data.free();
  }

  void reset(const std::size_t n) {
    allocate(n);
    for (std::size_t i = 0; i < n; ++i) {
      _data[i] = 1;
    }
  }

  void mark_active(const std::size_t u) {
    __atomic_store_n(&_data[u], 1, __ATOMIC_RELAXED);
  }

  void mark_inactive(const std::size_t u) {
    __atomic_store_n(&_data[u], 0, __ATOMIC_RELAXED);
  }

  [[nodiscard]] bool is_active(const std::size_t u) const {
    return __atomic_load_n(&_data[u], __ATOMIC_RELAXED);
  }

  StaticArray<std::uint8_t> take() {
    return std::move(_data);
  }

  void set(StaticArray<std::uint8_t> data) {
    _data = std::move(data);
  }

private:
  StaticArray<std::uint8_t> _data;
};

template <> class ActiveSet<false> {
public:
  void allocate(const std::size_t) {}
  void free() {}
  void reset(const std::size_t) {}
  void mark_active(const std::size_t) {}
  void mark_inactive(const std::size_t) {}

  [[nodiscard]] bool is_active(const std::size_t) const {
    return true;
  }

  StaticArray<std::uint8_t> take() {
    return {};
  }

  void set(StaticArray<std::uint8_t>) {}
};

} // namespace kaminpar
