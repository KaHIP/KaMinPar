/*******************************************************************************
 * Addressable growable double-ended priority queue.
 *
 * @file:   addressable_dynamic_de_heap.h
 * @author: Daniel Seemaier
 * @author: Daniel Salwasser
 * @date:   01.01.2025
 ******************************************************************************/
#pragma once

#include <cstdlib>
#include <vector>

#include "kaminpar-common/datastructures/dynamic_binary_forest.h"

namespace kaminpar {

template <typename ID, typename Key, template <typename...> typename Container = std::vector>
class AddressableDynamicBinaryMinMaxForest {
  using Self = AddressableDynamicBinaryMinMaxForest;

public:
  AddressableDynamicBinaryMinMaxForest() {}

  AddressableDynamicBinaryMinMaxForest(const std::size_t capacity, const std::size_t heaps)
      : _max_forest(capacity, heaps),
        _min_forest(capacity, heaps) {}

  AddressableDynamicBinaryMinMaxForest(const Self &) = delete;
  AddressableDynamicBinaryMinMaxForest &operator=(const Self &) = delete;

  AddressableDynamicBinaryMinMaxForest(Self &&) noexcept = default;
  AddressableDynamicBinaryMinMaxForest &operator=(Self &&) noexcept = default;

  void init(const std::size_t capacity, const std::size_t heaps) {
    _max_forest.init(capacity, heaps);
    _min_forest.init(capacity, heaps);
  }

  [[nodiscard]] inline std::size_t capacity() const {
    return _max_forest.capacity();
  }

  [[nodiscard]] inline std::size_t size() const {
    return _max_forest.size();
  }

  [[nodiscard]] inline std::size_t size(const std::size_t heap) const {
    return _max_forest.size(heap);
  }

  void push(const std::size_t heap, const ID id, const Key key) {
    _max_forest.push(heap, id, key);
    _min_forest.push(heap, id, key);
  }

  void change_priority(const std::size_t heap, const ID id, const Key key) {
    _max_forest.change_priority(heap, id, key);
    _min_forest.change_priority(heap, id, key);
  }

  Key key(const std::size_t heap, const ID id) const {
    KASSERT(_max_forest.contains(id));
    KASSERT(_min_forest.contains(id));
    KASSERT(_max_forest.key(heap, id) == _min_forest.key(heap, id));
    return _max_forest.key(heap, id);
  }

  [[nodiscard]] bool contains(const ID id) const {
    KASSERT(_max_forest.contains(id) == _min_forest.contains(id));
    return _max_forest.contains(id);
  }

  bool empty(const std::size_t heap) const {
    KASSERT(_max_forest.empty(heap) == _min_forest.empty(heap));
    return _max_forest.empty(heap);
  }

  [[nodiscard]] ID peek_min_id(const std::size_t heap) const {
    return _min_forest.peek_id(heap);
  }

  [[nodiscard]] ID peek_max_id(const std::size_t heap) const {
    return _max_forest.peek_id(heap);
  }

  [[nodiscard]] Key peek_min_key(const std::size_t heap) const {
    return _min_forest.peek_key(heap);
  }

  [[nodiscard]] Key peek_max_key(const std::size_t heap) const {
    return _max_forest.peek_key(heap);
  }

  void remove(const std::size_t heap, const ID id) {
    _max_forest.remove(heap, id);
    _min_forest.remove(heap, id);
  }

  void pop_min(const std::size_t heap) {
    _max_forest.remove(heap, _min_forest.peek_id(heap));
    _min_forest.pop(heap);
  }

  void pop_max(const std::size_t heap) {
    KASSERT(!_min_forest.empty(heap));
    KASSERT(!_max_forest.empty(heap));

    _min_forest.remove(heap, _max_forest.peek_id(heap));
    _max_forest.pop(heap);
  }

  void clear() {
    _min_forest.clear();
    _max_forest.clear();
  }

  void clear(const std::size_t heap) {
    _min_forest.clear(heap);
    _max_forest.clear(heap);
  }

  const auto &elements(const std::size_t heap) const {
    return _max_forest.elements(heap);
  }

private:
  DynamicBinaryMaxForest<ID, Key, Container> _max_forest;
  DynamicBinaryMinForest<ID, Key, Container> _min_forest;
};

} // namespace kaminpar
