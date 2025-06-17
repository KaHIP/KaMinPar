/*******************************************************************************
 * Forest of growable priority queues: there is a fixed number of PQs, but each
 * element can only be in one PQ at a time.
 *
 * @file:   dynamic_binary_forest.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <vector>

#include "kaminpar-common/datastructures/binary_heap.h"

namespace kaminpar {

template <
    typename ID,
    typename Key,
    template <typename> typename Comparator,
    template <typename...> typename Container = std::vector>
class DynamicBinaryForest {
  static constexpr std::size_t kTreeArity = 4;
  static constexpr ID kInvalidID = std::numeric_limits<ID>::max();

public:
  struct HeapElement {
    ID id;
    Key key;
  };

  explicit DynamicBinaryForest() {}

  explicit DynamicBinaryForest(const std::size_t capacity, const std::size_t heaps)
      : _id_pos(capacity, kInvalidID),
        _heaps(heaps) {}

  DynamicBinaryForest(const DynamicBinaryForest &) = delete;
  DynamicBinaryForest &operator=(const DynamicBinaryForest &) = delete;

  DynamicBinaryForest(DynamicBinaryForest &&) noexcept = default;
  DynamicBinaryForest &operator=(DynamicBinaryForest &&) noexcept = default;

  void init(const std::size_t capacity, const std::size_t heaps) {
    _id_pos.resize(capacity, kInvalidID);
    _heaps.resize(heaps);
  }

  [[nodiscard]] std::size_t capacity() const {
    return _id_pos.size();
  }

  [[nodiscard]] bool contains(const ID id) const {
    KASSERT(static_cast<std::size_t>(id) < _id_pos.size());
    return _id_pos[id] != kInvalidID;
  }

  void push(const std::size_t heap, const ID id, const Key key) {
    KASSERT(key != Comparator<Key>::kMinValue);
    KASSERT(_comparator(key, Comparator<Key>::kMinValue));

    _heaps[heap].push_back({id, key});
    _id_pos[id] = _heaps[heap].size() - 1;
    sift_up(heap, _id_pos[id]);
  }

  [[nodiscard]] ID peek_id(const std::size_t heap) const {
    KASSERT(!empty(heap));
    return _heaps[heap].front().id;
  }

  [[nodiscard]] Key peek_key(const std::size_t heap) const {
    KASSERT(!empty(heap));
    return _heaps[heap].front().key;
  }

  void remove(const std::size_t heap, const ID id) {
    decrease_priority(heap, id, Comparator<Key>::kMinValue);
    pop(heap);
  }

  void pop(const std::size_t heap) {
    _id_pos[_heaps[heap].back().id] = 0;
    _id_pos[_heaps[heap].front().id] = kInvalidID;
    std::swap(_heaps[heap].front(), _heaps[heap].back());
    _heaps[heap].pop_back();
    sift_down(heap, 0);
  }

  void push_or_change_priority(const std::size_t heap, const ID id, const Key &new_key) {
    if (contains(heap, id)) {
      change_priority(heap, id, new_key);
    } else {
      push(heap, id, new_key);
    }
  }

  void change_priority(const std::size_t heap, const ID id, const Key &new_key) {
    const Key &old_key = key(heap, id);
    if (_comparator(new_key, old_key)) {
      increase_priority(heap, id, new_key);
    } else if (_comparator(old_key, new_key)) {
      decrease_priority(heap, id, new_key);
    }
  }

  // *NOTE*: "decrease" is in respect to the priority
  // e.g., decreasing an integral key in a BinaryMinHeap *increases* its
  // priority hence, *increase_priority* must be called instead of
  // decrease_priority
  void decrease_priority(const std::size_t heap, const ID id, const Key &new_key) {
    KASSERT(contains(id));
    KASSERT(_comparator(key(heap, id), new_key));

    _heaps[heap][_id_pos[id]].key = new_key;
    sift_up(heap, _id_pos[id]);
  }

  void increase_priority(const std::size_t heap, const ID id, const Key &new_key) {
    KASSERT(contains(id));
    KASSERT(_comparator(new_key, key(heap, id)));

    _heaps[heap][_id_pos[id]].key = new_key;
    sift_down(heap, _id_pos[id]);
  }

  void clear(const std::size_t heap) {
    for (std::size_t i = 0; i < size(heap); ++i) {
      _id_pos[_heaps[heap][i].id] = kInvalidID;
    }
    _heaps[heap].clear();
  }

  void clear() {
    for (std::size_t i = 0; i < _heaps.size(); ++i) {
      clear(i);
    }
  }

  [[nodiscard]] std::size_t size(const std::size_t heap) const {
    return _heaps[heap].size();
  }

  [[nodiscard]] std::size_t size() const {
    std::size_t total_size = 0;
    for (std::size_t heap = 0; heap < _heaps.size(); ++heap) {
      total_size += size(heap);
    }
    return total_size;
  }

  [[nodiscard]] bool empty(const std::size_t heap) const {
    return _heaps[heap].empty();
  }

  const auto &elements(const std::size_t heap) const {
    return _heaps[heap];
  }

  Key key(const std::size_t heap, const std::size_t pos) const {
    KASSERT(heap < _heaps.size());
    KASSERT(pos < _id_pos.size());
    KASSERT(_id_pos[pos] < _heaps[heap].size());

    return _heaps[heap][_id_pos[pos]].key;
  }

private:
  void sift_up(const std::size_t heap, std::size_t pos) {
    while (pos != 0) {
      const std::size_t parent = (pos - 1) / kTreeArity;
      if (_comparator(_heaps[heap][parent].key, _heaps[heap][pos].key)) {
        swap(heap, pos, parent);
      }
      pos = parent;
    }
  }

  void sift_down(const std::size_t heap, std::size_t pos) {
    while (true) {
      const std::size_t first_child = kTreeArity * pos + 1;
      if (first_child >= size(heap))
        return;

      std::size_t smallest_child = first_child;
      for (std::size_t c = first_child + 1;
           c < std::min(kTreeArity * pos + kTreeArity + 1, size(heap));
           ++c) {
        if (_comparator(_heaps[heap][smallest_child].key, _heaps[heap][c].key)) {
          smallest_child = c;
        }
      }

      if (_comparator(_heaps[heap][smallest_child].key, _heaps[heap][pos].key) ||
          _heaps[heap][smallest_child].key == _heaps[heap][pos].key) {
        return;
      }

      swap(heap, pos, smallest_child);
      pos = smallest_child;
    }
  }

  void swap(const std::size_t heap, const std::size_t a, const std::size_t b) {
    std::swap(_id_pos[_heaps[heap][a].id], _id_pos[_heaps[heap][b].id]);
    std::swap(_heaps[heap][a], _heaps[heap][b]);
  }

  Container<std::size_t> _id_pos;
  std::vector<std::vector<HeapElement>> _heaps;
  Comparator<Key> _comparator{};
};

template <typename ID, typename Key, template <typename...> typename Container = std::vector>
using DynamicBinaryMaxForest =
    DynamicBinaryForest<ID, Key, binary_heap::max_heap_comparator, Container>;

template <typename ID, typename Key, template <typename...> typename Container = std::vector>
using DynamicBinaryMinForest =
    DynamicBinaryForest<ID, Key, binary_heap::min_heap_comparator, Container>;

} // namespace kaminpar
