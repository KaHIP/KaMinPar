/*******************************************************************************
 * Hand rolled priority queues that can grow.
 *
 * @file:   binary_heap.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <cstdlib>
#include <vector>

#include "kaminpar-common/datastructures/binary_heap.h"

namespace kaminpar {

//! Dynamic binary heap, not addressable
template <
    typename ID,
    typename Key,
    template <typename> typename Comparator,
    template <typename...> typename Container = std::vector>
class DynamicBinaryHeap {
  static constexpr std::size_t kTreeArity = 4;

public:
  struct HeapElement {
    ID id;
    Key key;
  };

  explicit DynamicBinaryHeap(const std::size_t initial_capacity = 0) {
    _heap.reserve(initial_capacity);
  }

  DynamicBinaryHeap(const DynamicBinaryHeap &) = delete;
  DynamicBinaryHeap &operator=(const DynamicBinaryHeap &) = delete;

  DynamicBinaryHeap(DynamicBinaryHeap &&) noexcept = default;
  DynamicBinaryHeap &operator=(DynamicBinaryHeap &&) noexcept = default;

  void push(const ID id, const Key key) {
    _heap.push_back({id, key});
    sift_up(_heap.size() - 1);
  }

  [[nodiscard]] ID peek_id() const {
    KASSERT(!empty());
    return _heap.front().id;
  }

  [[nodiscard]] Key peek_key() const {
    KASSERT(!empty());
    return _heap.front().key;
  }

  void pop() {
    std::swap(_heap.front(), _heap.back());
    _heap.pop_back();
    sift_down(0);
  }

  void clear() {
    _heap.clear();
  }

  [[nodiscard]] std::size_t size() const {
    return _heap.size();
  }

  [[nodiscard]] bool empty() const {
    return _heap.empty();
  }

  const auto &elements() {
    return _heap;
  }

private:
  void sift_up(std::size_t pos) {
    while (pos != 0) {
      const std::size_t parent = (pos - 1) / kTreeArity;
      if (_comparator(_heap[parent].key, _heap[pos].key)) {
        std::swap(_heap[pos], _heap[parent]);
      }
      pos = parent;
    }
  }

  void sift_down(std::size_t pos) {
    while (true) {
      const std::size_t first_child = kTreeArity * pos + 1;
      if (first_child >= size())
        return;

      std::size_t smallest_child = first_child;
      for (std::size_t c = first_child + 1; c < std::min(kTreeArity * pos + kTreeArity + 1, size());
           ++c) {
        if (_comparator(_heap[smallest_child].key, _heap[c].key)) {
          smallest_child = c;
        }
      }

      if (_comparator(_heap[smallest_child].key, _heap[pos].key) ||
          _heap[smallest_child].key == _heap[pos].key) {
        return;
      }

      std::swap(_heap[pos], _heap[smallest_child]);
      pos = smallest_child;
    }
  }

  Container<HeapElement> _heap{};
  Comparator<Key> _comparator{};
};

template <typename ID, typename Key, template <typename...> typename Container = std::vector>
using DynamicBinaryMaxHeap =
    DynamicBinaryHeap<ID, Key, binary_heap::max_heap_comparator, Container>;

template <typename ID, typename Key, template <typename...> typename Container = std::vector>
using DynamicBinaryMinHeap =
    DynamicBinaryHeap<ID, Key, binary_heap::min_heap_comparator, Container>;

} // namespace kaminpar
