/*******************************************************************************
 * Hand rolled priority queues.
 *
 * @file:   binary_heap.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <limits>
#include <utility>
#include <vector>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/scalable_vector.h"

namespace kaminpar {
namespace binary_heap {
template <typename Key> struct max_heap_comparator {
  constexpr static auto kMinValue = std::numeric_limits<Key>::max();
  bool operator()(Key a, Key b) {
    return b > a;
  }
};

template <typename Key> struct min_heap_comparator {
  constexpr static auto kMinValue = std::numeric_limits<Key>::lowest();
  bool operator()(Key a, Key b) {
    return a > b;
  }
};
} // namespace binary_heap

//! Addressable binary heap with fixed capacity.
/*
 * The next class is based on iq_queue from the RoutingKit project
 *
 * Copyright (c) 2016, RoutingKit contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer. Redistributions in binary
 * form must reproduce the above copyright notice, this list of conditions and
 * the following disclaimer in the documentation and/or other materials provided
 * with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
template <typename Key, template <typename> typename Comparator> class SharedBinaryHeap {
  using ID = std::size_t;
  static constexpr std::size_t kTreeArity = 4;

public:
  static constexpr ID kInvalidID = std::numeric_limits<ID>::max();

  explicit SharedBinaryHeap(const ID shared_capacity, std::size_t *handles) noexcept
      : _capacity(shared_capacity),
        _heap(),
        _id_pos(handles) {}

  SharedBinaryHeap(const SharedBinaryHeap &) = delete;
  SharedBinaryHeap &operator=(const SharedBinaryHeap &) = delete;

  SharedBinaryHeap(SharedBinaryHeap &&) noexcept = default;
  SharedBinaryHeap &operator=(SharedBinaryHeap &&) noexcept = default;

  [[nodiscard]] bool empty() const {
    return _heap.empty();
  }

  [[nodiscard]] std::size_t size() const {
    return _heap.size();
  }

  [[nodiscard]] std::size_t shared_capacity() const {
    return _capacity;
  }

  [[nodiscard]] bool contains(const ID id) const {
    KASSERT(id < shared_capacity());
    return _id_pos[id] != kInvalidID;
  }

  Key key(const ID id) const {
    KASSERT(contains(id));
    return _heap[_id_pos[id]].key;
  }

  [[nodiscard]] ID peek_id() const {
    KASSERT(!empty());
    return _heap.front().id;
  }

  Key peek_key() const {
    KASSERT(!empty());
    return _heap.front().key;
  }

  void remove(const ID id) {
    KASSERT(contains(id));
    decrease_priority(id, Comparator<Key>::kMinValue);
    pop();
    KASSERT(!contains(id));
  }

  void pop() {
    KASSERT(!empty());
    std::swap(_heap.front(), _heap.back());
    _id_pos[_heap.front().id] = 0;
    _id_pos[_heap.back().id] = kInvalidID;
    _heap.pop_back();
    sift_down(0);
  }

  void push(const ID id, const Key &key) {
    KASSERT(!contains(id));
    const std::size_t pos = size();
    _heap.emplace_back(id, key);
    _id_pos[id] = pos;
    sift_up(pos);
  }

  void push_or_change_priority(const ID id, const Key &new_key) {
    if (contains(id)) {
      change_priority(id, new_key);
    } else {
      push(id, new_key);
    }
  }

  void change_priority(const ID id, const Key &new_key) {
    const Key &old_key = key(id);
    if (_comparator(new_key, old_key)) {
      increase_priority(id, new_key);
    } else if (_comparator(old_key, new_key)) {
      decrease_priority(id, new_key);
    }
  }

  void clear() {
    for (const auto &entry : _heap) {
      _id_pos[entry.id] = kInvalidID;
    }
    _heap.clear();
  }

  // *NOTE*: "decrease" is in respect to the priority
  // e.g., decreasing an integral key in a BinaryMinHeap *increases* its
  // priority hence, *increase_priority* must be called instead of
  // decrease_priority
  void decrease_priority(const ID id, const Key &new_key) {
    KASSERT(contains(id));
    KASSERT(_comparator(key(id), new_key));
    _heap[_id_pos[id]].key = new_key;
    sift_up(_id_pos[id]);
  }

  // deprecated
  void decrease_priority_by(const ID id, const Key &delta) {
    KASSERT(contains(id));
    KASSERT(delta > 0);
    decrease_priority(id, key(id) - delta);
  }

  void increase_priority(const ID id, const Key &new_key) {
    KASSERT(contains(id));
    KASSERT(_comparator(new_key, key(id)));
    _heap[_id_pos[id]].key = new_key;
    sift_down(_id_pos[id]);
  }

  void resize(std::size_t) {
    throw std::runtime_error("cannot be resized");
  }

private:
  struct HeapElement {
    HeapElement() noexcept : id(kInvalidID), key(0) {}
    HeapElement(const ID id, const Key &key) noexcept : id(id), key(key) {}
    ID id;
    Key key;
  };

  void sift_up(std::size_t pos) {
    while (pos != 0) {
      const std::size_t parent = (pos - 1) / kTreeArity;
      if (_comparator(_heap[parent].key, _heap[pos].key)) {
        std::swap(_heap[pos], _heap[parent]);
        std::swap(_id_pos[_heap[pos].id], _id_pos[_heap[parent].id]);
      }
      pos = parent;
    }
  }

  void sift_down(std::size_t pos) {
    while (true) {
      const std::size_t first_child = kTreeArity * pos + 1;
      if (first_child >= size()) {
        return;
      }

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
      std::swap(_id_pos[_heap[pos].id], _id_pos[_heap[smallest_child].id]);
      pos = smallest_child;
    }
  }

  ID _capacity;
  ScalableVector<HeapElement> _heap;
  std::size_t *_id_pos;
  Comparator<Key> _comparator{};
};

template <typename Key>
using SharedBinaryMaxHeap = SharedBinaryHeap<Key, binary_heap::max_heap_comparator>;

template <typename Key>
using SharedBinaryMinHeap = SharedBinaryHeap<Key, binary_heap::min_heap_comparator>;

//! Addressable binary heap with fixed capacity.
/*
 * The next class is based on iq_queue from the RoutingKit project
 *
 * Copyright (c) 2016, RoutingKit contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer. Redistributions in binary
 * form must reproduce the above copyright notice, this list of conditions and
 * the following disclaimer in the documentation and/or other materials
 * provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
template <typename Key, template <typename> typename Comparator> class BinaryHeap {
  using ID = std::size_t;

  static constexpr ID kInvalidID = std::numeric_limits<ID>::max();
  static constexpr std::size_t kTreeArity = 4;

public:
  explicit BinaryHeap(const std::size_t capacity)
      : _heap(capacity),
        _id_pos(capacity, kInvalidID),
        _size(0) {}

  BinaryHeap(const BinaryHeap &) = delete;
  BinaryHeap &operator=(const BinaryHeap &) = delete;

  BinaryHeap(BinaryHeap &&) noexcept = default;
  BinaryHeap &operator=(BinaryHeap &&) noexcept = default;

  [[nodiscard]] bool empty() const {
    return _size == 0;
  }

  [[nodiscard]] std::size_t size() const {
    return _size;
  }

  [[nodiscard]] std::size_t capacity() const {
    return _heap.size();
  }

  [[nodiscard]] bool contains(const ID id) const {
    KASSERT(id < capacity());
    return _id_pos[id] != kInvalidID;
  }

  Key key(const ID id) const {
    KASSERT(contains(id));
    return _heap[_id_pos[id]].key;
  }

  [[nodiscard]] ID peek_id() const {
    KASSERT(!empty());
    return _heap.front().id;
  }

  Key peek_key() const {
    KASSERT(!empty());
    return _heap.front().key;
  }

  void remove(const ID id) {
    KASSERT(contains(id));
    decrease_priority(id, Comparator<Key>::kMinValue);
    pop();
    KASSERT(!contains(id));
  }

  void clear() {
    for (std::size_t i = 0; i < _size; ++i) {
      _id_pos[_heap[i].id] = kInvalidID;
    }
    _size = 0;
  }

  void pop() {
    KASSERT(!empty());
    --_size;
    std::swap(_heap.front(), _heap[_size]);
    _id_pos[_heap[0].id] = 0;
    _id_pos[_heap[_size].id] = kInvalidID;
    sift_down(0);
  }

  void push(const ID id, const Key &key) {
    KASSERT(!contains(id));
    KASSERT(size() < _heap.size());
    const std::size_t pos = _size;
    ++_size;
    _heap[pos] = {id, key};
    _id_pos[id] = pos;
    sift_up(pos);
  }

  void push_or_change_priority(const ID id, const Key &new_key) {
    if (contains(id)) {
      change_priority(id, new_key);
    } else {
      push(id, new_key);
    }
  }

  void change_priority(const ID id, const Key &new_key) {
    const Key &old_key = key(id);
    if (_comparator(new_key, old_key)) {
      increase_priority(id, new_key);
    } else if (_comparator(old_key, new_key)) {
      decrease_priority(id, new_key);
    }
  }

  // *NOTE*: "decrease" is in respect to the priority
  // e.g., decreasing an integral key in a BinaryMinHeap *increases* its
  // priority hence, *increase_priority* must be called instead of
  // decrease_priority
  void decrease_priority(const ID id, const Key &new_key) {
    KASSERT(contains(id));
    KASSERT(_comparator(key(id), new_key));
    _heap[_id_pos[id]].key = new_key;
    sift_up(_id_pos[id]);
  }

  // deprecated
  void decrease_priority_by(const ID id, const Key &delta) {
    KASSERT(contains(id));
    KASSERT(delta > 0);
    decrease_priority(id, key(id) - delta);
  }

  void increase_priority(const ID id, const Key &new_key) {
    KASSERT(contains(id));
    KASSERT(_comparator(new_key, key(id)));
    _heap[_id_pos[id]].key = new_key;
    sift_down(_id_pos[id]);
  }

  void resize(const std::size_t capacity) {
    KASSERT(empty(), "heap should be empty when resizing it");
    _id_pos.resize(capacity, kInvalidID);
    _heap.resize(capacity);
  }

  [[nodiscard]] std::size_t memory_in_kb() const {
    return _id_pos.size() * sizeof(std::size_t) / 1000 + _heap.size() * sizeof(HeapElement) / 1000;
  }

private:
  struct HeapElement {
    HeapElement() noexcept : id(kInvalidID), key(0) {}
    HeapElement(const ID id, const Key &key) noexcept : id(id), key(key) {}
    ID id;
    Key key;
  };

  void sift_up(std::size_t pos) {
    while (pos != 0) {
      const std::size_t parent = (pos - 1) / kTreeArity;
      if (_comparator(_heap[parent].key, _heap[pos].key)) {
        std::swap(_heap[pos], _heap[parent]);
        std::swap(_id_pos[_heap[pos].id], _id_pos[_heap[parent].id]);
      }
      pos = parent;
    }
  }

  void sift_down(std::size_t pos) {
    while (true) {
      const std::size_t first_child = kTreeArity * pos + 1;
      if (first_child >= _size)
        return;

      std::size_t smallest_child = first_child;
      for (std::size_t c = first_child + 1; c < std::min(kTreeArity * pos + kTreeArity + 1, _size);
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
      std::swap(_id_pos[_heap[pos].id], _id_pos[_heap[smallest_child].id]);
      pos = smallest_child;
    }
  }

  std::vector<HeapElement> _heap;
  std::vector<std::size_t> _id_pos;
  std::size_t _size;
  Comparator<Key> _comparator{};
};

template <typename Key> using BinaryMaxHeap = BinaryHeap<Key, binary_heap::max_heap_comparator>;

template <typename Key> using BinaryMinHeap = BinaryHeap<Key, binary_heap::min_heap_comparator>;

template <
    typename ID,
    typename Key,
    template <typename>
    typename Comparator,
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
  DynamicBinaryForest(DynamicBinaryForest &&) noexcept = default;
  DynamicBinaryForest &operator=(const DynamicBinaryForest &) = delete;
  DynamicBinaryForest &operator=(DynamicBinaryForest &&) noexcept = default;

  void init(const std::size_t capacity, const std::size_t heaps) {
    _id_pos.resize(capacity, kInvalidID);
    _heaps.resize(heaps);
  }

  std::size_t capacity() const {
    return _id_pos.size();
  }

  bool contains(const ID id) const {
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

template <typename ID, typename Key, template <typename...> typename Container = std::vector>
class DynamicBinaryMinMaxForest {
  struct HeapElement {
    ID complementary_pos;
    ID id;
    Key key;
  };

  template <template <typename> typename Comparator> class DynamicBinaryForest {
    static constexpr std::size_t kTreeArity = 4;
    static constexpr ID kInvalidID = std::numeric_limits<ID>::max();

  public:
    explicit DynamicBinaryForest() {}

    DynamicBinaryForest(const DynamicBinaryForest &) = delete;
    DynamicBinaryForest &operator=(const DynamicBinaryForest &) = delete;

    DynamicBinaryForest(DynamicBinaryForest &&) noexcept = default;
    DynamicBinaryForest &operator=(DynamicBinaryForest &&) noexcept = default;

    Container<HeapElement> *init(const std::size_t num_heaps) {
      _heaps.resize(num_heaps);
      return _heaps.data();
    }

    void init_complementary_data(Container<HeapElement> *complementary_heaps) {
      _complementary_heaps = complementary_heaps;
    }

    [[nodiscard]] bool empty(const std::size_t heap) const {
      return _heaps[heap].empty();
    }

    [[nodiscard]] std::size_t size(const std::size_t heap) const {
      return _heaps[heap].size();
    }

    HeapElement &push(const std::size_t heap, const ID id, const Key key) {
      KASSERT(key != Comparator<Key>::kMinValue);
      KASSERT(_comparator(key, Comparator<Key>::kMinValue));

      const std::size_t initial_pos = size(heap);
      _heaps[heap].emplace_back(0, id, key);

      const std::size_t final_pos = sift_up(heap, initial_pos);
      _heaps[heap][final_pos].complementary_pos = final_pos;

      return _heaps[heap][final_pos];
    }

    [[nodiscard]] const HeapElement &peek(const std::size_t heap) const {
      return _heaps[heap].front();
    }

    [[nodiscard]] ID peek_id(const std::size_t heap) const {
      return _heaps[heap].front().id;
    }

    [[nodiscard]] Key peek_key(const std::size_t heap) const {
      return _heaps[heap].front().key;
    }

    void pop(const std::size_t heap) {
      swap<true>(heap, size(heap) - 1, 0);
      _heaps[heap].pop_back();
      sift_down(heap, 0);
    }

    void remove(const std::size_t heap, const std::size_t pos) {
      _heaps[heap][pos].key = Comparator<Key>::kMinValue;
      sift_up(heap, pos);
      pop(heap);
    }

    void clear() {
      for (std::size_t heap = 0; heap < _heaps.size(); ++heap) {
        _heaps[heap].clear();
      }
    }

  private:
    std::size_t sift_up(const std::size_t heap, std::size_t pos) {
      while (pos != 0) {
        const std::size_t parent_pos = (pos - 1) / kTreeArity;

        if (!_comparator(_heaps[heap][parent_pos].key, _heaps[heap][pos].key)) {
          return pos;
        }

        swap<true>(heap, parent_pos, pos);
        pos = parent_pos;
      }

      return 0;
    }

    void sift_down(const std::size_t heap_id, std::size_t pos) {
      const auto &heap = _heaps[heap_id];
      const std::size_t heap_size = heap.size();

      while (true) {
        const std::size_t first_child = kTreeArity * pos + 1;
        if (first_child >= heap_size) {
          return;
        }

        const std::size_t last_child_p1 = std::min(first_child + kTreeArity, heap_size);

        std::size_t smallest_child = first_child;
        for (std::size_t child = first_child + 1; child < last_child_p1; ++child) {
          if (_comparator(heap[smallest_child].key, heap[child].key)) {
            smallest_child = child;
          }
        }

        if (_comparator(heap[smallest_child].key, heap[pos].key) ||
            heap[smallest_child].key == heap[pos].key) {
          return;
        }

        swap(heap_id, smallest_child, pos);
        pos = smallest_child;
      }
    }

    template <bool kSyncOnlyFirst = false>
    void swap(const std::size_t heap, const std::size_t a, const std::size_t b) {
      if constexpr (kSyncOnlyFirst) {
        _complementary_heaps[heap][_heaps[heap][a].complementary_pos].complementary_pos = b;
      } else {
        std::swap(
            _complementary_heaps[heap][_heaps[heap][a].complementary_pos].complementary_pos,
            _complementary_heaps[heap][_heaps[heap][b].complementary_pos].complementary_pos
        );
      }

      std::swap(_heaps[heap][a], _heaps[heap][b]);
    }

    Container<Container<HeapElement>> _heaps;
    Container<HeapElement> *_complementary_heaps;
    Comparator<Key> _comparator{};
  };

  using DynamicBinaryMaxForest = DynamicBinaryForest<binary_heap::max_heap_comparator>;
  using DynamicBinaryMinForest = DynamicBinaryForest<binary_heap::min_heap_comparator>;

public:
  DynamicBinaryMinMaxForest() {}

  DynamicBinaryMinMaxForest(const DynamicBinaryMinMaxForest &) = delete;
  DynamicBinaryMinMaxForest &operator=(const DynamicBinaryMinMaxForest &) = delete;

  DynamicBinaryMinMaxForest(DynamicBinaryMinMaxForest &&) noexcept = default;
  DynamicBinaryMinMaxForest &operator=(DynamicBinaryMinMaxForest &&) noexcept = default;

  void init(const std::size_t num_heaps) {
    _num_heaps = num_heaps;
    _size = 0;

    auto *max_forest_data = _max_forest.init(num_heaps);
    auto *min_forest_data = _min_forest.init(num_heaps);

    _max_forest.init_complementary_data(min_forest_data);
    _min_forest.init_complementary_data(max_forest_data);
  }

  [[nodiscard]] std::size_t size() const {
    return _size;
  }

  [[nodiscard]] std::size_t size(const std::size_t heap) const {
    KASSERT(heap < _num_heaps);
    KASSERT(_max_forest.size(heap) == _min_forest.size(heap));

    return _max_forest.size(heap);
  }

  [[nodiscard]] bool empty(const std::size_t heap) const {
    KASSERT(heap < _num_heaps);
    KASSERT(_max_forest.empty(heap) == _min_forest.empty(heap));

    return _max_forest.empty(heap);
  }

  void push(const std::size_t heap, const ID id, const Key key) {
    KASSERT(heap < _num_heaps);

    auto &element_min = _min_forest.push(heap, id, key);
    auto &element_max = _max_forest.push(heap, id, key);
    std::swap(element_min.complementary_pos, element_max.complementary_pos);

    _size += 1;
  }

  [[nodiscard]] ID peek_min_id(const std::size_t heap) const {
    KASSERT(heap < _num_heaps);
    KASSERT(!_min_forest.empty(heap));

    return _min_forest.peek_id(heap);
  }

  [[nodiscard]] ID peek_max_id(const std::size_t heap) const {
    KASSERT(heap < _num_heaps);
    KASSERT(!_max_forest.empty(heap));

    return _max_forest.peek_id(heap);
  }

  [[nodiscard]] Key peek_min_key(const std::size_t heap) const {
    KASSERT(heap < _num_heaps);
    KASSERT(!_min_forest.empty(heap));

    return _min_forest.peek_key(heap);
  }

  [[nodiscard]] Key peek_max_key(const std::size_t heap) const {
    KASSERT(heap < _num_heaps);
    KASSERT(!_max_forest.empty(heap));

    return _max_forest.peek_key(heap);
  }

  void pop_min(const std::size_t heap) {
    KASSERT(heap < _num_heaps);
    KASSERT(!_min_forest.empty(heap));
    KASSERT(!_max_forest.empty(heap));

    _max_forest.remove(heap, _min_forest.peek(heap).complementary_pos);
    _min_forest.pop(heap);
    _size -= 1;
  }

  void pop_max(const std::size_t heap) {
    KASSERT(!_min_forest.empty(heap));
    KASSERT(!_max_forest.empty(heap));

    _min_forest.remove(heap, _max_forest.peek(heap).complementary_pos);
    _max_forest.pop(heap);
    _size -= 1;
  }

  void clear() {
    _min_forest.clear();
    _max_forest.clear();
    _size = 0;
  }

private:
  std::size_t _num_heaps;
  std::size_t _size;

  DynamicBinaryMinForest _min_forest;
  DynamicBinaryMaxForest _max_forest;
};

//! Dynamic binary heap, not addressable
template <
    typename ID,
    typename Key,
    template <typename>
    typename Comparator,
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
