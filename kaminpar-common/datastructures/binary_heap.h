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
  constexpr static auto kMaxValue = std::numeric_limits<Key>::lowest();

  bool operator()(Key a, Key b) {
    return b > a;
  }
};

template <typename Key> struct min_heap_comparator {
  constexpr static auto kMinValue = std::numeric_limits<Key>::lowest();
  constexpr static auto kMaxValue = std::numeric_limits<Key>::max();

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
  explicit BinaryHeap(const std::size_t capacity = 0)
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

} // namespace kaminpar
