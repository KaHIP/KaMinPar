/*******************************************************************************
 * @file:   queue.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Queue with fixed capacity.
 ******************************************************************************/
#pragma once

#include <vector>

#include "kaminpar-common/assert.h"

namespace kaminpar {
/*!
 * Queue with fixed capacity. Add new elements to its tail and remove elements
 * from its head. Its capacity limits the number of push_head() calls without
 * invoking clear(), *not* just the number of elements it can hold
 * simultaneously.
 *
 * @tparam T Type of element.
 */
template <typename T> class Queue {
public:
  using value_type = T;
  using size_type = std::size_t;
  using reference = T &;
  using const_reference = const T &;
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;

  explicit Queue(const std::size_t capacity) : _data(capacity) {}

  Queue(const Queue &) = delete;
  Queue &operator=(const Queue &) = delete;

  Queue(Queue &&) noexcept = default;
  Queue &operator=(Queue &&) noexcept = default;

  // Access operators
  const_reference tail() const {
    KASSERT(!empty());
    return _data[_tail];
  }

  const_reference head() const {
    KASSERT(!empty());
    return _data[_head];
  }

  const_reference operator[](const size_type pos) const {
    KASSERT(_head + pos < _data.size());
    return _data[_head + pos];
  }

  reference operator[](const size_type pos) {
    KASSERT(_head + pos < _data.size());
    return _data[_head + pos];
  }

  // Iterators
  iterator begin() {
    return _data.begin() + _head;
  }

  const_iterator cbegin() const {
    return _data.cbegin() + _head;
  }

  iterator end() {
    return _data.begin() + _tail;
  }

  const_iterator cend() const {
    return _data.cbegin() + _tail;
  }

  // Capacity
  [[nodiscard]] bool empty() const {
    return _head == _tail;
  }

  [[nodiscard]] size_type size() const {
    return _tail - _head;
  }

  std::size_t capacity() {
    return _data.size();
  }

  void resize(const std::size_t capacity) {
    _data.resize(capacity);
    clear();
  }

  // Modification
  void pop_head() {
    KASSERT(_head < _tail);
    ++_head;
  }

  void push_head(const_reference element) {
    KASSERT(0 < _head);
    _data[--_head] = element;
  }

  void push_tail(const_reference element) {
    KASSERT(_tail < _data.size());
    _data[_tail++] = element;
  }

  void pop_tail() {
    KASSERT(_head < _tail);
    --_tail;
  }

  void clear() {
    _head = _tail = 0;
  }

  [[nodiscard]] std::size_t memory_in_kb() const {
    return _data.size() * sizeof(T) / 1000;
  }

private:
  std::vector<T> _data;
  std::size_t _head = 0;
  std::size_t _tail = 0;
};
} // namespace kaminpar
