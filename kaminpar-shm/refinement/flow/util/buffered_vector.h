#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <span>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

template <typename T> class BufferedVector {
  static_assert(std::copyable<T>, "BufferedVector requires T to be copyable");

  static constexpr std::size_t kBufferSize = 1024;

  struct BufferData {
    std::size_t num_buffered_elements = 0;
    std::array<T, kBufferSize> buffer;

    [[nodiscard]] std::span<const T> view() const {
      return std::span<const T>(buffer.data(), num_buffered_elements);
    }
  };

public:
  class Buffer {
    friend BufferedVector<T>;

  public:
    void push_back(T t) {
      _data.buffer[_data.num_buffered_elements] = t;
      if (++_data.num_buffered_elements == kBufferSize) [[unlikely]] {
        flush();
      }
    }

    [[nodiscard]] bool empty() const {
      return _data.num_buffered_elements == 0;
    }

  private:
    Buffer(BufferedVector<T> &buffered_vector, BufferData &data)
        : _buffered_vector(buffered_vector),
          _data(data) {}

    void flush() {
      _buffered_vector.atomic_push_back(_data.view());
      _data.num_buffered_elements = 0;
    }

  private:
    BufferedVector<T> &_buffered_vector;
    BufferData &_data;
  };

public:
  BufferedVector() : _size(0) {}

  BufferedVector(const BufferedVector &) = delete;
  BufferedVector &operator=(const BufferedVector &) = delete;

  BufferedVector(BufferedVector &&) noexcept = default;
  BufferedVector &operator=(BufferedVector &&) noexcept = default;

  void clear() {
    _size = 0;
  }

  void resize(const std::size_t size) {
    _size = size;
  }

  void reserve(const std::size_t size) {
    if (_data.size() < size) {
      _data.resize(size, static_array::noinit);
    }
  }

  T pop_back() {
    KASSERT(_size > 0);

    return _data[--_size];
  }

  void push_back(T t) {
    KASSERT(_size < _data.size());

    _data[_size] = t;
    _size += 1;
  }

  void push_back(std::span<const T> ts) {
    KASSERT(_size + ts.size() <= _data.size());

    std::copy(ts.begin(), ts.end(), _data.begin() + _size);
    _size += ts.size();
  }

  void atomic_push_back(T t) {
    KASSERT(_size < _data.size());

    const std::size_t pos = __atomic_fetch_add(&_size, 1, __ATOMIC_RELAXED);
    _data[pos] = t;
  }

  void atomic_push_back(std::span<const T> ts) {
    KASSERT(_size + ts.size() <= _data.size());

    const std::size_t pos = __atomic_fetch_add(&_size, ts.size(), __ATOMIC_RELAXED);
    std::copy(ts.begin(), ts.end(), _data.begin() + pos);
  }

  void flush() {
    for (BufferData &data : _buffers) {
      if (data.num_buffered_elements > 0) {
        push_back(data.view());
        data.num_buffered_elements = 0;
      }
    }
  }

  [[nodiscard]] Buffer local_buffer() {
    return Buffer(*this, _buffers.local());
  }

  [[nodiscard]] bool empty() const {
    return _size == 0;
  }

  [[nodiscard]] std::size_t size() const {
    return _size;
  }

  [[nodiscard]] const T &operator[](const std::size_t pos) const {
    return _data[pos];
  }

  [[nodiscard]] T &operator[](const std::size_t pos) {
    return _data[pos];
  }

private:
  std::size_t _size;
  StaticArray<T> _data;

  tbb::enumerable_thread_specific<BufferData> _buffers;
};

} // namespace kaminpar::shm
