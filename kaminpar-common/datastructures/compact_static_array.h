/*******************************************************************************
 * A static array which stores integers with only as many bytes as the largest
 * integer requires.
 *
 * @file:   compact_static_array.h
 * @author: Daniel Salwasser
 * @date:   12.01.2024
 ******************************************************************************/
#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/math.h"

namespace kaminpar {

/*!
 * A static array which stores integers with only as many bytes as the largest integer requires.
 *
 * @tparam Int The type of integer to store.
 */
template <std::unsigned_integral Int> class CompactStaticArray {
  static_assert(std::numeric_limits<Int>::is_integer);

  class CompactStaticArrayIterator {
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = Int;
    using reference = Int &;
    using pointer = Int *;
    using difference_type = std::ptrdiff_t;

    CompactStaticArrayIterator(
        const std::size_t byte_width, const Int read_mask, const std::uint8_t *data
    )
        : _byte_width(byte_width),
          _mask(read_mask),
          _data(data) {}

    CompactStaticArrayIterator(const CompactStaticArrayIterator &) = default;
    CompactStaticArrayIterator &operator=(const CompactStaticArrayIterator &) = default;

    Int operator*() const {
      return *reinterpret_cast<const Int *>(_data) & _mask;
    }

    pointer operator->() const {
      return reinterpret_cast<const Int *>(_data);
    }

    reference operator[](const difference_type n) const {
      return reinterpret_cast<const Int *>(_data + _byte_width * n);
    }

    CompactStaticArrayIterator &operator++() {
      return _data += _byte_width, *this;
    }

    CompactStaticArrayIterator &operator--() {
      return _data -= _byte_width, *this;
    }

    CompactStaticArrayIterator operator++(int) const {
      return CompactStaticArrayIterator{_byte_width, _mask, _data + _byte_width};
    }

    CompactStaticArrayIterator operator--(int) const {
      return CompactStaticArrayIterator{_byte_width, _mask, _data - _byte_width};
    }

    CompactStaticArrayIterator operator+(const difference_type n) const {
      return CompactStaticArrayIterator{_byte_width, _mask, _data + _byte_width * n};
    }

    CompactStaticArrayIterator operator-(const difference_type n) const {
      return CompactStaticArrayIterator{_byte_width, _mask, _data - _byte_width * n};
    }

    CompactStaticArrayIterator &operator+=(const difference_type n) {
      return _data += _byte_width * n, *this;
    }

    CompactStaticArrayIterator &operator-=(const difference_type n) {
      return _data -= _byte_width * n, *this;
    }

    difference_type operator+(const CompactStaticArrayIterator &other) const {
      return (reinterpret_cast<difference_type>(_data) / _byte_width) +
             (reinterpret_cast<difference_type>(other._data) / _byte_width);
    }

    difference_type operator-(const CompactStaticArrayIterator &other) const {
      return (reinterpret_cast<difference_type>(_data) / _byte_width) -
             (reinterpret_cast<difference_type>(other._data) / _byte_width);
    }

    bool operator==(const CompactStaticArrayIterator &other) const {
      return _data == other._data;
    }

    bool operator!=(const CompactStaticArrayIterator &other) const {
      return _data != other._data;
    }

    bool operator>(const CompactStaticArrayIterator &other) const {
      return _data > other._data;
    }

    bool operator<(const CompactStaticArrayIterator &other) const {
      return _data < other._data;
    }

    bool operator>=(const CompactStaticArrayIterator &other) const {
      return _data >= other._data;
    }

    bool operator<=(const CompactStaticArrayIterator &other) const {
      return _data <= other._data;
    }

  private:
    const std::size_t _byte_width;
    const Int _mask;
    const std::uint8_t *_data;
  };

public:
  using value_type = Int;
  using size_type = std::size_t;
  using reference = value_type &;
  using const_reference = const value_type &;
  using iterator = CompactStaticArrayIterator;
  using const_iterator = const CompactStaticArrayIterator;

  /*!
   * Constructs an unitialized CompactStaticArray.
   */
  CompactStaticArray()
      : _byte_width(0),
        _size(0),
        _num_values(0),
        _unrestricted_size(0),
        _unrestricted_num_values(0) {
    RECORD_DATA_STRUCT(0, _struct);
  }

  /*!
   * Constructs an unitialized CompactStaticArray.
   *
   * @param byte_width The number of bytes needed to store the largest integer in the array.
   * @param size num_values number of values to store.
   */
  CompactStaticArray(const std::size_t byte_width, const std::size_t num_values) {
    RECORD_DATA_STRUCT(0, _struct);
    resize(byte_width, num_values);
  }

  /*!
   * Constructs an unitialized CompactStaticArray.
   *
   * @param byte_width The number of bytes needed to store the largest integer in the array.
   * @param actual_size The number of bytes that the compact representation in memory uses.
   * @param data The pointer to the memory location where the data is compactly stored.
   */
  CompactStaticArray(
      const std::size_t byte_width,
      const std::size_t actual_size,
      std::unique_ptr<std::uint8_t[]> data
  )
      : _byte_width(byte_width),
        _size(actual_size),
        _num_values((_size - (sizeof(Int) - _byte_width)) / _byte_width),
        _values(std::move(data)),
        _read_mask(std::numeric_limits<Int>::max() >> ((sizeof(Int) - byte_width) * 8)),
        _write_mask(std::numeric_limits<Int>::max() << (byte_width * 8)),
        _unrestricted_size(_size),
        _unrestricted_num_values(_num_values) {
    RECORD_DATA_STRUCT(0, _struct);
    KASSERT(actual_size >= sizeof(Int) - _byte_width);
    KASSERT(byte_width >= 1u);
    KASSERT(byte_width <= 8u);
  }

  CompactStaticArray(const CompactStaticArray &) = delete;
  CompactStaticArray &operator=(const CompactStaticArray &) = delete;

  CompactStaticArray(CompactStaticArray &&) noexcept = default;
  CompactStaticArray &operator=(CompactStaticArray &&) noexcept = default;

  /*!
   * Resizes the array.
   *
   * @param byte_width The number of bytes needed to store the largest integer in the array.
   * @param num_values The number of values to store.
   */
  void resize(const std::size_t byte_width, const std::size_t num_values) {
    KASSERT(byte_width >= 1u);
    KASSERT(byte_width <= 8u);

    _byte_width = byte_width;
    _size = num_values * byte_width + sizeof(Int) - byte_width;

    _num_values = num_values;
    _values = std::make_unique<std::uint8_t[]>(_size);

    _read_mask = std::numeric_limits<Int>::max() >> ((sizeof(Int) - byte_width) * 8);
    _write_mask = std::numeric_limits<Int>::max() << (byte_width * 8);

    _unrestricted_size = _size;
    _unrestricted_num_values = num_values;

    IF_HEAP_PROFILING(_struct->size = std::max(_struct->size, _size));
  }

  /*!
   * Restricts the array to a specific size. This operation can be undone by calling unrestrict().
   *
   * @param new_size The number of values to be visible.
   */
  void restrict(const std::size_t new_num_values) {
    KASSERT(new_num_values <= _num_values);

    _unrestricted_size = _size;
    _size = new_num_values * _byte_width + sizeof(Int) - _byte_width;

    _unrestricted_num_values = _num_values;
    _num_values = new_num_values;
  }

  /*!
   * Undos the previous restriction. It does nothing when the restrict method has previously not
   * been invoked.
   */
  void unrestrict() {
    _size = _unrestricted_size;
    _num_values = _unrestricted_num_values;
  }

  /*!
   * Stores an integer.
   *
   * @param pos The position in the array at which the integer is to be stored.
   * @param value The value to store.
   */
  void write(const std::size_t pos, Int value) {
    KASSERT(pos < _num_values);
    KASSERT(math::byte_width(value) <= _byte_width);

    std::uint8_t *data = _values.get() + pos * _byte_width;
    for (std::size_t i = 0; i < _byte_width; ++i) {
      *data++ = value & 0b11111111;
      value >>= 8;
    }
  }

  /*!
   * Accesses an integer.
   *
   * @param pos The position of the integer in the array to be returned.
   * @return The integer stored at the given position in the array.
   */
  [[nodiscard]] Int operator[](const std::size_t pos) const {
    KASSERT(pos < _num_values);
    return *reinterpret_cast<const Int *>(_values.get() + pos * _byte_width) & _read_mask;
  }

  /*!
   * Returns an interator to the beginning.
   *
   * @return An interator to the beginning.
   */
  [[nodiscard]] CompactStaticArrayIterator begin() const {
    return CompactStaticArrayIterator{_byte_width, _read_mask, _values.get()};
  }

  /*!
   * Returns an interator to the end.
   *
   * @return An interator to the end.
   */
  [[nodiscard]] CompactStaticArrayIterator end() const {
    const std::uint8_t *data = _values.get() + _size - (sizeof(Int) - _byte_width);
    return CompactStaticArrayIterator{_byte_width, _read_mask, data};
  }

  /*!
   * Returns whether the array is empty.
   *
   * @return Whether the array is empty.
   */
  [[nodiscard]] bool empty() const {
    return _num_values == 0;
  }

  /*!
   * Returns the number of integers in the array.
   *
   * @return The number of integers in the array.
   */
  [[nodiscard]] std::size_t size() const {
    return _num_values;
  }

  /*!
   * Returns the number of bytes needed to store the largest integer in the array.
   *
   * @return The number of bytes needed to store the largest integer in the array.
   */
  [[nodiscard]] std::uint8_t byte_width() const {
    return _byte_width;
  }

  /*!
   * Returns the memory space of this array in bytes.
   *
   * @return The memory space of this array in bytes.
   */
  [[nodiscard]] std::size_t memory_space() const {
    return _unrestricted_size;
  }

  /*!
   * Returns a pointer to the memory location where the data is compactly stored.
   *
   * @returns A pointer to the memory location where the data is compactly stored.
   */
  [[nodiscard]] const std::uint8_t *data() const {
    return _values.get();
  }

private:
  std::size_t _byte_width;
  std::size_t _size;

  std::size_t _num_values;
  std::unique_ptr<std::uint8_t[]> _values;

  Int _read_mask;
  Int _write_mask;

  std::size_t _unrestricted_size;
  std::size_t _unrestricted_num_values;

  IF_HEAP_PROFILING(heap_profiler::DataStructure *_struct);
};

} // namespace kaminpar
