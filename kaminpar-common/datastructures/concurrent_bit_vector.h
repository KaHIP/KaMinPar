/*******************************************************************************
 * A concurrent bit vector which stores bits compactly and uses atomic
 * read/write operations.
 *
 * @file:   concurrent_bit_vector.h
 * @author: Daniel Salwasser
 * @date:   25.01.2024
 ******************************************************************************/
#pragma once

#include <memory>

#include <kassert/kassert.hpp>

#include "kaminpar-common/math.h"

namespace kaminpar {

/*!
 * A concurrent bit vector which stores bits compactly and uses atomic read/write operations.
 *
 * @tparam Size The type of index to use to access bits.
 */
template <typename Size> class ConcurrentBitVector {
public:
  /*!
   * Constructs a new empty ConcurrentBitVector.
   */
  ConcurrentBitVector() : _size(0), _byte_capacity(0) {}

  /*!
   * Constructs a new ConcurrentBitVector
   *
   * @param size The number of bits to store.
   */
  ConcurrentBitVector(const Size size)
      : _size(size),
        _byte_capacity(math::div_ceil(size, 8)),
        _data(std::make_unique<std::uint8_t[]>(_byte_capacity)) {}

  ConcurrentBitVector(const ConcurrentBitVector &) = delete;
  ConcurrentBitVector &operator=(const ConcurrentBitVector &) = delete;

  ConcurrentBitVector(ConcurrentBitVector &&) noexcept = default;
  ConcurrentBitVector &operator=(ConcurrentBitVector &&) noexcept = default;

  /*!
   * Atomically loads a bit.
   *
   * @param pos The position of the bit to load.
   * @return Whether the bit is set.
   */
  [[nodiscard]] bool load(const Size pos) const noexcept {
    KASSERT(pos < _size);

    std::uint8_t *ptr = _data.get() + (pos / 8);
    const std::uint8_t mask = 1 << (pos % 8);
    return (__atomic_load_n(ptr, __ATOMIC_RELAXED) & mask) != 0;
  }

  /*!
   * Atomically sets a bit.
   *
   * @param pos The position of the bit to set.
   */
  void set(const Size pos) noexcept {
    KASSERT(pos < _size);

    std::uint8_t *ptr = _data.get() + (pos / 8);
    const std::uint8_t mask = 1 << (pos % 8);
    __atomic_fetch_or(ptr, mask, __ATOMIC_RELAXED);
  }

  /*!
   * Atomically unsets a bit.
   *
   * @param pos The position of the bit to unset.
   */
  void unset(const Size pos) noexcept {
    KASSERT(pos < _size);

    std::uint8_t *ptr = _data.get() + (pos / 8);
    const std::uint8_t mask = ~(1 << (pos % 8));
    __atomic_fetch_and(ptr, mask, __ATOMIC_RELAXED);
  }

  /*!
   * Sets (non-atomically) all bits in the vector.
   */
  void set_all() noexcept {
    std::fill(_data.get(), _data.get() + _byte_capacity, 0b11111111);
  }

  /*!
   * Resizes the vector.
   *
   * @param size The number of bits to store.
   */
  void resize(const Size size) {
    KASSERT(size > 0);

    _size = size;
    _byte_capacity = math::div_ceil(size, 8);
    _data = std::make_unique<std::uint8_t[]>(_byte_capacity);
  }

  /*!
   * Frees the memory used by this data structure.
   */
  void free() {
    _size = 0;
    _byte_capacity = 0;
    _data.reset();
  }

  /*!
   * Returns the amount of bits that this vector stores.
   *
   * @return The amount of bits that this vector stores.
   */
  [[nodiscard]] Size size() const noexcept {
    return _size;
  }

  /*!
   * Returns the amount of bits that this vector can store, i.e. the size including internal
   * fragmentation.
   *
   * @return The amount of bits that this vector can store.
   */
  [[nodiscard]] Size capacity() const noexcept {
    return _byte_capacity * 8;
  }

private:
  Size _size;
  Size _byte_capacity;
  std::unique_ptr<std::uint8_t[]> _data;
};

} // namespace kaminpar
