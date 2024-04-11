/*******************************************************************************
 * Hash map with linear probing that stores keys and values in the same 32/64
 * bit array entry.
 *
 * The hash map can only store positive, non-zero values.
 * The key of a value that is decreased to 0 is automatically erased from the
 * hash table.
 * New keys are automatically inserted when increasing their value, i.e., there
 * is no explicit insert operation.
 * Exceeding the hash table's capacity causes undefined behaviour.
 *
 * Reads are always allowed, but locking is required before calling any of the
 * modifying operations.
 *
 * @file:   compact_hash_map.h
 * @author: Daniel Seemaier
 * @date:   21.02.2024
 ******************************************************************************/
#pragma once

#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

#include "kaminpar-common/math.h"

namespace kaminpar {
template <typename Type> class CompactHashMap {
  using MutType = std::remove_const_t<Type>;
  static_assert(std::is_unsigned_v<Type>);

public:
  [[nodiscard]] static int compute_key_bits(const MutType max_key) {
    return math::ceil_log2(max_key);
  }

  CompactHashMap(Type *data, const std::size_t size, const int key_bits)
      : _data(data),
        _value_mask(size - 1),
        _key_bits(key_bits) {
    KASSERT(math::is_power_of_2(size));
  }

  // May not be called concurrently
  bool decrease_by(const MutType key, const MutType value) {
    const auto [start_pos, start_entry] = find(key);
    const MutType start_value = decode_value(start_entry);

    if (start_value > value) {
      // Simple case: just decrease, but do not erase
      write_pos(start_pos, start_entry - value);
      return false;
    }

    // Harder case: erase element
    std::size_t hole_pos = start_pos;
    std::size_t cur_pos = start_pos;
    MutType cur_entry;

    do {
      cur_pos = hash(cur_pos + 1);
      cur_entry = read_pos(cur_pos);

      if (cur_entry == 0 || movable(decode_key(cur_entry), cur_pos, hole_pos)) {
        write_pos(hole_pos, cur_entry);
        hole_pos = cur_pos;
      }
    } while (cur_entry != 0);
    return true;
  }

  // May not be called concurrently
  bool increase_by(const MutType key, const MutType value) {
    const auto [pos, entry] = find(key);
    write_pos(pos, encode_key_value(key, decode_value(entry) + value));
    return entry == 0;
  }

  [[nodiscard]] MutType get(const MutType key) const {
    return decode_value(find(key).second);
  }

  [[nodiscard]] std::size_t capacity() const {
    return _value_mask + 1;
  }

  [[nodiscard]] std::size_t count() const {
    std::size_t num_nz = 0;
    for (std::size_t i = 0; i < capacity(); ++i) {
      if (read_pos(i) != 0) {
        ++num_nz;
      }
    }
    return num_nz;
  }

  template <typename Lambda> void for_each(Lambda &&lambda) const {
    for (std::size_t i = 0; i < capacity(); ++i) {
      const auto entry = read_pos(i);
      if (entry != 0) {
        lambda(decode_key(entry), decode_value(entry));
      }
    }
  }

private:
  // Decide whether we are allowed to move an element with the given hash from some position to
  // another
  [[nodiscard]] bool
  movable(const MutType key, const std::size_t from, const std::size_t to) const {
    const MutType h_key = hash(key);

    // Case 1:
    //       |from  |hash  |to
    // ------|------|------|------
    //
    // Case 2:
    //       |hash  |to    |from
    // ------|------|------|------
    //
    // Case 3:
    //       |to    |from  |hash
    // ------|------|------|------
    return (from < h_key) + (h_key <= to) + (from > to) >= 2;
  }

  // Find the right cell or point to the first empty cell if the key is not yet contained in the
  // hash table
  [[nodiscard]] std::pair<std::size_t, MutType> find(const MutType key) const {
    std::size_t pos = key - 1;
    MutType entry;

    do {
      pos = hash(pos + 1);
      entry = read_pos(pos);
    } while (entry != 0 && decode_key(entry) != key);

    return {pos, entry};
  }

  [[nodiscard]] MutType hash(const MutType key) const {
    return key & _value_mask;
  }

  [[nodiscard]] MutType read_pos(const std::size_t pos) const {
    return __atomic_load_n(&_data[pos], __ATOMIC_RELAXED);
  }

  void write_pos(const std::size_t pos, const MutType value) const {
    __atomic_store_n(&_data[pos], value, __ATOMIC_RELAXED);
  }

  [[nodiscard]] MutType decode_key(const MutType entry) const {
    return entry >> value_bits();
  }

  [[nodiscard]] MutType decode_value(const MutType entry) const {
    return (entry << key_bits()) >> key_bits();
  }

  MutType encode_key_value(const MutType key, const MutType value) {
    KASSERT(key == 0 || math::ceil_log2(key) <= _key_bits, "key too large");
    return (key << value_bits()) | value;
  }

  [[nodiscard]] int key_bits() const {
    return _key_bits;
  }

  [[nodiscard]] int value_bits() const {
    return std::numeric_limits<MutType>::digits - _key_bits;
  }

  Type *_data;
  MutType _value_mask;
  int _key_bits;
};
} // namespace kaminpar
