/*******************************************************************************
 * Hash map with linear probing that stores keys and values in the same 32/64
 * bit array entry.
 *
 * Allows concurrent reads but requires locking for writes.
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

#include "kaminpar-common/logger.h"
#include "kaminpar-common/math.h"

namespace kaminpar {
template <typename Type> class CompactHashMap {
  static_assert(std::is_unsigned_v<Type>);

  SET_DEBUG(true);

public:
  [[nodiscard]] static int compute_key_bits(const Type max_key) {
    return math::ceil_log2(max_key);
  }

  CompactHashMap(Type *data, const std::size_t size, const int key_bits)
      : _data(data),
        _value_mask(size - 1),
        _key_bits(key_bits) {
    KASSERT(math::is_power_of_2(size));
  }

  void decrease_by(const Type key, const Type value) {
    // Assertion: hash-table is locked for other modiyfing operations
    const auto [start_pos, start_entry] = find(key);
    const Type start_value = decode_value(start_entry);

    KASSERT(decode_key(start_entry) == key);
    KASSERT(start_value > 0);

    // Simple case: just decrease, but do not erase
    if (start_value > value) {
      write_pos(start_pos, start_entry - value);
      return;
    }

    // Erase element
    KASSERT(start_value == value);

    std::size_t hole_pos = hash(key);
    std::size_t cur_pos = hash(key);

    Type cur_entry;

    do {
      cur_pos = hash(cur_pos + 1);
      cur_entry = read_pos(cur_pos);

      if (cur_entry == 0) {
        break;
      }

      const std::size_t cur_key = decode_key(cur_entry);
      const std::size_t cur_key_hash = hash(cur_key);
      if (cur_key_hash <= hole_pos || cur_pos < cur_key_hash) {
        write_pos(hole_pos, cur_entry);
        hole_pos = cur_key_hash;
      } else {
        // skip element
      }
    } while (true);

    write_pos(hole_pos, 0);
  }

  void increase_by(const Type key, const Type value) {
    // Assertion: hash-table is locked for other modifying operations
    const auto [pos, entry] = find(key);
    write_pos(pos, encode_key_value(key, decode_value(entry) + value));
  }

  [[nodiscard]] Type get(const Type key) const {
    const auto [pos, entry] = find(key);
    return decode_value(entry);
  }

private:
  [[nodiscard]] std::pair<std::size_t, Type> find(const Type key) const {
    std::size_t pos = key - 1;
    Type entry;

    do {
      pos = hash(pos + 1);
      entry = read_pos(pos);
    } while (entry != 0 && decode_key(entry) != key);

    return {pos, entry};
  }

  [[nodiscard]] Type hash(const Type key) const {
    return key & _value_mask;
  }

  [[nodiscard]] Type read_pos(const std::size_t pos) const {
    return __atomic_load_n(&_data[pos], __ATOMIC_RELAXED);
  }

  void write_pos(const std::size_t pos, const Type value) const {
    __atomic_store_n(&_data[pos], value, __ATOMIC_RELAXED);
  }

  [[nodiscard]] Type decode_key(const Type entry) const {
    return entry >> value_bits();
  }

  [[nodiscard]] Type decode_value(const Type entry) const {
    return (entry << key_bits()) >> key_bits();
  }

  Type encode_key_value(const Type key, const Type value) {
    KASSERT(key == 0 || math::ceil_log2(key) <= _key_bits, "key too large");
    return (key << value_bits()) | value;
  }

  [[nodiscard]] int key_bits() const {
    return _key_bits;
  }

  [[nodiscard]] int value_bits() const {
    return std::numeric_limits<Type>::digits - _key_bits;
  }

  Type *_data;
  Type _value_mask;
  int _key_bits;
};
} // namespace kaminpar
