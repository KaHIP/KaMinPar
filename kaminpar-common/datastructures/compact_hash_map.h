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
  using MutType = std::remove_const_t<Type>;
  static_assert(std::is_unsigned_v<Type>);

  SET_DEBUG(true);

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

  void decrease_by(const MutType key, const MutType value) {
    // Assertion: hash-table is locked for other modiyfing operations
    const auto [start_pos, start_entry] = find(key);
    const MutType start_value = decode_value(start_entry);

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

    MutType cur_entry;

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

  void increase_by(const MutType key, const MutType value) {
    // Assertion: hash-table is locked for other modifying operations
    const auto [pos, entry] = find(key);
    write_pos(pos, encode_key_value(key, decode_value(entry) + value));
  }

  [[nodiscard]] MutType get(const MutType key) const {
    const auto [pos, entry] = find(key);
    return decode_value(entry);
  }

private:
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
