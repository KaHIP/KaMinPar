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

  SET_DEBUG(false);
  constexpr static MutType kKeyToDebug = 15;

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
    DBGC(key == kKeyToDebug) << "decrease_by(" << key << ", " << value << ")";

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
      if ((cur_key_hash <= hole_pos && cur_pos > hole_pos) || cur_pos < cur_key_hash) {
        DBGC(key == kKeyToDebug) << "move hole to pos = " << cur_pos << ", entry = " << cur_entry
                                 << " = " << decode_key(cur_entry) << ":" << decode_value(cur_entry)
                                 << " --> entry moved to " << hole_pos;
        write_pos(hole_pos, cur_entry);
        hole_pos = cur_pos;
      } else {
        DBGC(key == kKeyToDebug) << "skipping pos = " << cur_pos << ", entry = " << cur_entry
                                 << " = " << decode_key(cur_entry) << ":"
                                 << decode_value(cur_entry);
        // skip element
      }
    } while (true);

    write_pos(hole_pos, 0);

    KASSERT(
        [&] {
          for (std::size_t pos = 0; pos < _value_mask + 1; ++pos) {
            if (read_pos(pos) == 0) {
              continue;
            }

            const MutType key = decode_key(read_pos(pos));
            for (std::size_t pos2 = hash(key); pos2 < _value_mask + 1 + pos; ++pos2) {
              if (read_pos(hash(pos2)) == 0) {
                LOG_WARNING << "key = " << key << " at position = " << pos
                            << ": hash(key) = " << hash(key)
                            << ", but empty slot at position = " << hash(pos2);
                return false;
              } else if (decode_key(read_pos(hash(pos2))) == key) {
                break;
              }
            }
          }
          return true;
        }(),
        "bad hash table state after erasing key = " << key,
        assert::heavy
    );
  }

  void increase_by(const MutType key, const MutType value) {
    DBGC(key == kKeyToDebug) << "increase_by(key = " << key << ", value = " << value << ")";

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
      DBGC(key == kKeyToDebug) << "do-while: {pos = " << pos << ", entry = " << entry << " = "
                               << decode_key(entry) << ":" << decode_value(entry) << "}";
    } while (entry != 0 && decode_key(entry) != key);

    DBGC(key == kKeyToDebug) << "find(key = " << key << ") --> {pos = " << pos
                             << ", entry = " << entry << " = " << decode_key(entry) << ":"
                             << decode_value(entry) << "}";

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
