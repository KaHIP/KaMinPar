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
  constexpr static MutType kKeyToDebug = 14;
  constexpr static std::uint64_t kFlagToDebug = 0;

public:
  [[nodiscard]] static int compute_key_bits(const MutType max_key) {
    return math::ceil_log2(max_key);
  }

  CompactHashMap(
      Type *data, const std::size_t size, const int key_bits, const std::uint64_t flag = 0
  )
      : _data(data),
        _value_mask(size - 1),
        _key_bits(key_bits),
        _flag(flag) {
    KASSERT(math::is_power_of_2(size));
  }

  void decrease_by(const MutType key, const MutType value) {
    // Assertion: hash-table is locked for other modiyfing operations
    const auto [start_pos, start_entry] = find(key);
    const MutType start_value = decode_value(start_entry);

    KASSERT(
        decode_key(start_entry) == key,
        "decrease_by(" << key << ", " << value
                       << "): key not contained in table: " << dbg_stringify()
    );
    KASSERT(
        start_value > 0,
        "decrease_by(" << key << ", " << value
                       << "): key contained in table, but value is 0: " << dbg_stringify()
    );

    // Simple case: just decrease, but do not erase
    if (start_value > value) {
      DBGC(_flag == kFlagToDebug) << "decrease_by(" << key << ", " << value << "): " << start_entry
                                  << " = " << decode_key(start_entry) << ":"
                                  << decode_value(start_entry) << " --> " << start_entry - value
                                  << " = " << decode_key(start_entry - value) << ":"
                                  << decode_value(start_entry - value);
      write_pos(start_pos, start_entry - value);

      DBGC(_flag == kFlagToDebug) << "RESULT: " << dbg_stringify();
      return;
    }

    // Erase element
    DBGC(_flag == kFlagToDebug) << "decrease_by(" << key << ", " << value << "): " << start_entry
                                << " = " << decode_key(start_entry) << ":"
                                << decode_value(start_entry) << " --> erase(" << key << ")";
    KASSERT(start_value == value);

    std::size_t hole_pos = start_pos;
    std::size_t cur_pos = start_pos;

    do {
      cur_pos = hash(cur_pos + 1);
      const MutType cur_entry = read_pos(cur_pos);

      if (cur_entry == 0) {
        break;
      }

      const std::size_t cur_key = decode_key(cur_entry);
      const std::size_t cur_key_hash = hash(cur_key);

      if ((cur_pos > hole_pos && cur_key_hash <= hole_pos) ||
          (cur_pos < hole_pos && cur_key_hash > cur_pos && cur_key_hash <= hole_pos) ||
          (cur_pos > hole_pos && cur_pos < cur_key_hash)) {
        // DBGC(key == kKeyToDebug) << "move hole to pos = " << cur_pos << ", entry = " << cur_entry
        //                          << " = " << decode_key(cur_entry) << ":" <<
        //                          decode_value(cur_entry)
        //                          << " --> entry moved to " << hole_pos;
        write_pos(hole_pos, cur_entry);
        hole_pos = cur_pos;
      } else {
        // DBGC(key == kKeyToDebug) << "skipping pos = " << cur_pos << ", entry = " << cur_entry
        //                          << " = " << decode_key(cur_entry) << ":"
        //                          << decode_value(cur_entry);
        //  skip element
      }
    } while (true);

    write_pos(hole_pos, 0);

    DBGC(_flag == kFlagToDebug) << "RESULT: " << dbg_stringify();

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
        _flag << "/bad hash table state after erasing key = " << key,
        assert::normal
    );
  }

  void increase_by(const MutType key, const MutType value) {
    DBGC(_flag == kFlagToDebug) << "increase_by(key = " << key << ", value = " << value << ")";

    // Assertion: hash-table is locked for other modifying operations
    const auto [pos, entry] = find(key);
    write_pos(pos, encode_key_value(key, decode_value(entry) + value));

    DBGC(_flag == kFlagToDebug) << "RESULT: " << dbg_stringify();
  }

  [[nodiscard]] MutType get(const MutType key) const {
    const auto [pos, entry] = find(key);
    return decode_value(entry);
  }

private:
  [[nodiscard]] std::string dbg_stringify() const {
    std::stringstream ss;
    ss << _flag << "/" << _value_mask + 1 << "(";
    for (std::size_t i = 0; i < _value_mask + 1; ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << i << "/" << read_pos(i) << "=" << decode_key(read_pos(i)) << ":"
         << decode_value(read_pos(i));
    }
    ss << ")";
    return ss.str();
  }

  [[nodiscard]] std::pair<std::size_t, MutType> find(const MutType key) const {
    std::size_t pos = key - 1;
    MutType entry;

    do {
      pos = hash(pos + 1);
      entry = read_pos(pos);
      // DBGC(key == kKeyToDebug) << "do-while: {pos = " << pos << ", entry = " << entry << " = "
      //                          << decode_key(entry) << ":" << decode_value(entry) << "}";
    } while (entry != 0 && decode_key(entry) != key);

    // DBGC(key == kKeyToDebug) << "find(key = " << key << ") --> {pos = " << pos
    //                          << ", entry = " << entry << " = " << decode_key(entry) << ":"
    //                          << decode_value(entry) << "}";

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
  std::uint64_t _flag;
};
} // namespace kaminpar
