/*******************************************************************************
 * A two-level vector which stores small values in a contiguous vector and large values in a hash
 * table.
 *
 * @file:   concurrent_two_level_vector.h
 * @author: Daniel Salwasser
 * @date:   18.01.2024
 ******************************************************************************/
#pragma once

#include <limits>

#include <kassert/kassert.hpp>

#ifdef KAMINPAR_USES_GROWT
#include <allocator/alignedallocator.hpp>
#include <data-structures/table_config.hpp>
#include <utils/hash/murmur2_hash.hpp>
#else
#include <tbb/concurrent_hash_map.h>
#endif

#include "kaminpar-common/datastructures/scalable_vector.h"

namespace kaminpar {

#ifdef KAMINPAR_USES_GROWT
/*!
 * A concurrent two-level vector which consists of a vector and a hash table. The data structure
 * stores values of small size directly in the vector and bigger values in the hash table.
 *
 * @tparam Value The type of integer to store.
 * @tparam Size The type of integer to access the values with.
 * @tparam FirstValue The type of integer to store in the vector. It has to be smaller than the
 * value type.
 */
template <typename Value, typename Size = std::size_t, typename FirstValue = std::uint16_t>
class ConcurrentTwoLevelVector {
  static_assert(std::numeric_limits<Value>::is_integer);
  static_assert(std::numeric_limits<FirstValue>::is_integer);
  static_assert(sizeof(FirstValue) < sizeof(Value));

  using HasherType = utils_tm::hash_tm::murmur2_hash;
  using AllocatorType = ::growt::AlignedAllocator<>;
  using ConcurrentHashTable = typename ::growt::
      table_config<std::size_t, Value, HasherType, AllocatorType, hmod::growable>::table_type;

  // The maximum value of the FirstValue type is used as a special marker in the vector to indicate
  // that the value is stored in the hash table.
  static constexpr FirstValue kMaxFirstValue = std::numeric_limits<FirstValue>::max();

public:
  /*!
   * Constructs a new ConcurrentTwoLevelVector.
   *
   * @param capacity The capacity of the vector.
   */
  ConcurrentTwoLevelVector(const Size capacity = 0) : _values(capacity), _table(0) {}

  ConcurrentTwoLevelVector(const ConcurrentTwoLevelVector &) = delete;
  ConcurrentTwoLevelVector &operator=(const ConcurrentTwoLevelVector &) = delete;

  ConcurrentTwoLevelVector(ConcurrentTwoLevelVector &&) noexcept = default;
  ConcurrentTwoLevelVector &operator=(ConcurrentTwoLevelVector &&) noexcept = default;

  /*!
   * Resizes the vector.
   *
   * @param capacity The capacity to resize to.
   */
  void resize(const Size capacity) {
    _values.resize(capacity);
  }

  /*!
   * Frees the memory used by this data structure.
   */
  void free() {
    _values.clear();
    _values.shrink_to_fit();

    _table = ConcurrentHashTable(0);
  }

  /*!
   * Resets the vector such that new elements can be inserted.
   */
  void reset() {
    // As Growt does not provide a clear function, just create a new hash table.
    _table = ConcurrentHashTable(0);
  }

  /*!
   * Accesses a value at a given position.
   *
   * @param pos The position of the value in the vector to return.
   * @return The value at the given position.
   */
  [[nodiscard]] Value operator[](const Size pos) {
    KASSERT(pos < _values.size());

    const Value value = _values[pos];
    if (value < kMaxFirstValue) {
      return value;
    }

    auto table_handle = _table.get_handle();
    auto it = table_handle.find(pos);
    while (it == table_handle.end()) {
      it = table_handle.find(pos);
    }

    return (*it).second;
  }

  /*!
   * Inserts a value at a given position.
   *
   * @param pos The position in the vector at which the value is to be inserted.
   * @param value The value to insert.
   */
  void insert(const Size pos, const Value value) {
    KASSERT(pos < _values.size());

    if (value < kMaxFirstValue) {
      _values[pos] = value;
    } else {
      _values[pos] = kMaxFirstValue;
      _table.get_handle().insert(pos, value);
    }
  }

  /**
   * Adds atomically a value to a value already stored in the vector.
   *
   * @param pos The position in the vector at which the value is to be added.
   * @param delta The value to add.
   */
  void atomic_add(const Size pos, const Value delta) {
    KASSERT(pos < _values.size());

    FirstValue value = _values[pos];
    bool success;
    do {
      if (value == kMaxFirstValue) {
        _table.get_handle().insert_or_update(
            pos, delta, [&](auto &lhs, const auto rhs) { return lhs += rhs; }, delta
        );
        break;
      }

      const Value new_value = static_cast<Value>(value) + delta;
      if (new_value < kMaxFirstValue) {
        success = __atomic_compare_exchange_n(
            &_values[pos], &value, new_value, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED
        );
      } else {
        success = __atomic_compare_exchange_n(
            &_values[pos], &value, kMaxFirstValue, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED
        );

        if (success) {
          _table.get_handle().insert_or_update(
              pos, new_value, [&](auto &lhs, const auto rhs) { return lhs += rhs; }, new_value
          );
          break;
        }
      }

    } while (!success);
  }

  /**
   * Subtracts atomically a value from a value already stored in the vector.
   *
   * @param pos The position in the vector at which the value is to be subtracted.
   * @param delta The value to subtract.
   */
  void atomic_sub(const Size pos, const Value delta) {
    KASSERT(pos < _values.size());

    FirstValue value = _values[pos];
    bool success;
    do {
      if (value == kMaxFirstValue) {
        _table.get_handle().insert_or_update(
            pos, -delta, [&](auto &lhs, const auto rhs) { return lhs -= rhs; }, delta
        );
        break;
      }

      success = __atomic_compare_exchange_n(
          &_values[pos], &value, value - delta, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED
      );
    } while (!success);
  }

private:
  scalable_vector<FirstValue> _values;
  ConcurrentHashTable _table;
};
#else
/*!
 * A concurrent two-level vector which consists of a vector and a hash table. The data structure
 * stores values of small size directly in the vector and bigger values in the hash table.
 *
 * @tparam Value The type of integer to store.
 * @tparam Size The type of integer to access the values with.
 * @tparam FirstValue The type of integer to store in the vector. It has to be smaller than the
 * value type.
 */
template <typename Value, typename Size = std::size_t, typename FirstValue = std::uint16_t>
class ConcurrentTwoLevelVector {
  static_assert(std::numeric_limits<Value>::is_integer);
  static_assert(std::numeric_limits<FirstValue>::is_integer);
  static_assert(sizeof(FirstValue) < sizeof(Value));

  using ConcurrentHashTable = tbb::concurrent_hash_map<Size, Value>;

  // The maximum value of the FirstValue type is used as a special marker in the vector to indicate
  // that the value is stored in the hash table.
  static constexpr FirstValue kMaxFirstValue = std::numeric_limits<FirstValue>::max();

public:
  /*!
   * Constructs a new ConcurrentTwoLevelVector.
   *
   * @param capacity The capacity of the vector.
   */
  ConcurrentTwoLevelVector(const Size capacity = 0) : _values(capacity) {}

  ConcurrentTwoLevelVector(const ConcurrentTwoLevelVector &) = delete;
  ConcurrentTwoLevelVector &operator=(const ConcurrentTwoLevelVector &) = delete;

  ConcurrentTwoLevelVector(ConcurrentTwoLevelVector &&) noexcept = default;
  ConcurrentTwoLevelVector &operator=(ConcurrentTwoLevelVector &&) noexcept = default;

  /*!
   * Resizes the vector.
   *
   * @param capacity The capacity to resize to.
   */
  void resize(const Size capacity) {
    _values.resize(capacity);
  }

  /*!
   * Frees the memory used by this data structure.
   */
  void free() {
    _values.clear();
    _values.shrink_to_fit();

    _table.clear();
  }

  /*!
   * Resets the vector such that new elements can be inserted.
   */
  void reset() {
    _table.clear();
  }

  /*!
   * Accesses a value at a given position.
   *
   * @param pos The position of the value in the vector to return.
   * @return The value at the given position.
   */
  [[nodiscard]] Value operator[](const Size pos) {
    KASSERT(pos < _values.size());

    const Value value = _values[pos];
    if (value < kMaxFirstValue) {
      return value;
    }

    typename ConcurrentHashTable::const_accessor entry;
    bool found;
    do {
      found = _table.find(entry, pos);
    } while (!found);

    return entry->second;
  }

  /*!
   * Inserts a value at a given position.
   *
   * @param pos The position in the vector at which the value is to be inserted.
   * @param value The value to insert.
   */
  void insert(const Size pos, const Value value) {
    KASSERT(pos < _values.size());

    if (value < kMaxFirstValue) {
      _values[pos] = value;
    } else {
      _values[pos] = kMaxFirstValue;

      typename ConcurrentHashTable::accessor entry;
      _table.insert(entry, pos);
      entry->second = value;
    }
  }

  /**
   * Adds atomically a value to a value already stored in the vector.
   *
   * @param pos The position in the vector at which the value is to be added.
   * @param delta The value to add.
   */
  void atomic_add(const Size pos, const Value delta) {
    KASSERT(pos < _values.size());

    Value value = _values[pos];
    bool success;
    do {
      if (value == kMaxFirstValue) {
        typename ConcurrentHashTable::accessor entry;
        if (_table.insert(entry, pos)) {
          entry->second = delta;
        } else {
          entry->second += delta;
        }

        break;
      }

      const Value new_value = value + delta;
      if (new_value < kMaxFirstValue) {
        success = __atomic_compare_exchange_n(
            &_values[pos], &value, new_value, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED
        );
      } else {
        success = __atomic_compare_exchange_n(
            &_values[pos], &value, kMaxFirstValue, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED
        );

        if (success) {
          typename ConcurrentHashTable::accessor entry;
          if (_table.insert(entry, pos)) {
            entry->second = new_value;
          } else {
            entry->second += new_value;
          }

          break;
        }
      }

    } while (!success);
  }

  /**
   * Subtracts atomically a value from a value already stored in the vector.
   *
   * @param pos The position in the vector at which the value is to be subtracted.
   * @param delta The value to subtract.
   */
  void atomic_sub(const Size pos, const Value delta) {
    KASSERT(pos < _values.size());

    Value value = _values[pos];
    bool success;
    do {
      if (value == kMaxFirstValue) {
        typename ConcurrentHashTable::accessor entry;
        if (_table.insert(entry, pos)) {
          entry->second = -delta;
        } else {
          entry->second -= delta;
        }

        break;
      }

      success = __atomic_compare_exchange_n(
          &_values[pos], &value, value - delta, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED
      );
    } while (!success);
  }

private:
  scalable_vector<FirstValue> _values;
  ConcurrentHashTable _table;
};
#endif

} // namespace kaminpar
