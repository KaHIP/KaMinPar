/*******************************************************************************
 * Owning flat hash map with linear probing and dense iteration over live entries.
 *
 * @file:   flat_hash_map.h
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>

#include "kaminpar-common/datastructures/scalable_vector.h"

namespace kaminpar {

template <typename Key, typename Value> class FlatHashMap {
public:
  struct Entry {
    Key first;
    Value second;
  };

  using iterator = typename ScalableVector<Entry>::iterator;
  using const_iterator = typename ScalableVector<Entry>::const_iterator;

  FlatHashMap() {
    rehash(kInitialCapacity);
  }

  explicit FlatHashMap(const std::size_t capacity) {
    rehash(table_capacity_for(capacity));
  }

  [[nodiscard]] bool empty() const {
    return _entries.empty();
  }

  [[nodiscard]] std::size_t size() const {
    return _entries.size();
  }

  [[nodiscard]] std::size_t capacity() const {
    return _buckets.size() / kInverseMaxLoadFactor;
  }

  void reserve(const std::size_t capacity) {
    if (capacity > this->capacity()) {
      rehash(table_capacity_for(capacity));
    }
  }

  void resize(const std::size_t capacity) {
    reserve(capacity);
  }

  Value &operator[](const Key &key) {
    std::size_t bucket = find_bucket(key);
    if (is_occupied(bucket)) {
      return _entries[_buckets[bucket].entry].second;
    }

    ensure_capacity_for_insert();
    bucket = find_bucket(key);

    const std::size_t entry = _entries.size();
    _buckets[bucket] = Bucket{key, entry};
    _entries.push_back(Entry{key, Value{}});
    _entry_buckets.push_back(bucket);
    return _entries.back().second;
  }

  [[nodiscard]] bool contains(const Key &key) const {
    return is_occupied(find_bucket(key));
  }

  [[nodiscard]] iterator find(const Key &key) {
    const std::size_t bucket = find_bucket(key);
    return is_occupied(bucket) ? _entries.begin() + _buckets[bucket].entry : _entries.end();
  }

  [[nodiscard]] const_iterator find(const Key &key) const {
    const std::size_t bucket = find_bucket(key);
    return is_occupied(bucket) ? _entries.begin() + _buckets[bucket].entry : _entries.end();
  }

  [[nodiscard]] Value *get_if_contained(const Key &key) {
    const auto it = find(key);
    return it != end() ? &it->second : nullptr;
  }

  [[nodiscard]] const Value *get_if_contained(const Key &key) const {
    const auto it = find(key);
    return it != end() ? &it->second : nullptr;
  }

  [[nodiscard]] iterator begin() {
    return _entries.begin();
  }

  [[nodiscard]] iterator end() {
    return _entries.end();
  }

  [[nodiscard]] const_iterator begin() const {
    return _entries.begin();
  }

  [[nodiscard]] const_iterator end() const {
    return _entries.end();
  }

  [[nodiscard]] const_iterator cbegin() const {
    return _entries.cbegin();
  }

  [[nodiscard]] const_iterator cend() const {
    return _entries.cend();
  }

  [[nodiscard]] FlatHashMap &entries() {
    return *this;
  }

  [[nodiscard]] const FlatHashMap &entries() const {
    return *this;
  }

  void clear() {
    for (const std::size_t bucket : _entry_buckets) {
      _buckets[bucket].entry = kInvalidEntry;
    }

    _entries.clear();
    _entry_buckets.clear();
  }

private:
  struct Bucket {
    Key key{};
    std::size_t entry = kInvalidEntry;
  };

  static constexpr std::size_t kInvalidEntry = std::numeric_limits<std::size_t>::max();
  static constexpr std::size_t kInitialCapacity = 16;
  static constexpr std::size_t kInverseMaxLoadFactor = 2;

  [[nodiscard]] static std::uint64_t mix(std::uint64_t key) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key;
  }

  [[nodiscard]] static std::size_t table_capacity_for(const std::size_t capacity) {
    std::size_t table_capacity = kInitialCapacity;
    const std::size_t min_table_capacity = std::max<std::size_t>(
        kInitialCapacity,
        kInverseMaxLoadFactor * std::max<std::size_t>(capacity, 1)
    );

    while (table_capacity < min_table_capacity) {
      table_capacity *= 2;
    }

    return table_capacity;
  }

  [[nodiscard]] static std::size_t hash_key(const Key &key) {
    if constexpr (std::is_enum_v<Key>) {
      return static_cast<std::size_t>(
          mix(static_cast<std::uint64_t>(static_cast<std::underlying_type_t<Key>>(key)))
      );
    } else if constexpr (std::is_integral_v<Key>) {
      return static_cast<std::size_t>(mix(static_cast<std::uint64_t>(key)));
    } else {
      return static_cast<std::size_t>(mix(static_cast<std::uint64_t>(std::hash<Key>{}(key))));
    }
  }

  [[nodiscard]] bool is_occupied(const std::size_t bucket) const {
    return _buckets[bucket].entry != kInvalidEntry;
  }

  [[nodiscard]] std::size_t find_bucket(const Key &key) const {
    std::size_t bucket = hash_key(key) & (_buckets.size() - 1);

    while (is_occupied(bucket) && !(_buckets[bucket].key == key)) {
      bucket = (bucket + 1) & (_buckets.size() - 1);
    }

    return bucket;
  }

  void ensure_capacity_for_insert() {
    if ((_entries.size() + 1) * kInverseMaxLoadFactor > _buckets.size()) {
      rehash(2 * _buckets.size());
    }
  }

  void rehash(const std::size_t table_capacity) {
    ScalableVector<Bucket> new_buckets(table_capacity);

    for (std::size_t entry = 0; entry < _entries.size(); ++entry) {
      const Key &key = _entries[entry].first;
      std::size_t bucket = hash_key(key) & (new_buckets.size() - 1);

      while (new_buckets[bucket].entry != kInvalidEntry) {
        bucket = (bucket + 1) & (new_buckets.size() - 1);
      }

      new_buckets[bucket] = Bucket{key, entry};
      _entry_buckets[entry] = bucket;
    }

    _buckets = std::move(new_buckets);
  }

  ScalableVector<Bucket> _buckets;
  ScalableVector<Entry> _entries;
  ScalableVector<std::size_t> _entry_buckets;
};

} // namespace kaminpar
