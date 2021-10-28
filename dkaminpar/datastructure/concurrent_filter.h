/*******************************************************************************
 * @file:   concurrent_filter.h
 *
 * @author: Daniel Seemaier
 * @date:   28.10.2021
 * @brief:  Concurrent filter: flag elements concurrently. Uses growt.
 ******************************************************************************/
#pragma once

#include <tbb/concurrent_set.h>

namespace dkaminpar {
template <typename Key> class ConcurrentFilter {
public:
  bool flag(const Key &key) { return _set.insert(key).second; }
  bool flagged(const Key &key) const { return _set.contains(key); }

private:
  tbb::concurrent_set<Key> _set{};
};
} // namespace dkaminpar