/*******************************************************************************
 * Utility functions to map integers to bit buckets.
 *
 * @file:   degree_buckets.h
 * @author: Daniel Seemaier
 * @date:   09.05.2023
 ******************************************************************************/
#pragma once

#include <cstdlib>
#include <limits>

#include "kaminpar-common/math.h"

namespace kaminpar {
template <typename NodeID>
static constexpr std::size_t kNumberOfDegreeBuckets = std::numeric_limits<NodeID>::digits + 1;

template <typename NodeID> inline NodeID lowest_degree_in_bucket(const std::size_t bucket) {
  return (1u << bucket) >> 1u;
}

template <typename NodeID> inline NodeID degree_bucket(const NodeID degree) {
  return (degree == 0) ? 0 : math::floor_log2(degree) + 1;
}
} // namespace kaminpar
