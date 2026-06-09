/*******************************************************************************
 * Lightweight validation helpers for graph IO.
 *
 * @file:   io_validation.h
 ******************************************************************************/
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "kaminpar-io/util/io_exception.h"

namespace kaminpar::io {

inline void raise_if(const bool condition, const char *message) {
  if (condition) [[unlikely]] {
    throw IOException(message);
  }
}

inline std::size_t checked_add(const std::size_t lhs, const std::size_t rhs, const char *message) {
  raise_if(lhs > std::numeric_limits<std::size_t>::max() - rhs, message);
  return lhs + rhs;
}

template <typename Lhs>
std::size_t checked_mul(const Lhs lhs, const std::size_t rhs, const char *message) {
  static_assert(std::is_unsigned_v<Lhs>);
  raise_if(rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs, message);
  return static_cast<std::size_t>(lhs) * rhs;
}

template <typename Int> void ensure_fits(const std::uint64_t value, const char *message) {
  raise_if(value > static_cast<std::uint64_t>(std::numeric_limits<Int>::max()), message);
}

template <typename Int> Int checked_cast(const std::uint64_t value, const char *message) {
  ensure_fits<Int>(value, message);
  return static_cast<Int>(value);
}

template <typename Weight, typename RawWeight>
Weight parse_positive_weight(const RawWeight weight, const char *message) {
  if constexpr (std::is_signed_v<RawWeight>) {
    raise_if(weight <= 0, message);
  } else {
    raise_if(weight == 0, message);
  }
  raise_if(
      static_cast<std::uint64_t>(weight) >
          static_cast<std::uint64_t>(std::numeric_limits<Weight>::max()),
      message
  );
  return static_cast<Weight>(weight);
}

inline std::uint64_t
fetch_unsigned(const std::uint8_t *data, const bool is_64_bit, const std::uint64_t pos) {
  if (is_64_bit) [[unlikely]] {
    return reinterpret_cast<const std::uint64_t *>(data)[pos];
  }
  return reinterpret_cast<const std::uint32_t *>(data)[pos];
}

inline std::int64_t
fetch_signed(const std::uint8_t *data, const bool is_64_bit, const std::uint64_t pos) {
  if (is_64_bit) [[unlikely]] {
    return reinterpret_cast<const std::int64_t *>(data)[pos];
  }
  return reinterpret_cast<const std::int32_t *>(data)[pos];
}

} // namespace kaminpar::io
