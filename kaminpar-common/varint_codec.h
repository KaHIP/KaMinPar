/*******************************************************************************
 * Encoding and decoding methods for VarInts.
 *
 * @file:   varint_codec.h
 * @author: Daniel Salwasser
 * @date:   11.11.2023
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <tuple>
#include <utility>

#include <immintrin.h>

namespace kaminpar {

namespace debug {

/*!
 * Whether to track statistics on encoded VarInts.
 */
static constexpr bool kTrackVarintStats = false;

/*!
 * Statistics about encoded VarInts.
 */
struct VarIntStats {
  std::size_t varint_count;
  std::size_t signed_varint_count;
  std::size_t marked_varint_count;

  std::size_t varint_bytes;
  std::size_t signed_varint_bytes;
  std::size_t marked_varint_bytes;
};

/*!
 * Reset the global statistics on encoded VarInts.
 */
void varint_stats_reset();

/*!
 * Returns a reference to the global statistics on encoded VarInts.
 *
 * @return A reference to the global statistics on encoded VarInts.
 */
VarIntStats &varint_stats_global();

} // namespace debug

/*!
 * Returns the maximum number of bytes that a VarInt needs to be stored.
 *
 * @tparam Int The type of integer whose encoded maximum length is returned.
 */
template <typename Int> [[nodiscard]] std::size_t varint_max_length() {
  return (sizeof(Int) * 8) / 7 + 1;
}

/*!
 * Returns the number of bytes a VarInt needs to be stored.
 *
 * @tparam Int The type of integer whose encoded length is returned.
 * @param Int The integer to store.
 * @return The number of bytes the integer needs to be stored.
 */
template <typename Int> [[nodiscard]] std::size_t varint_length(Int i) {
  std::size_t len = 1;

  while (i > 0b01111111) {
    i >>= 7;
    len++;
  }

  return len;
}

/*!
 * Returns the number of bytes a signed VarInt needs to be stored.
 *
 * @tparam Int The type of integer whose encoded length is returned.
 * @param Int The integer to store.
 * @return The number of bytes the integer needs to be stored.
 */
template <typename Int> [[nodiscard]] std::size_t signed_varint_length(Int i) {
  std::size_t len = 1;

  if (i < 0) {
    i *= -1;
  }
  i >>= 6;

  if (i > 0) {
    len += varint_length(i);
  }

  return len;
}

/*!
 * Returns the number of bytes a marked VarInt needs to be stored.
 *
 * @tparam Int The type of integer whose encoded length is returned.
 * @param Int The integer to store.
 * @return The number of bytes the integer needs to be stored.
 */
template <typename Int> [[nodiscard]] std::size_t marked_varint_length(Int i) {
  std::size_t len = 1;

  i >>= 6;
  if (i > 0) {
    len += varint_length(i);
  }

  return len;
}

/*!
 * Writes an integer to a memory location as a VarInt.
 *
 * @tparam Int The type of integer to encode.
 * @param Int The integer to store.
 * @param ptr The pointer to the memory location to write the integer to.
 * @return The number of bytes that the integer occupies at the memory location.
 */
template <typename Int> std::size_t varint_encode(Int i, std::uint8_t *ptr) {
  std::size_t len = 1;

  while (i > 0b01111111) {
    std::uint8_t octet = (i & 0b01111111) | 0b10000000;
    *ptr = octet;

    i >>= 7;
    ptr++;
    len++;
  }

  std::uint8_t last_octet = i & 0b01111111;
  *ptr = last_octet;

  if (debug::kTrackVarintStats) {
    debug::varint_stats_global().varint_count++;
    debug::varint_stats_global().varint_bytes += len;
  }

  return len;
}

/*!
 * Writes an integer to a memory location as a signed VarInt.
 *
 * @tparam Int The type of integer to encode.
 * @param Int The integer to store.
 * @param ptr The pointer to the memory location to write the integer to.
 * @return The number of bytes that the integer occupies at the memory location.
 */
template <typename Int> std::size_t signed_varint_encode(Int i, std::uint8_t *ptr) {
  std::uint8_t first_octet;

  if (i < 0) {
    i *= -1;

    first_octet = (i & 0b00111111) | 0b01000000;
  } else {
    first_octet = (i & 0b00111111);
  }

  i >>= 6;

  if (i > 0) {
    first_octet |= 0b10000000;
    *ptr = first_octet;

    std::size_t len = varint_encode<Int>(i, ptr + 1) + 1;

    if (debug::kTrackVarintStats) {
      debug::varint_stats_global().signed_varint_count++;
      debug::varint_stats_global().signed_varint_bytes += len;
    }

    return len;
  }

  if (debug::kTrackVarintStats) {
    debug::varint_stats_global().signed_varint_count++;
    debug::varint_stats_global().signed_varint_bytes++;
  }

  *ptr = first_octet;
  return 1;
}

/*!
 * Writes an integer to a memory location as a marked VarInt.
 *
 * @tparam Int The type of integer to encode.
 * @param Int The integer to store.
 * @param marker_set Whether the integer is marked.
 * @param ptr The pointer to the memory location to write the integer to.
 * @return The number of bytes that the integer occupies at the memory location.
 */
template <typename Int>
std::size_t marked_varint_encode(Int i, bool marker_set, std::uint8_t *ptr) {
  std::uint8_t first_octet;

  if (marker_set) {
    first_octet = (i & 0b00111111) | 0b01000000;
  } else {
    first_octet = (i & 0b00111111);
  }

  i >>= 6;

  if (i > 0) {
    first_octet |= 0b10000000;
    *ptr = first_octet;

    std::size_t len = varint_encode<Int>(i, ptr + 1) + 1;

    if (debug::kTrackVarintStats) {
      debug::varint_stats_global().marked_varint_count++;
      debug::varint_stats_global().marked_varint_bytes += len;
    }

    return len;
  }

  if (debug::kTrackVarintStats) {
    debug::varint_stats_global().marked_varint_count++;
    debug::varint_stats_global().marked_varint_bytes++;
  }

  *ptr = first_octet;
  return 1;
}

/*!
 * Reads an integer encoded as a VarInt from a memory location. The decoding is implemented as a
 * loop with non intrinsic operations.
 *
 * @tparam Int The type of integer to decode.
 * @param ptr The pointer to the memory location to read the integer from.
 * @return A pair consisting of the decoded integer and the number of bytes that the encoded integer
 * occupied at the memory location.
 */
template <typename Int>
[[nodiscard]] std::pair<Int, std::size_t> varint_decode_general(const std::uint8_t *ptr) {
  Int result = 0;
  std::size_t shift = 0;
  std::size_t position = 0;

  while (true) {
    const std::uint8_t byte = ptr[position++];

    if ((byte & 0b10000000) == 0) {
      result |= static_cast<Int>(byte) << shift;
      break;
    } else {
      result |= static_cast<Int>(byte & 0b01111111) << shift;
    }

    shift += 7;
  }

  return std::make_pair(result, position);
}

/*!
 * Reads an integer encoded as a VarInt from a memory location.
 *
 * @tparam Int The type of integer to decode.
 * @param ptr The pointer to the memory location to read the integer from.
 * @return A pair consisting of the decoded integer and the number of bytes that the encoded integer
 * occupied at the memory location.
 */
template <typename Int>
[[nodiscard]] std::pair<Int, std::size_t> varint_decode(const std::uint8_t *ptr) {
  return varint_decode_general<Int>(ptr);
}

#ifdef KAMINPAR_COMPRESSION_FAST_DECODING
/*!
 * Reads an 32-bit integer encoded as a VarInt from a memory location. The decoding is implemented
 * as an unrolled loop with intrinsic operations.
 *
 * @param ptr The pointer to the memory location to read the integer from.
 * @return A pair consisting of the decoded integer and the number of bytes that the encoded integer
 * occupied at the memory location.
 */
template <>
inline std::pair<std::uint32_t, std::size_t> varint_decode<std::uint32_t>(const std::uint8_t *ptr) {
  if ((ptr[0] & 0b10000000) == 0) {
    std::uint32_t result = *ptr & 0b01111111;
    return std::make_pair(result, 1);
  }

  if ((ptr[1] & 0b10000000) == 0) {
    std::uint32_t result = _pext_u32(*reinterpret_cast<const std::uint32_t *>(ptr), 0x7F7F);
    return std::make_pair(result, 2);
  }

  if ((ptr[2] & 0b10000000) == 0) {
    std::uint32_t result = _pext_u32(*reinterpret_cast<const std::uint32_t *>(ptr), 0x7F7F7F);
    return std::make_pair(result, 3);
  }

  if ((ptr[3] & 0b10000000) == 0) {
    std::uint32_t result = _pext_u32(*reinterpret_cast<const std::uint32_t *>(ptr), 0x7F7F7F7F);
    return std::make_pair(result, 4);
  }

  std::uint32_t result = static_cast<std::uint32_t>(
      _pext_u64(*reinterpret_cast<const std::uint64_t *>(ptr), 0x7F7F7F7F7F)
  );
  return std::make_pair(result, 5);
}

/*!
 * Reads an 64-bit integer encoded as a VarInt from a memory location. The decoding is implemented
 * as an unrolled loop with intrinsic operations.
 *
 * @param ptr The pointer to the memory location to read the integer from.
 * @return A pair consisting of the decoded integer and the number of bytes that the encoded integer
 * occupied at the memory location.
 */
template <>
inline std::pair<std::uint64_t, std::size_t> varint_decode<std::uint64_t>(const std::uint8_t *ptr) {
  if ((ptr[0] & 0b10000000) == 0) {
    const std::uint64_t result = *ptr & 0b01111111;
    return std::make_pair(result, 1);
  }

  if ((ptr[1] & 0b10000000) == 0) {
    const std::uint64_t result = _pext_u32(*reinterpret_cast<const std::uint32_t *>(ptr), 0x7F7F);
    return std::make_pair(result, 2);
  }

  if ((ptr[2] & 0b10000000) == 0) {
    const std::uint64_t result = _pext_u32(*reinterpret_cast<const std::uint32_t *>(ptr), 0x7F7F7F);
    return std::make_pair(result, 3);
  }

  if ((ptr[3] & 0b10000000) == 0) {
    const std::uint64_t result =
        _pext_u32(*reinterpret_cast<const std::uint32_t *>(ptr), 0x7F7F7F7F);
    return std::make_pair(result, 4);
  }

  if ((ptr[4] & 0b10000000) == 0) {
    const std::uint64_t result =
        _pext_u64(*reinterpret_cast<const std::uint64_t *>(ptr), 0x7F7F7F7F7F);
    return std::make_pair(result, 5);
  }

  if ((ptr[5] & 0b10000000) == 0) {
    const std::uint64_t result =
        _pext_u64(*reinterpret_cast<const std::uint64_t *>(ptr), 0x7F7F7F7F7F7F);
    return std::make_pair(result, 6);
  }

  if ((ptr[6] & 0b10000000) == 0) {
    const std::uint64_t result =
        _pext_u64(*reinterpret_cast<const std::uint64_t *>(ptr), 0x7F7F7F7F7F7F7F);
    return std::make_pair(result, 7);
  }

  if ((ptr[7] & 0b10000000) == 0) {
    const std::uint64_t result =
        _pext_u64(*reinterpret_cast<const std::uint64_t *>(ptr), 0x7F7F7F7F7F7F7F7F);
    return std::make_pair(result, 8);
  }

  if ((ptr[8] & 0b10000000) == 0) {
    const std::uint64_t result =
        _pext_u64(*reinterpret_cast<const std::uint64_t *>(ptr), 0x7F7F7F7F7F7F7F7F) |
        (static_cast<std::uint64_t>(ptr[8] & 0b01111111) << 56);
    return std::make_pair(result, 9);
  }

  const std::uint64_t result =
      _pext_u64(*reinterpret_cast<const std::uint64_t *>(ptr), 0x7F7F7F7F7F7F7F7F) |
      (static_cast<std::uint64_t>(ptr[8] & 0b01111111) << 56) |
      (static_cast<std::uint64_t>(ptr[9] & 0b00000001) << 63);
  return std::make_pair(result, 10);
}
#endif

/*!
 * Reads an integer encoded as a signed VarInt from a memory location.
 *
 * @tparam Int The type of integer to decode.
 * @param ptr The pointer to the memory location to read the integer from.
 * @return A pair consisting of the decoded integer and the number of bytes that the encoded integer
 * occupied at the memory location.
 */
template <typename Int>
[[nodiscard]] std::pair<Int, std::size_t> signed_varint_decode(const std::uint8_t *ptr) {
  const std::uint8_t first_byte = *ptr;
  const bool is_continuation_bit_set = (first_byte & 0b10000000) != 0;
  const bool is_negative = (first_byte & 0b01000000) != 0;

  Int result = first_byte & 0b00111111;
  std::size_t shift = 0;
  std::size_t position = 1;

  if (is_continuation_bit_set) {
    while (true) {
      const std::uint8_t byte = ptr[position++];

      if ((byte & 0b10000000) == 0) {
        result |= static_cast<Int>(byte) << (shift + 6);
        break;
      } else {
        result |= static_cast<Int>(byte & 0b01111111) << (shift + 6);
      }

      shift += 7;
    }
  }

  if (is_negative) {
    result *= -1;
  }

  return std::make_pair(result, position);
}

/*!
 * Reads an integer encoded as a marked VarInt from a memory location.
 *
 * @tparam Int The type of integer to decode.
 * @param ptr The pointer to the memory location to read the integer from.
 * @return A tuple consisting of the decoded integer, whether the markes is set and the number of
 * bytes that the encoded integer occupied at the memory location.
 */
template <typename Int>
[[nodiscard]] std::tuple<Int, bool, std::size_t> marked_varint_decode(const std::uint8_t *ptr) {
  const std::uint8_t first_byte = *ptr;
  const bool is_continuation_bit_set = (first_byte & 0b10000000) != 0;
  const bool is_marker_set = (first_byte & 0b01000000) != 0;

  Int result = first_byte & 0b00111111;
  std::size_t shift = 0;
  std::size_t position = 1;

  if (is_continuation_bit_set) {
    while (true) {
      const std::uint8_t byte = ptr[position++];

      if ((byte & 0b10000000) == 0) {
        result |= static_cast<Int>(byte) << (shift + 6);
        break;
      } else {
        result |= static_cast<Int>(byte & 0b01111111) << (shift + 6);
      }

      shift += 7;
    }
  }

  return std::make_tuple(result, is_marker_set, position);
}

} // namespace kaminpar
