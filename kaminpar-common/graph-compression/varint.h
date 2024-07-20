/// Encoding and decoding routines for variable-length integers (VarInts).
/// @file varint.hpp
/// @author Daniel Salwasser
#pragma once

#include <concepts>
#include <cstdint>
#include <type_traits>
#include <utility>

#ifdef KAMINPAR_COMPRESSION_FAST_DECODING
#include <immintrin.h>
#endif

namespace kaminpar {

template <typename Int> [[nodiscard]] constexpr std::size_t varint_max_length() {
  return (sizeof(Int) * 8) / 7 + 1;
}

template <std::integral Int> [[nodiscard]] std::size_t varint_length(Int i) {
  std::size_t len = 1;

  while (i > 0b01111111) {
    i >>= 7;
    len++;
  }

  return len;
}

template <std::integral Int> std::size_t varint_encode(Int i, std::uint8_t *ptr) {
  std::size_t len = 1;

  while (i > 0b01111111) {
    const std::uint8_t octet = (i & 0b01111111) | 0b10000000;
    *ptr = octet;

    i >>= 7;
    ptr += 1;
    len += 1;
  }

  const std::uint8_t last_octet = i & 0b01111111;
  *ptr = last_octet;

  return len;
}

template <std::integral Int> void varint_encode(Int i, std::uint8_t **ptr) {
  while (i > 0b01111111) {
    const std::uint8_t octet = (i & 0b01111111) | 0b10000000;
    **ptr = octet;

    i >>= 7;
    *ptr += 1;
  }

  const std::uint8_t last_octet = i & 0b01111111;
  **ptr = last_octet;
  *ptr += 1;
}

template <std::integral Int> [[nodiscard]] Int varint_decode(const std::uint8_t *data) {
  Int value = 0;

  Int shift = 0;
  while (true) {
    const std::uint8_t byte = *data;

    if ((byte & 0b10000000) == 0) {
      value |= static_cast<Int>(byte) << shift;
      break;
    } else {
      value |= static_cast<Int>(byte & 0b01111111) << shift;
    }

    shift += 7;
    data += 1;
  }

  return value;
}

template <std::integral Int> [[nodiscard]] Int varint_decode_loop(const std::uint8_t **data) {
  Int value = 0;

  Int shift = 0;
  while (true) {
    const std::uint8_t octet = **data;
    *data += 1;

    if ((octet & 0b10000000) == 0) {
      value |= static_cast<Int>(octet) << shift;
      break;
    } else {
      value |= static_cast<Int>(octet & 0b01111111) << shift;
    }

    shift += 7;
  }

  return value;
}

template <std::integral Int>
[[nodiscard]] Int varint_decode_pext_unrolled(const std::uint8_t **data) {
#ifdef KAMINPAR_COMPRESSION_FAST_DECODING
  const std::uint8_t *data_ptr = *data;
  if ((data_ptr[0] & 0b10000000) == 0) {
    const std::uint32_t result = *data_ptr & 0b01111111;
    *data += 1;
    return result;
  }

  if ((data_ptr[1] & 0b10000000) == 0) {
    const std::uint32_t result =
        _pext_u32(*reinterpret_cast<const std::uint32_t *>(data_ptr), 0x7F7F);
    *data += 2;
    return result;
  }

  if ((data_ptr[2] & 0b10000000) == 0) {
    const std::uint32_t result =
        _pext_u32(*reinterpret_cast<const std::uint32_t *>(data_ptr), 0x7F7F7F);
    *data += 3;
    return result;
  }

  if ((data_ptr[3] & 0b10000000) == 0) {
    const std::uint32_t result =
        _pext_u32(*reinterpret_cast<const std::uint32_t *>(data_ptr), 0x7F7F7F7F);
    *data += 4;
    return result;
  }

  const std::uint32_t result = static_cast<std::uint32_t>(
      _pext_u64(*reinterpret_cast<const std::uint64_t *>(data_ptr), 0x7F7F7F7F7F)
  );
  *data += 5;
  return result;
#else
  return varint_decode_loop<Int>(data);
#endif
}

template <std::integral Int>
[[nodiscard]] Int varint_decode_pext_branchless(const std::uint8_t **data) {
#ifdef KAMINPAR_COMPRESSION_FAST_DECODING
  const std::uint8_t *data_ptr = *data;

  const std::uint64_t word = *reinterpret_cast<const std::uint64_t *>(data_ptr);
  const std::uint64_t continuation_bits = ~word & 0x8080808080;
  const std::uint64_t mask = continuation_bits ^ (continuation_bits - 1);
  const std::uint64_t length = (std::countr_zero(continuation_bits) + 1) / 8;

  const Int result = _pext_u64(word & mask, 0x7F7F7F7F7F);
  *data += length;
  return result;
#else
  return varint_decode_loop<Int>(data);
#endif
}

template <std::integral Int, bool kOverwriteFastPEXT = false>
[[nodiscard]] Int varint_decode(const std::uint8_t **data) {
  // TOOD: Implement 64-bit decoding routine with PEXT instructions
  if constexpr (!kOverwriteFastPEXT && sizeof(Int) == 4) {
#ifdef KAMINPAR_COMPRESSION_FAST_DECODING
    return varint_decode_pext_unrolled<Int>(data);
#endif
  }

  return varint_decode_loop<Int>(data);
}

template <std::integral Int> [[nodiscard]] std::size_t marked_varint_length(Int i) {
  std::size_t len = 1;
  i >>= 6;

  if (i > 0) {
    len += varint_length(i);
  }

  return len;
}

template <std::integral Int>
void marked_varint_encode(Int i, const bool marked, std::uint8_t **ptr) {
  std::uint8_t first_octet = i & 0b00111111;
  if (marked) {
    first_octet |= 0b01000000;
  }

  i >>= 6;

  if (i == 0) {
    **ptr = first_octet;
    *ptr += 1;
    return;
  }

  first_octet |= 0b10000000;
  **ptr = first_octet;
  *ptr += 1;

  varint_encode(i, ptr);
}

template <std::integral Int>
[[nodiscard]] std::pair<Int, bool> marked_varint_decode(const std::uint8_t *ptr) {
  const std::uint8_t first_octet = *ptr;
  ptr += 1;

  const bool is_continuation_bit_set = (first_octet & 0b10000000) != 0;
  const bool is_marked = (first_octet & 0b01000000) != 0;

  Int result = first_octet & 0b00111111;
  if (is_continuation_bit_set) {
    Int shift = 6;

    while (true) {
      const std::uint8_t octet = *ptr;
      ptr += 1;

      if ((octet & 0b10000000) == 0) {
        result |= static_cast<Int>(octet) << shift;
        break;
      } else {
        result |= static_cast<Int>(octet & 0b01111111) << shift;
      }

      shift += 7;
    }
  }

  return std::make_pair(result, is_marked);
}

template <std::integral Int>
[[nodiscard]] std::pair<Int, bool> marked_varint_decode(const std::uint8_t **ptr) {
  const std::uint8_t first_octet = **ptr;
  *ptr += 1;

  const bool is_continuation_bit_set = (first_octet & 0b10000000) != 0;
  const bool is_marked = (first_octet & 0b01000000) != 0;

  Int result = first_octet & 0b00111111;
  if (is_continuation_bit_set) {
    Int shift = 6;

    while (true) {
      const std::uint8_t octet = **ptr;
      *ptr += 1;

      if ((octet & 0b10000000) == 0) {
        result |= static_cast<Int>(octet) << shift;
        break;
      } else {
        result |= static_cast<Int>(octet & 0b01111111) << shift;
      }

      shift += 7;
    }
  }

  return std::make_pair(result, is_marked);
}

template <std::integral Int> [[nodiscard]] std::make_unsigned_t<Int> zigzag_encode(const Int i) {
  return (i >> (sizeof(Int) * 8 - 1)) ^ (i << 1);
}

template <std::integral Int> [[nodiscard]] std::make_signed_t<Int> zigzag_decode(const Int i) {
  return (i >> 1) ^ -(i & 1);
}

template <std::integral Int> [[nodiscard]] std::size_t signed_varint_length(const Int i) {
  return varint_length(zigzag_encode(i));
}

template <std::integral Int> std::size_t signed_varint_encode(const Int i, std::uint8_t *ptr) {
  return varint_encode(zigzag_encode(i), ptr);
}

template <std::integral Int> void signed_varint_encode(const Int i, std::uint8_t **ptr) {
  varint_encode(zigzag_encode(i), ptr);
}

template <std::integral Int> [[nodiscard]] Int signed_varint_decode(const std::uint8_t *data) {
  return zigzag_decode(varint_decode<std::make_unsigned_t<Int>>(data));
}

template <std::integral Int> [[nodiscard]] Int signed_varint_decode(const std::uint8_t **data) {
  return zigzag_decode(varint_decode<std::make_unsigned_t<Int>>(data));
}

} // namespace kaminpar
