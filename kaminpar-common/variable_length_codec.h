/*******************************************************************************
 * Variable length encoding and decoding methods for integers.
 *
 * @file:   variable_length_codec.h
 * @author: Daniel Salwasser
 * @date:   11.11.2023
 ******************************************************************************/
#pragma once

#include <cstdint>

namespace kaminpar {

struct VarIntCodec {
  template <typename Int> static std::size_t length(Int i) {
    size_t len = 1;

    while (i > 0b01111111) {
      i >>= 7;
      len++;
    }

    return len;
  }

  template <typename Int> static std::size_t length_signed(Int i) {
    size_t len = 1;

    if (i < 0) {
      i *= -1;
    }
    i >>= 6;

    if (i > 0) {
      len += length(i);
    }

    return len;
  }

  template <typename Int> static std::size_t encode(Int i, std::uint8_t *ptr) {
    size_t len = 1;

    while (i > 0b01111111) {
      std::uint8_t octet = (i & 0b01111111) | 0b10000000;
      *ptr = octet;

      i >>= 7;
      ptr++;
      len++;
    }

    std::uint8_t last_octet = i & 0b01111111;
    *ptr = last_octet;

    return len;
  }

  template <typename Int> static std::size_t encode_signed(Int i, std::uint8_t *ptr) {
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
      return encode(i, ptr + 1) + 1;
    }

    *ptr = first_octet;
    return 1;
  }

  template <typename Int> static std::pair<Int, std::size_t> decode(const std::uint8_t *ptr) {
    Int i = 0;
    std::size_t len = 0;

    while (true) {
      std::uint8_t octet = *ptr;
      i |= (octet & 0b01111111) << (7 * len++);

      if ((octet & 0b10000000) == 0) {
        break;
      }

      ptr++;
    }

    return {i, len};
  }

  template <typename Int>
  static std::pair<Int, std::size_t> decode_signed(const std::uint8_t *ptr) {
    Int value = 0;
    std::size_t len = 0;

    std::uint8_t first_octet = *ptr++;
    value |= (first_octet & 0b00111111);

    bool is_continuation_bit_set = (first_octet & 0b10000000) != 0;
    if (is_continuation_bit_set) {
      while (true) {
        std::uint8_t octet = *ptr;
        value |= (octet & 0b01111111) << (6 + 7 * len++);

        if ((octet & 0b10000000) == 0) {
          break;
        }

        ptr++;
      }
    }

    bool is_negative = (first_octet & 0b01000000) != 0;
    if (is_negative) {
      value *= -1;
    }

    return {value, len + 1};
  }
};

} // namespace kaminpar