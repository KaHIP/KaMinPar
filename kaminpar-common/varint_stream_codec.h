/*******************************************************************************
 * Encoding and decoding methods for the StreamVByte codec.
 *
 * @file:   varint_stream_codec.h
 * @author: Daniel Salwasser
 * @date:   29.12.2023
 ******************************************************************************/
#pragma once

#include <array>
#include <cstdint>

#include <immintrin.h>

#include "kaminpar-common/constexpr_utils.h"
#include "kaminpar-common/varint_codec.h"

namespace kaminpar {

/*!
 * An encoder for writing variable length integers with the StreamVByte codec.
 *
 * @tparam Int The type of integer to encode.
 */
template <typename Int> class VarIntStreamEncoder {
  static_assert(sizeof(Int) == 4);

public:
  /*!
   * Constructs a new VarIntStreamEncoder.
   *
   * @param ptr The pointer to the memory location where the encoded integers are written.
   * @param count The amount of integers to encode.
   */
  VarIntStreamEncoder(std::uint8_t *ptr, std::size_t count)
      : _control_bytes_ptr(ptr),
        _data_ptr(ptr + count / 4 + ((count % 4) != 0)),
        _count(count),
        _buffered(0) {}

  /*!
   * Encodes an integer.
   *
   * @param i The integer to encode.
   * @return The number of bytes that the integer requires to be stored in encoded format. It
   * includes the control byte if it is the last integer of a block.
   */
  std::size_t add(Int i) {
    if (_buffered == 3) {
      _buffer[3] = i;
      write_stream();

      _buffered = 0;
      return needed_bytes(i);
    }

    _buffer[_buffered] = i;
    return needed_bytes(i) + (_buffered++ == 0);
  }

  /*!
   * Writes the remaining integers added to the encoder which do not form a complete block to
   * memory.
   */
  void flush() {
    if (_buffered == 0) {
      return;
    }

    const std::uint8_t control_byte =
        ((needed_bytes(_buffer[3]) - 1) << 6) | (((needed_bytes(_buffer[2]) - 1) & 0b11) << 4) |
        (((needed_bytes(_buffer[1]) - 1) & 0b11) << 2) | ((needed_bytes(_buffer[0]) - 1) & 0b11);
    *_control_bytes_ptr++ = control_byte;

    for (std::size_t i = 0; i < _buffered; ++i) {
      Int value = _buffer[i];
      do {
        *_data_ptr++ = static_cast<std::uint8_t>(value);
        value >>= 8;
      } while (value > 0);
    }
  }

private:
  std::uint8_t *_control_bytes_ptr;
  std::uint8_t *_data_ptr;
  const std::size_t _count;

  std::size_t _buffered;
  std::array<Int, 4> _buffer;

  void write_stream() {
    const std::uint8_t control_byte =
        ((needed_bytes(_buffer[3]) - 1) << 6) | (((needed_bytes(_buffer[2]) - 1) & 0b11) << 4) |
        (((needed_bytes(_buffer[1]) - 1) & 0b11) << 2) | ((needed_bytes(_buffer[0]) - 1) & 0b11);
    *_control_bytes_ptr++ = control_byte;

    for (Int value : _buffer) {
      do {
        *_data_ptr++ = static_cast<std::uint8_t>(value);
        value >>= 8;
      } while (value > 0);
    }
  }

  std::uint8_t needed_bytes(Int i) const {
    std::size_t len = 1;

    while (i > 0b11111111) {
      i >>= 8;
      len++;
    }

    return len;
  }
};

/*!
 * A decoder for reading variable length integers stored with the StreamVByte codec.
 *
 * @tparam Int The type of integer to decode.
 */
template <typename Int> class VarIntStreamDecoder {
  static_assert(sizeof(Int) == 4);

  static constexpr std::array<std::uint8_t, 256> create_length_table() {
    std::array<std::uint8_t, 256> length_table{};

    constexpr_for<256>([&](const std::uint8_t control_byte) {
      length_table[control_byte] = 0;

      constexpr_for<4>([&](const std::uint8_t i) {
        const std::uint8_t length = ((control_byte >> (2 * i)) & 0b11) + 1;
        length_table[control_byte] += length;
      });
    });

    return length_table;
  }

  static constexpr std::array<std::array<std::uint8_t, 16>, 256> create_shuffle_table() {
    std::array<std::array<std::uint8_t, 16>, 256> shuffle_table{};

    constexpr_for<256>([&](const std::uint8_t control_byte) {
      std::uint8_t byte = 0;
      std::uint8_t pos = 0;

      constexpr_for<4>([&](const std::uint8_t i) {
        std::uint8_t c = (control_byte >> (2 * i)) & 0b11;

        std::uint8_t j = 0;
        while (j <= c) {
          shuffle_table[control_byte][pos++] = byte++;
          j += 1;
        }

        while (j < 4) {
          shuffle_table[control_byte][pos++] = 0b11111111;
          j += 1;
        }
      });
    });

    return shuffle_table;
  }

  static const constexpr std::array<std::uint8_t, 256> kLengthTable = create_length_table();

  static const constexpr std::array<std::array<std::uint8_t, 16>, 256> kShuffleTable =
      create_shuffle_table();

public:
  /*!
   * Constructs a new VarIntStreamDecoder.
   *
   * @param ptr The pointer to the memory location where the encoded integers are stored.
   * @param count The amount of integers that are stored at the memory location.
   */
  VarIntStreamDecoder(const std::uint8_t *ptr, const std::size_t count)
      : _control_bytes_ptr(ptr),
        _control_bytes(count / 4),
        _data_ptr(ptr + _control_bytes + ((count % 4) != 0)),
        _count(count) {}

  /*!
   * Decodes the encoded integers.
   *
   * @param max_count The amount of integers to decode, it has to be less then the amount of
   * integers stored that are stored.
   * @param l The function to be called with the decoded integers, i.e. the function has one
   * parameter of type Int.
   */
  template <typename Lambda> void decode(const std::size_t max_count, Lambda &&l) {
    constexpr bool non_stoppable = std::is_void<std::invoke_result_t<Lambda, std::uint32_t>>::value;

    // max_count = std::min(max_count, _count);

    const std::size_t control_bytes = max_count / 4;
    for (std::size_t i = 0; i < control_bytes; ++i) {
      const std::uint8_t control_byte = _control_bytes_ptr[i];
      const std::uint8_t length = kLengthTable[control_byte];

      __m128i data = _mm_loadu_si128((const __m128i *)_data_ptr);
      _data_ptr += length;

      const std::uint8_t *shuffle_mask = kShuffleTable[control_byte].data();
      data = _mm_shuffle_epi8(data, *(const __m128i *)shuffle_mask);

      if constexpr (non_stoppable) {
        l(_mm_extract_epi32(data, 0));
        l(_mm_extract_epi32(data, 1));
        l(_mm_extract_epi32(data, 2));
        l(_mm_extract_epi32(data, 3));
      } else {
        if (l(_mm_extract_epi32(data, 0))) {
          return;
        }

        if (l(_mm_extract_epi32(data, 1))) {
          return;
        }

        if (l(_mm_extract_epi32(data, 2))) {
          return;
        }

        if (l(_mm_extract_epi32(data, 3))) {
          return;
        }
      }
    }

    switch (max_count % 4) {
    case 1: {
      const std::uint8_t control_byte = _control_bytes_ptr[control_bytes];
      const std::uint8_t *shuffle_mask = kShuffleTable[control_byte].data();

      __m128i data = _mm_loadu_si128((const __m128i *)_data_ptr);
      data = _mm_shuffle_epi8(data, *(const __m128i *)shuffle_mask);

      if constexpr (non_stoppable) {
        l(_mm_extract_epi32(data, 0));
      } else {
        if (l(_mm_extract_epi32(data, 0))) {
          return;
        }
      }
      break;
    }
    case 2: {
      const std::uint8_t control_byte = _control_bytes_ptr[control_bytes];
      const std::uint8_t *shuffle_mask = kShuffleTable[control_byte].data();

      __m128i data = _mm_loadu_si128((const __m128i *)_data_ptr);
      data = _mm_shuffle_epi8(data, *(const __m128i *)shuffle_mask);

      if constexpr (non_stoppable) {
        l(_mm_extract_epi32(data, 0));
        l(_mm_extract_epi32(data, 1));
      } else {
        if (l(_mm_extract_epi32(data, 0))) {
          return;
        }

        if (l(_mm_extract_epi32(data, 1))) {
          return;
        }
      }
      break;
    }
    case 3: {
      const std::uint8_t control_byte = _control_bytes_ptr[control_bytes];
      const std::uint8_t *shuffle_mask = kShuffleTable[control_byte].data();

      __m128i data = _mm_loadu_si128((const __m128i *)_data_ptr);
      data = _mm_shuffle_epi8(data, *(const __m128i *)shuffle_mask);

      if constexpr (non_stoppable) {
        l(_mm_extract_epi32(data, 0));
        l(_mm_extract_epi32(data, 1));
        l(_mm_extract_epi32(data, 2));
      } else {
        if (l(_mm_extract_epi32(data, 0))) {
          return;
        }

        if (l(_mm_extract_epi32(data, 1))) {
          return;
        }

        if (l(_mm_extract_epi32(data, 2))) {
          return;
        }
      }
      break;
    }
    }
  }

private:
  const std::uint8_t *_control_bytes_ptr;
  const std::size_t _control_bytes;
  const std::uint8_t *_data_ptr;
  const std::size_t _count;
};

} // namespace kaminpar
