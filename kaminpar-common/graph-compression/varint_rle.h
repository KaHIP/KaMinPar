/*******************************************************************************
 * Encoding and decoding methods for run-length VarInts.
 *
 * @file:   varint_rle.h
 * @author: Daniel Salwasser
 * @date:   29.12.2023
 ******************************************************************************/
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "kaminpar-common/math.h"

namespace kaminpar {

/*!
 * An encoder for writing run-length VarInts.
 *
 * @tparam Int The type of integer to encode.
 */
template <typename Int> class VarIntRunLengthEncoder {
  static_assert(sizeof(Int) == 4 || sizeof(Int) == 8);

public:
  static constexpr std::size_t kBufferSize = (sizeof(Int) == 4) ? 64 : 32;

  /*!
   * Constructs a new VarIntRunLengthEncoder.
   *
   * @param ptr The pointer to the memory location where the encoded integers are written.
   */
  VarIntRunLengthEncoder(std::uint8_t *ptr) : _ptr(ptr) {}

  /*!
   * Encodes an integer.
   *
   * @param i The integer to encode.
   * @return The number of bytes that the integer requires to be stored in encoded format. It
   * includes the control byte if it is the first integer of a block.
   */
  std::size_t add(Int i) {
    std::uint8_t size = math::byte_width(i);

    if (_buffer.empty()) {
      _num_buffered = size++;
    } else if (_buffer.size() == kBufferSize || _num_buffered != size) {
      flush();
      _num_buffered = size++;
    }

    _buffer.push_back(i);
    return size;
  }

  /*!
   * Writes the remaining integers added to the encoder which do not form a complete block to
   * memory.
   */
  void flush() {
    if (_buffer.empty()) {
      return;
    }

    if constexpr (sizeof(Int) == 4) {
      const std::uint8_t header =
          (static_cast<std::uint8_t>(_buffer.size() - 1) << 2) | ((_num_buffered - 1) & 0b00000011);
      *_ptr++ = header;
    } else if constexpr (sizeof(Int) == 8) {
      const std::uint8_t header =
          (static_cast<std::uint8_t>(_buffer.size() - 1) << 3) | ((_num_buffered - 1) & 0b00000111);
      *_ptr++ = header;
    }

    for (Int value : _buffer) {
      for (std::uint8_t i = 0; i < _num_buffered; ++i) {
        *_ptr++ = static_cast<std::uint8_t>(value);
        value >>= 8;
      }
    }

    _buffer.clear();
  }

private:
  std::uint8_t *_ptr;

  std::uint8_t _num_buffered = 0;
  std::vector<Int> _buffer;
};

/*!
 * A decoder for reading run-length VarInts.
 *
 * @tparam Int The type of integer to decode.
 */
template <typename Int> class VarIntRunLengthDecoder {
  static_assert(sizeof(Int) == 4 || sizeof(Int) == 8);

public:
  /*!
   * Constructs a new VarIntRunLengthDecoder.
   *
   * @param num_values The number of integers that are encoded.
   * @param ptr The pointer to the memory location where the encoded integers are stored.
   */
  VarIntRunLengthDecoder(const std::size_t num_values, const std::uint8_t *ptr)
      : _num_values(num_values),
        _ptr(ptr) {}

  /*!
   * Decodes the encoded integers.
   *
   * @param l The function to be called with the decoded integers, i.e. the function has one
   * parameter of type Int.
   */
  template <typename Lambda> void decode(Lambda &&l) {
    constexpr bool kNonStoppable = std::is_void_v<std::invoke_result_t<Lambda, Int>>;

    std::size_t num_decoded = 0;
    while (num_decoded < _num_values) {
      const std::uint8_t run_header = *_ptr++;

      if constexpr (sizeof(Int) == 4) {
        const std::size_t run_length = (run_header >> 2) + 1;
        const std::size_t run_size = (run_header & 0b00000011) + 1;

        num_decoded += run_length;

        if constexpr (kNonStoppable) {
          decode32(run_length, run_size, std::forward<Lambda>(l));
        } else {
          const bool stop = decode32(run_length, run_size, std::forward<Lambda>(l));
          if (stop) [[unlikely]] {
            return;
          }
        }
      } else if constexpr (sizeof(Int) == 8) {
        const std::size_t run_length = (run_header >> 3) + 1;
        const std::size_t run_size = (run_header & 0b00000111) + 1;

        num_decoded += run_length;

        if constexpr (kNonStoppable) {
          decode64(run_length, run_size, std::forward<Lambda>(l));
        } else {
          const bool stop = decode64(run_length, run_size, std::forward<Lambda>(l));
          if (stop) [[unlikely]] {
            return;
          }
        }
      }
    }
  }

private:
  template <typename Lambda>
  bool decode32(const std::size_t run_length, const std::size_t run_size, Lambda &&l) {
    constexpr bool kNonStoppable = std::is_void_v<std::invoke_result_t<Lambda, std::uint32_t>>;

    switch (run_size) {
    case 1:
      for (std::size_t i = 0; i < run_length; ++i) {
        const std::uint32_t value = static_cast<std::uint32_t>(*_ptr);
        _ptr += 1;

        if constexpr (kNonStoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) [[unlikely]] {
            return true;
          }
        }
      }
      break;
    case 2:
      for (std::size_t i = 0; i < run_length; ++i) {
        const std::uint32_t value = *reinterpret_cast<const std::uint16_t *>(_ptr);
        _ptr += 2;

        if constexpr (kNonStoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) [[unlikely]] {
            return true;
          }
        }
      }
      break;
    case 3:
      for (std::size_t i = 0; i < run_length; ++i) {
        const std::uint32_t value = *reinterpret_cast<const std::uint32_t *>(_ptr) & 0xFFFFFF;
        _ptr += 3;

        if constexpr (kNonStoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) [[unlikely]] {
            return true;
          }
        }
      }
      break;
    case 4:
      for (std::size_t i = 0; i < run_length; ++i) {
        const std::uint32_t value = *reinterpret_cast<const std::uint32_t *>(_ptr);
        _ptr += 4;

        if constexpr (kNonStoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) [[unlikely]] {
            return true;
          }
        }
      }
      break;
    default:
      __builtin_unreachable();
    }

    return false;
  }

  template <typename Lambda>
  bool decode64(const std::size_t run_length, const std::size_t run_size, Lambda &&l) {
    constexpr bool kNonStoppable = std::is_void_v<std::invoke_result_t<Lambda, std::uint64_t>>;

    switch (run_size) {
    case 1:
      for (std::size_t i = 0; i < run_length; ++i) {
        const std::uint64_t value = static_cast<std::uint64_t>(*_ptr);
        _ptr += 1;

        if constexpr (kNonStoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) [[unlikely]] {
            return true;
          }
        }
      }
      break;
    case 2:
      for (std::size_t i = 0; i < run_length; ++i) {
        const std::uint64_t value = *reinterpret_cast<const std::uint16_t *>(_ptr);
        _ptr += 2;

        if constexpr (kNonStoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) [[unlikely]] {
            return true;
          }
        }
      }
      break;
    case 3:
      for (std::size_t i = 0; i < run_length; ++i) {
        const std::uint64_t value = *reinterpret_cast<const std::uint32_t *>(_ptr) & 0xFFFFFF;
        _ptr += 3;

        if constexpr (kNonStoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) [[unlikely]] {
            return true;
          }
        }
      }
      break;
    case 4:
      for (std::size_t i = 0; i < run_length; ++i) {
        const std::uint64_t value = *reinterpret_cast<const std::uint32_t *>(_ptr);
        _ptr += 4;

        if constexpr (kNonStoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) [[unlikely]] {
            return true;
          }
        }
      }
      break;
    case 5:
      for (std::size_t i = 0; i < run_length; ++i) {
        const std::uint64_t value = *reinterpret_cast<const std::uint64_t *>(_ptr) & 0xFFFFFFFFFF;
        _ptr += 5;

        if constexpr (kNonStoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) [[unlikely]] {
            return true;
          }
        }
      }
      break;
    case 6:
      for (std::size_t i = 0; i < run_length; ++i) {
        const std::uint64_t value = *reinterpret_cast<const std::uint64_t *>(_ptr) & 0xFFFFFFFFFFFF;
        _ptr += 6;

        if constexpr (kNonStoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) [[unlikely]] {
            return true;
          }
        }
      }
      break;
    case 7:
      for (std::size_t i = 0; i < run_length; ++i) {
        const std::uint64_t value =
            *reinterpret_cast<const std::uint64_t *>(_ptr) & 0xFFFFFFFFFFFFFF;
        _ptr += 7;

        if constexpr (kNonStoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) [[unlikely]] {
            return true;
          }
        }
      }
      break;
    case 8:
      for (std::size_t i = 0; i < run_length; ++i) {
        const std::uint64_t value = *reinterpret_cast<const std::uint64_t *>(_ptr);
        _ptr += 8;

        if constexpr (kNonStoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) [[unlikely]] {
            return true;
          }
        }
      }
      break;
    default:
      __builtin_unreachable();
    }

    return false;
  }

private:
  const std::size_t _num_values;
  const std::uint8_t *_ptr;
};

} // namespace kaminpar
