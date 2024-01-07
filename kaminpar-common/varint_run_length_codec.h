/*******************************************************************************
 * Encoding and decoding methods for run-length VarInts.
 *
 * @file:   varint_run_length_codec.h
 * @author: Daniel Salwasser
 * @date:   29.12.2023
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

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
    std::uint8_t size = needed_bytes(i);

    if (_buffer.empty()) {
      _buffered_size = size++;
    } else if (_buffer.size() == kBufferSize || _buffered_size != size) {
      flush();
      _buffered_size = size++;
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

    const std::uint8_t *begin = _ptr;
    if constexpr (sizeof(Int) == 4) {
      const std::uint8_t header = (static_cast<std::uint8_t>(_buffer.size() - 1) << 2) |
                                  ((_buffered_size - 1) & 0b00000011);
      *_ptr++ = header;
    } else if constexpr (sizeof(Int) == 8) {
      const std::uint8_t header = (static_cast<std::uint8_t>(_buffer.size() - 1) << 3) |
                                  ((_buffered_size - 1) & 0b00000111);
      *_ptr++ = header;
    }

    for (Int value : _buffer) {
      for (std::uint8_t i = 0; i < _buffered_size; ++i) {
        *_ptr++ = static_cast<std::uint8_t>(value);
        value >>= 8;
      }
    }

    _buffer.clear();
  }

private:
  std::uint8_t *_ptr;

  std::uint8_t _buffered_size;
  std::vector<Int> _buffer;

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
   * @param ptr The pointer to the memory location where the encoded integers are stored.
   */
  VarIntRunLengthDecoder(const std::uint8_t *ptr) : _ptr(ptr) {}

  /*!
   * Decodes the encoded integers.
   *
   * @param max_decoded The amount of integers to decode.
   * @param l The function to be called with the decoded integers, i.e. the function has one
   * parameter of type Int.
   */
  template <typename Lambda> void decode(const std::size_t max_decoded, Lambda &&l) {
    constexpr bool non_stoppable = std::is_void<std::invoke_result_t<Lambda, std::uint32_t>>::value;

    std::size_t decoded = 0;
    while (decoded < max_decoded) {
      const std::uint8_t run_header = *_ptr++;

      if constexpr (sizeof(Int) == 4) {
        std::uint8_t run_length = (run_header >> 2) + 1;
        const std::uint8_t run_size = (run_header & 0b00000011) + 1;

        decoded += run_length;
        if (decoded > max_decoded) {
          run_length -= decoded - max_decoded;
        }

        if constexpr (non_stoppable) {
          decode32(run_length, run_size, std::forward<Lambda>(l));
        } else {
          const bool stop = decode32(run_length, run_size, std::forward<Lambda>(l));
          if (stop) {
            return;
          }
        }
      } else if constexpr (sizeof(Int) == 8) {
        std::uint8_t run_length = (run_header >> 3) + 1;
        const std::uint8_t run_size = (run_header & 0b00000111) + 1;

        decoded += run_length;
        if (decoded > max_decoded) {
          run_length -= decoded - max_decoded;
        }

        if constexpr (non_stoppable) {
          decode64(run_length, run_size, std::forward<Lambda>(l));
        } else {
          const bool stop = decode64(run_length, run_size, std::forward<Lambda>(l));
          if (stop) {
            return;
          }
        }
      }
    }
  }

private:
  const std::uint8_t *_ptr;

  template <typename Lambda>
  bool decode32(const std::uint8_t run_length, const std::uint8_t run_size, Lambda &&l) {
    constexpr bool non_stoppable = std::is_void<std::invoke_result_t<Lambda, std::uint32_t>>::value;

    switch (run_size) {
    case 1:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint32_t value = static_cast<std::uint32_t>(*_ptr);
        _ptr += 1;

        if constexpr (non_stoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) {
            return true;
          }
        }
      }
      break;
    case 2:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint32_t value = *((std::uint16_t *)_ptr);
        _ptr += 2;

        if constexpr (non_stoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) {
            return true;
          }
        }
      }
      break;
    case 3:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint32_t value = *((std::uint32_t *)_ptr) & 0xFFFFFF;
        _ptr += 3;

        if constexpr (non_stoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) {
            return true;
          }
        }
      }
      break;
    case 4:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint32_t value = *((std::uint32_t *)_ptr);
        _ptr += 4;

        if constexpr (non_stoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) {
            return true;
          }
        }
      }
      break;
    default:
      throw std::runtime_error("unexpected run size");
    }

    return false;
  }

  template <typename Lambda>
  bool decode64(const std::uint8_t run_length, const std::uint8_t run_size, Lambda &&l) {
    constexpr bool non_stoppable = std::is_void<std::invoke_result_t<Lambda, std::uint64_t>>::value;

    switch (run_size) {
    case 1:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint64_t value = static_cast<std::uint64_t>(*_ptr);
        _ptr += 1;

        if constexpr (non_stoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) {
            return true;
          }
        }
      }
      break;
    case 2:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint64_t value = *((std::uint16_t *)_ptr);
        _ptr += 2;

        if constexpr (non_stoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) {
            return true;
          }
        }
      }
      break;
    case 3:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint64_t value = *((std::uint32_t *)_ptr) & 0xFFFFFF;
        _ptr += 3;

        if constexpr (non_stoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) {
            return true;
          }
        }
      }
      break;
    case 4:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint64_t value = *((std::uint32_t *)_ptr);
        _ptr += 4;

        if constexpr (non_stoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) {
            return true;
          }
        }
      }
      break;
    case 5:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint64_t value = *((std::uint64_t *)_ptr) & 0xFFFFFFFFFF;
        _ptr += 5;

        if constexpr (non_stoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) {
            return true;
          }
        }
      }
      break;
    case 6:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint64_t value = *((std::uint64_t *)_ptr) & 0xFFFFFFFFFFFF;
        _ptr += 6;

        if constexpr (non_stoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) {
            return true;
          }
        }
      }
      break;
    case 7:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint64_t value = *((std::uint64_t *)_ptr) & 0xFFFFFFFFFFFFFF;
        _ptr += 7;

        if constexpr (non_stoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) {
            return true;
          }
        }
      }
      break;
    case 8:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint64_t value = *((std::uint64_t *)_ptr);
        _ptr += 8;

        if constexpr (non_stoppable) {
          l(value);
        } else {
          const bool stop = l(value);
          if (stop) {
            return true;
          }
        }
      }
      break;
    default:
      throw std::runtime_error("unexpected run size");
    }

    return false;
  }
};

}; // namespace kaminpar
