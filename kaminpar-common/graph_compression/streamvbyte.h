/*******************************************************************************
 * Endoder and decoder for StreamVByte.
 *
 * @file:   streamvbyte.h
 * @author: Daniel Salwasser
 * @date:   29.12.2023
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>

#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "kaminpar-common/constexpr_utils.h"
#include "kaminpar-common/math.h"

namespace kaminpar::streamvbyte {

enum class DifferentialCodingKind {
  NONE,
  D1,
  D2,
  DM,
  D4,
};

template <std::integral Int, DifferentialCodingKind GapKind = DifferentialCodingKind::NONE>
class StreamVByteEncoder {
  static constexpr std::size_t kIntByteWidth = sizeof(Int);
  static_assert(
      kIntByteWidth == 4 || kIntByteWidth == 8,
      "StreamVByte only supports 32-bit or 64-bit integers."
  );

  [[nodiscard]] static std::size_t required_byte_width(const Int value) {
    if constexpr (kIntByteWidth == 4) {
      return math::byte_width(value);
    } else if constexpr (kIntByteWidth == 8) {
      switch (math::byte_width(value)) {
      case 1:
        return 1;
      case 2:
        return 2;
      case 3:
        [[fallthrough]];
      case 4:
        return 4;
      case 5:
        [[fallthrough]];
      case 6:
        [[fallthrough]];
      case 7:
        [[fallthrough]];
      case 8:
        return 8;
      default:
        __builtin_unreachable();
      }
    } else {
      // Impossible case due to static_assert above
    }
  }

  [[nodiscard]] static std::uint8_t encoded_byte_width(const Int value) {
    if constexpr (kIntByteWidth == 4) {
      return required_byte_width(value) - 1;
    } else if constexpr (kIntByteWidth == 8) {
      switch (required_byte_width(value)) {
      case 1:
        return 0;
      case 2:
        return 1;
      case 4:
        return 2;
      case 8:
        return 3;
      default:
        __builtin_unreachable();
      }
    } else {
      // Impossible case due to static_assert above
    }
  }

public:
  explicit StreamVByteEncoder(const std::size_t num_values, std::uint8_t *ptr)
      : _num_values(num_values),
        _control_bytes_ptr(ptr),
        _data_ptr(ptr + math::div_ceil(num_values, 4)),
        _num_buffered(0),
        _prev_value(0),
        _prev2_value(0),
        _prev3_value(0),
        _prev4_value(0),
        _prev_max_value(0),
        _next_max_value(0) {
    std::fill(std::begin(_buffer), std::end(_buffer), 0);
  }

  std::size_t add(Int value) {
    if constexpr (GapKind == DifferentialCodingKind::D1) {
      const Int next_prev_value = value;
      value = value - _prev_value;

      _prev_value = next_prev_value;
    } else if constexpr (GapKind == DifferentialCodingKind::D2) {
      const Int next_prev_value = value;
      value = value - _prev2_value;

      _prev2_value = _prev_value;
      _prev_value = next_prev_value;
    } else if constexpr (GapKind == DifferentialCodingKind::DM) {
      _next_max_value = std::max(_prev_max_value, value);
      value = value - _prev_max_value;
    } else if constexpr (GapKind == DifferentialCodingKind::D4) {
      const Int next_prev_value = value;
      value = value - _prev4_value;

      _prev4_value = _prev3_value;
      _prev3_value = _prev2_value;
      _prev2_value = _prev_value;
      _prev_value = next_prev_value;
    }

    _buffer[_num_buffered] = value;

    if (_num_buffered == 3) {
      if constexpr (GapKind == DifferentialCodingKind::DM) {
        _prev_max_value = _next_max_value;
        _next_max_value = 0;
      }

      unchecked_flush();
      return required_byte_width(value);
    }

    const bool first_element = _num_buffered++ == 0;
    return required_byte_width(value) + (first_element ? 1 : 0);
  }

  std::uint8_t *flush() {
    if (_num_buffered > 0) [[likely]] {
      unchecked_flush(_num_buffered);
    }

    return _data_ptr;
  }

private:
  std::size_t _num_values;
  std::uint8_t *_control_bytes_ptr;
  std::uint8_t *_data_ptr;

  std::size_t _num_buffered;
  std::array<Int, 4> _buffer;

  Int _prev_value;
  Int _prev2_value;
  Int _prev3_value;
  Int _prev4_value;

  Int _prev_max_value;
  Int _next_max_value;

private:
  void unchecked_flush(const std::size_t num_values = 4) {
    const std::uint8_t control_byte =
        (encoded_byte_width(_buffer[3]) << 6) | (encoded_byte_width(_buffer[2]) << 4) |
        (encoded_byte_width(_buffer[1]) << 2) | encoded_byte_width(_buffer[0]);
    *_control_bytes_ptr++ = control_byte;

    for (std::size_t i = 0; i < num_values; ++i) {
      Int value = _buffer[i];

      do {
        *_data_ptr++ = static_cast<std::uint8_t>(value);
        value >>= 8;
      } while (value > 0);

      if constexpr (kIntByteWidth == 8) {
        std::size_t num_padding_bytes = required_byte_width(value) - math::byte_width(value);
        while (num_padding_bytes > 0) {
          *_data_ptr++ = static_cast<std::uint8_t>(0);
          num_padding_bytes -= 1;
        }
      }
    }

    _num_buffered = 0;
    std::fill(std::begin(_buffer), std::end(_buffer), 0);
  }
};

template <
    std::integral Int,
    bool PassPairs = false,
    DifferentialCodingKind GapKind = DifferentialCodingKind::NONE>
class StreamVByteDecoder {
  static constexpr std::size_t kIntByteWidth = sizeof(Int);
  static_assert(
      kIntByteWidth == 4 || kIntByteWidth == 8,
      "StreamVByte only supports 32-bit or 64-bit integers."
  );

  static constexpr bool k32BitInts = kIntByteWidth == 4;
  using LengthTable =
      std::conditional_t<k32BitInts, std::array<std::uint8_t, 256>, std::array<std::uint8_t, 16>>;
  using ShuffleTable = std::conditional_t<
      k32BitInts,
      std::array<std::array<std::uint8_t, 16>, 256>,
      std::array<std::array<std::uint8_t, 16>, 16>>;

  [[nodiscard]] static consteval LengthTable create_length_table() {
    LengthTable length_table{};

    if constexpr (k32BitInts) {
      constexpr_for<256>([&](const std::uint8_t control_byte) {
        length_table[control_byte] = 0;

        constexpr_for<4>([&](const std::uint8_t i) {
          const std::uint8_t header = (control_byte >> (2 * i)) & 0b11;
          const std::uint8_t length = header + 1;
          length_table[control_byte] += length;
        });
      });
    } else {
      const auto actual_length = [&](const std::uint8_t header) {
        switch (header) {
        case 0:
          return 1;
        case 1:
          return 2;
        case 2:
          return 4;
        case 3:
          return 8;
        default:
          __builtin_unreachable();
        }
      };

      constexpr_for<16>([&](const std::uint8_t control_byte) {
        length_table[control_byte] = 0;

        constexpr_for<2>([&](const std::uint8_t i) {
          const std::uint8_t header = (control_byte >> (2 * i)) & 0b11;
          const std::uint8_t length = actual_length(header);
          length_table[control_byte] += length;
        });
      });
    }

    return length_table;
  }

  [[nodiscard]] static consteval ShuffleTable create_shuffle_table() {
    ShuffleTable shuffle_table{};

    if constexpr (k32BitInts) {
      constexpr_for<256>([&](const std::uint8_t control_byte) {
        std::uint8_t byte = 0;
        std::uint8_t pos = 0;

        constexpr_for<4>([&](const std::uint8_t i) {
          const std::uint8_t header = (control_byte >> (2 * i)) & 0b11;
          const std::uint8_t length = header + 1;

          std::uint8_t j = 0;
          while (j < length) {
            shuffle_table[control_byte][pos++] = byte++;
            j += 1;
          }

          while (j < 4) {
            shuffle_table[control_byte][pos++] = 0b11111111;
            j += 1;
          }
        });
      });
    } else {
      const auto actual_length = [&](const std::uint8_t value) {
        switch (value) {
        case 0:
          return 1;
        case 1:
          return 2;
        case 2:
          return 4;
        case 3:
          return 8;
        default:
          __builtin_unreachable();
        }
      };

      constexpr_for<16>([&](const std::uint8_t control_byte) {
        std::uint8_t byte = 0;
        std::uint8_t pos = 0;

        constexpr_for<2>([&](const std::uint8_t i) {
          const std::uint8_t header = (control_byte >> (2 * i)) & 0b11;
          const std::uint8_t length = actual_length(header);

          std::uint8_t j = 0;
          while (j < length) {
            shuffle_table[control_byte][pos++] = byte++;
            j += 1;
          }

          while (j < 8) {
            shuffle_table[control_byte][pos++] = 0b11111111;
            j += 1;
          }
        });
      });
    }

    return shuffle_table;
  }

  static constexpr const LengthTable kLengthTable = create_length_table();
  static constexpr const ShuffleTable kShuffleTable = create_shuffle_table();

public:
  explicit StreamVByteDecoder(const std::size_t num_values, const std::uint8_t *ptr)
      : _num_control_bytes(num_values / 4),
        _control_bytes_ptr(ptr),
        _num_values(num_values),
        _data_ptr(ptr + _num_control_bytes + ((num_values % 4) != 0)) {}

  template <typename Lambda> void decode(Lambda &&l) {
    if constexpr (k32BitInts) {
      decode32(std::forward<Lambda>(l));
    } else {
      decode64(std::forward<Lambda>(l));
    }
  }

  [[nodiscard]] const std::uint8_t *get() {
    return _data_ptr;
  }

private:
#if defined(__x86_64__)
  template <typename Lambda> void decode32(Lambda &&l) {
    static_assert(
        std::is_invocable_v<Lambda, Int> || (PassPairs && std::is_invocable_v<Lambda, Int, Int>)
    );

    using LambdaReturnType = std::conditional_t<
        PassPairs,
        std::invoke_result<Lambda, Int, Int>,
        std::invoke_result<Lambda, Int>>::type;
    constexpr bool kNonStoppable = std::is_void_v<LambdaReturnType>;

    __m128i prev = _mm_setzero_si128();
    const auto decode_gaps = [&](__m128i data) {
      if constexpr (GapKind == DifferentialCodingKind::NONE) {
        prev = data;
        return;
      }

      if constexpr (GapKind == DifferentialCodingKind::D1) {
        const __m128i temp = _mm_add_epi32(_mm_slli_si128(data, 8), data);
        prev = _mm_add_epi32(
            _mm_add_epi32(_mm_slli_si128(temp, 4), temp), _mm_shuffle_epi32(prev, 0xff)
        );
      } else if constexpr (GapKind == DifferentialCodingKind::D2) {
        prev = _mm_add_epi32(
            _mm_add_epi32(_mm_slli_si128(data, 8), data),
            _mm_shuffle_epi32(prev, _MM_SHUFFLE(3, 2, 3, 2))
        );
      } else if constexpr (GapKind == DifferentialCodingKind::DM) {
        prev = _mm_add_epi32(data, _mm_shuffle_epi32(prev, 0xff));
      } else if constexpr (GapKind == DifferentialCodingKind::D4) {
        prev = _mm_add_epi32(data, prev);
      } else {
        // Impossible case due to static_assert above
      }
    };

    for (std::size_t i = 0; i < _num_control_bytes; ++i) {
      const std::uint8_t control_byte = _control_bytes_ptr[i];
      const std::uint8_t length = kLengthTable[control_byte];

      __m128i data = _mm_loadu_si128((const __m128i *)_data_ptr);
      _data_ptr += length;

      const std::uint8_t *shuffle_mask = kShuffleTable[control_byte].data();
      const __m128i mask = _mm_loadu_si128((const __m128i *)shuffle_mask);

      data = _mm_shuffle_epi8(data, mask);
      decode_gaps(data);

      if constexpr (kNonStoppable) {
        if constexpr (PassPairs) {
          l(_mm_extract_epi32(prev, 0), _mm_extract_epi32(prev, 1));
          l(_mm_extract_epi32(prev, 2), _mm_extract_epi32(prev, 3));
        } else {
          l(_mm_extract_epi32(prev, 0));
          l(_mm_extract_epi32(prev, 1));
          l(_mm_extract_epi32(prev, 2));
          l(_mm_extract_epi32(prev, 3));
        }
      } else {
        if constexpr (PassPairs) {
          if (l(_mm_extract_epi32(prev, 0), _mm_extract_epi32(prev, 1))) [[unlikely]] {
            return;
          }

          if (l(_mm_extract_epi32(prev, 2), _mm_extract_epi32(prev, 3))) [[unlikely]] {
            return;
          }
        } else {
          if (l(_mm_extract_epi32(prev, 0))) [[unlikely]] {
            return;
          }

          if (l(_mm_extract_epi32(prev, 1))) [[unlikely]] {
            return;
          }

          if (l(_mm_extract_epi32(prev, 2))) [[unlikely]] {
            return;
          }

          if (l(_mm_extract_epi32(prev, 3))) [[unlikely]] {
            return;
          }
        }
      }
    }

    if constexpr (PassPairs) {
      if (_num_values % 4 == 2) {
        const std::uint8_t control_byte = _control_bytes_ptr[_num_control_bytes];
        const std::uint8_t length = kLengthTable[control_byte];

        const std::uint8_t *shuffle_mask = kShuffleTable[control_byte].data();
        const __m128i mask = _mm_loadu_si128((const __m128i *)shuffle_mask);

        __m128i data = _mm_loadu_si128((const __m128i *)_data_ptr);
        _data_ptr += length - 2;

        data = _mm_shuffle_epi8(data, mask);
        decode_gaps(data);

        if constexpr (kNonStoppable) {
          l(_mm_extract_epi32(prev, 0), _mm_extract_epi32(prev, 1));
        } else {
          if (l(_mm_extract_epi32(prev, 0), _mm_extract_epi32(prev, 1))) [[unlikely]] {
            return;
          }
        }
      }
    } else {
      switch (_num_values % 4) {
      case 1: {
        const std::uint8_t control_byte = _control_bytes_ptr[_num_control_bytes];
        const std::uint8_t *shuffle_mask = kShuffleTable[control_byte].data();

        __m128i data = _mm_loadu_si128((const __m128i *)_data_ptr);
        const __m128i mask = _mm_loadu_si128((const __m128i *)shuffle_mask);

        data = _mm_shuffle_epi8(data, mask);
        decode_gaps(data);

        if constexpr (kNonStoppable) {
          l(_mm_extract_epi32(prev, 0));
        } else {
          if (l(_mm_extract_epi32(prev, 0))) [[unlikely]] {
            return;
          }
        }
        break;
      }
      case 2: {
        const std::uint8_t control_byte = _control_bytes_ptr[_num_control_bytes];
        const std::uint8_t *shuffle_mask = kShuffleTable[control_byte].data();

        __m128i data = _mm_loadu_si128((const __m128i *)_data_ptr);
        const __m128i mask = _mm_loadu_si128((const __m128i *)shuffle_mask);

        data = _mm_shuffle_epi8(data, mask);
        decode_gaps(data);

        if constexpr (kNonStoppable) {
          l(_mm_extract_epi32(prev, 0));
          l(_mm_extract_epi32(prev, 1));
        } else {
          if (l(_mm_extract_epi32(prev, 0))) [[unlikely]] {
            return;
          }

          if (l(_mm_extract_epi32(prev, 1))) [[unlikely]] {
            return;
          }
        }
        break;
      }
      case 3: {
        const std::uint8_t control_byte = _control_bytes_ptr[_num_control_bytes];
        const std::uint8_t *shuffle_mask = kShuffleTable[control_byte].data();

        __m128i data = _mm_loadu_si128((const __m128i *)_data_ptr);
        const __m128i mask = _mm_loadu_si128((const __m128i *)shuffle_mask);

        data = _mm_shuffle_epi8(data, mask);
        decode_gaps(data);

        if constexpr (kNonStoppable) {
          l(_mm_extract_epi32(prev, 0));
          l(_mm_extract_epi32(prev, 1));
          l(_mm_extract_epi32(prev, 2));
        } else {
          if (l(_mm_extract_epi32(prev, 0))) [[unlikely]] {
            return;
          }

          if (l(_mm_extract_epi32(prev, 1))) [[unlikely]] {
            return;
          }

          if (l(_mm_extract_epi32(prev, 2))) [[unlikely]] {
            return;
          }
        }
        break;
      }
      }
    }
  }
#elif defined(__aarch64__)
  template <typename Lambda>
  void decode32(Lambda &&l)
    requires(
        std::is_invocable_v<Lambda, Int> || (PassPairs && std::is_invocable_v<Lambda, Int, Int>)
    )
  {
    using LambdaReturnType = std::conditional_t<
        PassPairs,
        std::invoke_result<Lambda, Int, Int>,
        std::invoke_result<Lambda, Int>>::type;
    constexpr bool kNonStoppable = std::is_void_v<LambdaReturnType>;

    uint32x4_t prev = vdupq_n_u32(0);

    auto decode_values = [&](const std::uint8_t ctrl_byte) {
      const uint32x4_t data = vreinterpretq_u32_u8(
          vqtbl1q_u8(vld1q_u8(_data_ptr), vld1q_u8(kShuffleTable[ctrl_byte].data()))
      );

      if constexpr (GapKind == DifferentialCodingKind::D1) {
        const uint32x4_t zero = vdupq_n_u32(0);
        uint32x4_t shifted = vaddq_u32(vextq_u32(zero, data, 2), data);
        shifted = vaddq_u32(vextq_u32(zero, shifted, 3), shifted);
        prev = vaddq_u32(shifted, vdupq_laneq_u32(prev, 3));
      } else if constexpr (GapKind == DifferentialCodingKind::D2) {
        prev = vaddq_u32(
            vaddq_u32(vextq_u32(vdupq_n_u32(0), data, 2), data),
            vcombine_u32(vget_high_u32(prev), vget_high_u32(prev))
        );
      } else if constexpr (GapKind == DifferentialCodingKind::DM) {
        prev = vaddq_u32(data, vdupq_laneq_u32(prev, 3));
      } else if constexpr (GapKind == DifferentialCodingKind::D4) {
        prev = vaddq_u32(data, prev);
      } else { // DifferentialCodingKind::NONE
        prev = data;
      }

      std::array<std::uint32_t, 4> out;
      vst1q_u32(out.data(), prev);
      return std::make_tuple(out[0], out[1], out[2], out[3]);
    };

    for (std::size_t i = 0; i < _num_control_bytes; ++i) {
      const std::uint8_t ctrl_byte = _control_bytes_ptr[i];
      const auto [v0, v1, v2, v3] = decode_values(ctrl_byte);
      _data_ptr += kLengthTable[ctrl_byte];

      if constexpr (kNonStoppable) {
        if constexpr (PassPairs) {
          l(v0, v1);
          l(v2, v3);
        } else {
          l(v0);
          l(v1);
          l(v2);
          l(v3);
        }
      } else {
        if constexpr (PassPairs) {
          if (l(v0, v1)) [[unlikely]] {
            return;
          }

          if (l(v2, v3)) [[unlikely]] {
            return;
          }
        } else {
          if (l(v0)) [[unlikely]] {
            return;
          }

          if (l(v1)) [[unlikely]] {
            return;
          }

          if (l(v2)) [[unlikely]] {
            return;
          }

          if (l(v3)) [[unlikely]] {
            return;
          }
        }
      }
    }

    if constexpr (PassPairs) {
      if (_num_values % 4 == 2) {
        const std::uint8_t ctrl_byte = _control_bytes_ptr[_num_control_bytes];
        const auto [v0, v1, v2, v3] = decode_values(ctrl_byte);
        _data_ptr += kLengthTable[ctrl_byte] - 2;

        if constexpr (kNonStoppable) {
          l(v0, v1);
        } else {
          if (l(v0, v1)) [[unlikely]] {
            return;
          }
        }
      }
    } else {
      switch (_num_values % 4) {
      case 1: {
        const std::uint8_t ctrl_byte = _control_bytes_ptr[_num_control_bytes];
        const auto [v0, v1, v2, v3] = decode_values(ctrl_byte);

        if constexpr (kNonStoppable) {
          l(v0);
        } else {
          if (l(v0)) [[unlikely]] {
            return;
          }
        }
        break;
      }
      case 2: {
        const std::uint8_t ctrl_byte = _control_bytes_ptr[_num_control_bytes];
        const auto [v0, v1, v2, v3] = decode_values(ctrl_byte);

        if constexpr (kNonStoppable) {
          l(v0);
          l(v1);
        } else {
          if (l(v0)) [[unlikely]] {
            return;
          }

          if (l(v1)) [[unlikely]] {
            return;
          }
        }
        break;
      }
      case 3: {
        const std::uint8_t ctrl_byte = _control_bytes_ptr[_num_control_bytes];
        const auto [v0, v1, v2, v3] = decode_values(ctrl_byte);

        if constexpr (kNonStoppable) {
          l(v0);
          l(v1);
          l(v2);
        } else {
          if (l(v0)) [[unlikely]] {
            return;
          }

          if (l(v1)) [[unlikely]] {
            return;
          }

          if (l(v2)) [[unlikely]] {
            return;
          }
        }
        break;
      }
      }
    }
  }
#else
#error "Only x64 and ARM are supported"
#endif

#if defined(__x86_64__)
  template <typename Lambda> void decode64(Lambda &&l) {
    static_assert(std::is_invocable_v<Lambda, Int>);
    constexpr bool kNonStoppable = std::is_void_v<std::invoke_result_t<Lambda, Int>>;

    __m128i prev = _mm_setzero_si128();
    const auto decode_gaps = [&](__m128i data) {
      if constexpr (GapKind == DifferentialCodingKind::NONE) {
        prev = data;
        return;
      }

      if constexpr (GapKind == DifferentialCodingKind::D1) {
        const __m128i temp = _mm_add_epi64(_mm_slli_si128(data, 8), data);
        prev = _mm_add_epi32(
            _mm_add_epi32(_mm_slli_si128(temp, 4), temp), _mm_shuffle_epi32(prev, 0xff)
        );
      } else if constexpr (GapKind == DifferentialCodingKind::D2) {
        prev = _mm_add_epi32(
            _mm_add_epi32(_mm_slli_si128(data, 8), data),
            _mm_shuffle_epi32(prev, _MM_SHUFFLE(3, 2, 3, 2))
        );
      } else if constexpr (GapKind == DifferentialCodingKind::DM) {
        prev = _mm_add_epi32(data, _mm_shuffle_epi32(prev, 0xff));
      } else if constexpr (GapKind == DifferentialCodingKind::D4) {
        prev = _mm_add_epi32(data, prev);
      } else {
        // Impossible case due to static_assert above
      }
    };

    for (std::size_t i = 0; i < _num_control_bytes; ++i) {
      const std::uint8_t control_byte = _control_bytes_ptr[i];
      const std::uint8_t control_byte_lh = control_byte & 0b1111;
      const std::uint8_t control_byte_uh = control_byte >> 4;

      const std::uint8_t length1 = kLengthTable[control_byte_lh];
      const std::uint8_t length2 = kLengthTable[control_byte_uh];

      __m128i data1 = _mm_loadu_si128((const __m128i *)_data_ptr);
      _data_ptr += length1;

      __m128i data2 = _mm_loadu_si128((const __m128i *)_data_ptr);
      _data_ptr += length2;

      const std::uint8_t *shuffle_mask1 = kShuffleTable[control_byte_lh].data();
      const __m128i mask1 = _mm_loadu_si128((const __m128i *)shuffle_mask1);

      const std::uint8_t *shuffle_mask2 = kShuffleTable[control_byte_uh].data();
      const __m128i mask2 = _mm_loadu_si128((const __m128i *)shuffle_mask2);

      data1 = _mm_shuffle_epi8(data1, mask1);
      data2 = _mm_shuffle_epi8(data2, mask2);

      if constexpr (GapKind == DifferentialCodingKind::NONE) {
        if constexpr (kNonStoppable) {
          l(_mm_extract_epi64(data1, 0));
          l(_mm_extract_epi64(data1, 1));
          l(_mm_extract_epi64(data2, 0));
          l(_mm_extract_epi64(data2, 1));
        } else {
          if (l(_mm_extract_epi64(data1, 0))) [[unlikely]] {
            return;
          }

          if (l(_mm_extract_epi64(data1, 1))) [[unlikely]] {
            return;
          }

          if (l(_mm_extract_epi64(data2, 0))) [[unlikely]] {
            return;
          }

          if (l(_mm_extract_epi64(data2, 1))) [[unlikely]] {
            return;
          }
        }
      } else {
        decode_gaps(data1);
      }
    }

    switch (_num_values % 4) {
    case 1: {
      const std::uint8_t control_byte = _control_bytes_ptr[_num_control_bytes];
      const std::uint8_t control_byte_lh = control_byte & 0b1111;

      const std::uint8_t *shuffle_mask = kShuffleTable[control_byte_lh].data();
      const __m128i mask = _mm_loadu_si128((const __m128i *)shuffle_mask);

      __m128i data = _mm_loadu_si128((const __m128i *)_data_ptr);
      data = _mm_shuffle_epi8(data, mask);

      if constexpr (kNonStoppable) {
        l(_mm_extract_epi64(data, 0));
      } else {
        if (l(_mm_extract_epi64(data, 0))) [[unlikely]] {
          return;
        }
      }
      break;
    }
    case 2: {
      const std::uint8_t control_byte = _control_bytes_ptr[_num_control_bytes];
      const std::uint8_t control_byte_lh = control_byte & 0b1111;

      const std::uint8_t *shuffle_mask = kShuffleTable[control_byte_lh].data();
      const __m128i mask = _mm_loadu_si128((const __m128i *)shuffle_mask);

      __m128i data = _mm_loadu_si128((const __m128i *)_data_ptr);
      data = _mm_shuffle_epi8(data, mask);

      if constexpr (kNonStoppable) {
        l(_mm_extract_epi64(data, 0));
        l(_mm_extract_epi64(data, 1));
      } else {
        if (l(_mm_extract_epi64(data, 0))) [[unlikely]] {
          return;
        }

        if (l(_mm_extract_epi64(data, 1))) [[unlikely]] {
          return;
        }
      }
      break;
    }
    case 3: {
      const std::uint8_t control_byte = _control_bytes_ptr[_num_control_bytes];
      const std::uint8_t control_byte_lh = control_byte & 0b1111;
      const std::uint8_t control_byte_uh = control_byte >> 4;

      const std::uint8_t length1 = kLengthTable[control_byte_lh];
      __m128i data1 = _mm_loadu_si128((const __m128i *)_data_ptr);

      _data_ptr += length1;
      __m128i data2 = _mm_loadu_si128((const __m128i *)_data_ptr);

      const std::uint8_t *shuffle_mask1 = kShuffleTable[control_byte_lh].data();
      const __m128i mask1 = _mm_loadu_si128((const __m128i *)shuffle_mask1);

      const std::uint8_t *shuffle_mask2 = kShuffleTable[control_byte_uh].data();
      const __m128i mask2 = _mm_loadu_si128((const __m128i *)shuffle_mask2);

      data1 = _mm_shuffle_epi8(data1, mask1);
      data2 = _mm_shuffle_epi8(data2, mask2);

      if constexpr (kNonStoppable) {
        l(_mm_extract_epi64(data1, 0));
        l(_mm_extract_epi64(data1, 1));
        l(_mm_extract_epi64(data2, 0));
      } else {
        if (l(_mm_extract_epi64(data1, 0))) [[unlikely]] {
          return;
        }

        if (l(_mm_extract_epi64(data1, 1))) [[unlikely]] {
          return;
        }

        if (l(_mm_extract_epi64(data2, 0))) [[unlikely]] {
          return;
        }
      }
      break;
    }
    }
  }
#elif defined(__aarch64__)
  template <typename Lambda>
  void decode64(Lambda &&)
    requires(false)
  {
    // @todo implement decode for ARM
  }
#endif

private:
  const std::size_t _num_control_bytes;
  const std::uint8_t *_control_bytes_ptr;

  const std::size_t _num_values;
  const std::uint8_t *_data_ptr;
};

} // namespace kaminpar::streamvbyte
