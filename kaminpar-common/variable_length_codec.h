/*******************************************************************************
 * Variable length encoding and decoding methods for integers.
 *
 * @file:   variable_length_codec.h
 * @author: Daniel Salwasser
 * @date:   11.11.2023
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace kaminpar {

namespace debug {

static constexpr bool kTrackVarintStats = false;

struct VariabeLengthStats {
  std::size_t varint_count;
  std::size_t signed_varint_count;
  std::size_t marked_varint_count;

  std::size_t varint_bytes;
  std::size_t signed_varint_bytes;
  std::size_t marked_varint_bytes;
};

void varint_stats_reset();

VariabeLengthStats &varint_stats_global();

} // namespace debug

template <typename Int> static std::size_t varint_max_length() {
  return (sizeof(Int) * 8) / 7 + 1;
}

template <typename Int> static std::size_t varint_length(Int i) {
  std::size_t len = 1;

  while (i > 0b01111111) {
    i >>= 7;
    len++;
  }

  return len;
}

template <typename Int> static std::size_t signed_varint_length(Int i) {
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

template <typename Int> static std::size_t marked_varint_length(Int i) {
  std::size_t len = 1;

  i >>= 6;
  if (i > 0) {
    len += varint_length(i);
  }

  return len;
}

template <typename Int> static std::size_t varint_encode(Int i, std::uint8_t *ptr) {
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

template <typename Int> static std::size_t signed_varint_encode(Int i, std::uint8_t *ptr) {
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

    std::size_t len = varint_encode(i, ptr + 1) + 1;

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

template <typename Int>
static std::size_t marked_varint_encode(Int i, bool marker_set, std::uint8_t *ptr) {
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

    std::size_t len = varint_encode(i, ptr + 1) + 1;

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

template <typename Int> static std::pair<Int, std::size_t> varint_decode(const std::uint8_t *ptr) {
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
static std::pair<Int, std::size_t> signed_varint_decode(const std::uint8_t *ptr) {
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

template <typename Int>
static std::tuple<Int, bool, std::size_t> marked_varint_decode(const std::uint8_t *ptr) {
  std::uint8_t first_octet = *ptr++;
  Int value = first_octet & 0b00111111;
  bool is_marker_set = (first_octet & 0b01000000) != 0;
  bool is_continuation_bit_set = (first_octet & 0b10000000) != 0;

  std::size_t len = 0;
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

  return {value, is_marker_set, len + 1};
}

template <typename Int> class RLEncoder {
  static_assert(sizeof(Int) == 4 || sizeof(Int) == 8);

public:
  static constexpr std::size_t kBufferSize = (sizeof(Int) == 4) ? 64 : 32;

  RLEncoder(std::uint8_t *ptr) : ptr(ptr) {}

  std::size_t add(Int i) {
    std::uint8_t size = needed_bytes(i);

    if (buffer.empty()) {
      buffered_size = size++;
    } else if (buffer.size() == kBufferSize || buffered_size != size) {
      flush();
      buffered_size = size++;
    }

    buffer.push_back(i);
    return size;
  }

  void flush() {
    const std::uint8_t *begin = ptr;

    if constexpr (sizeof(Int) == 4) {
      std::uint8_t header =
          (static_cast<std::uint8_t>(buffer.size() - 1) << 2) | ((buffered_size - 1) & 0b00000011);
      *ptr++ = header;
    } else if constexpr (sizeof(Int) == 8) {
      std::uint8_t header =
          (static_cast<std::uint8_t>(buffer.size() - 1) << 3) | ((buffered_size - 1) & 0b00000111);
      *ptr++ = header;
    }

    for (Int i : buffer) {
      for (std::uint8_t j = 0; j < buffered_size; ++j) {
        *ptr++ = static_cast<std::uint8_t>(i);
        i >>= 8;
      }
    }

    buffer.clear();
  }

private:
  std::uint8_t *ptr;

  std::uint8_t buffered_size;
  std::vector<Int> buffer;

  std::uint8_t needed_bytes(Int i) const {
    std::size_t len = 1;

    while (i > 0b11111111) {
      i >>= 8;
      len++;
    }

    return len;
  }
};

template <typename Int> class RLDecoder {
  static_assert(sizeof(Int) == 4 || sizeof(Int) == 8);

public:
  RLDecoder(const std::uint8_t *ptr) : ptr(ptr) {}

  template <typename Lambda> void decode(const std::uint8_t *end, Lambda &&l) {
    while (ptr < end) {
      const std::uint8_t run_header = *ptr++;

      if constexpr (sizeof(Int) == 4) {
        const std::uint8_t run_length = ((run_header & 0b11111100) >> 2) + 1;
        const std::uint8_t run_size = (run_header & 0b00000011) + 1;

        decode32(run_length, run_size, std::forward<Lambda>(l));
      } else if constexpr (sizeof(Int) == 8) {
        const std::uint8_t run_length = ((run_header & 0b11111000) >> 3) + 1;
        const std::uint8_t run_size = (run_header & 0b00000111) + 1;

        decode64(run_length, run_size, std::forward<Lambda>(l));
      }
    }
  }

  template <typename Lambda> void decode(const std::size_t max_decoded, Lambda &&l) {
    std::size_t decoded = 0;
    while (decoded < max_decoded) {
      const std::uint8_t run_header = *ptr++;

      if constexpr (sizeof(Int) == 4) {
        std::uint8_t run_length = (run_header >> 2) + 1;
        const std::uint8_t run_size = (run_header & 0b00000011) + 1;

        decoded += run_length;
        if (decoded > max_decoded) {
          run_length -= decoded - max_decoded;
        }

        decode32(run_length, run_size, std::forward<Lambda>(l));
      } else if constexpr (sizeof(Int) == 8) {
        std::uint8_t run_length = (run_header >> 3) + 1;
        const std::uint8_t run_size = (run_header & 0b00000111) + 1;

        decoded += run_length;
        if (decoded > max_decoded) {
          run_length -= decoded - max_decoded;
        }

        decode64(run_length, run_size, std::forward<Lambda>(l));
      }
    }
  }

private:
  const std::uint8_t *ptr;

  template <typename Lambda>
  void decode32(const std::uint8_t run_length, const std::uint8_t run_size, Lambda &&l) {
    switch (run_size) {
    case 1:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint32_t value = static_cast<std::uint32_t>(*ptr);
        ptr += 1;

        l(value);
      }
      break;
    case 2:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint32_t value = *((std::uint16_t *)ptr);
        ptr += 2;

        l(value);
      }
      break;
    case 3:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint32_t value = *((std::uint32_t *)ptr) & 0xFFFFFF;
        ptr += 3;

        l(value);
      }
      break;
    case 4:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint32_t value = *((std::uint32_t *)ptr);
        ptr += 4;

        l(value);
      }
      break;
    default:
      break;
    }
  }

  template <typename Lambda>
  void decode64(const std::uint8_t run_length, const std::uint8_t run_size, Lambda &&l) {
    switch (run_size) {
    case 1:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint64_t value = static_cast<std::uint64_t>(*ptr);
        ptr += 1;

        l(value);
      }
      break;
    case 2:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint64_t value = *((std::uint16_t *)ptr);
        ptr += 2;

        l(value);
      }
      break;
    case 3:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint64_t value = *((std::uint32_t *)ptr) & 0xFFFFFF;
        ptr += 3;

        l(value);
      }
      break;
    case 4:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint64_t value = *((std::uint32_t *)ptr);
        ptr += 4;

        l(value);
      }
      break;
    case 5:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint64_t value = *((std::uint64_t *)ptr) & 0xFFFFFFFFFF;
        ptr += 5;

        l(value);
      }
      break;
    case 6:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint64_t value = *((std::uint64_t *)ptr) & 0xFFFFFFFFFFFF;
        ptr += 6;

        l(value);
      }
      break;
    case 7:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint64_t value = *((std::uint64_t *)ptr) & 0xFFFFFFFFFFFFFF;
        ptr += 7;

        l(value);
      }
      break;
    case 8:
      for (std::uint8_t i = 0; i < run_length; ++i) {
        std::uint64_t value = *((std::uint64_t *)ptr);
        ptr += 8;

        l(value);
      }
      break;
    default:
      break;
    }
  }
};

} // namespace kaminpar
