#include <gmock/gmock.h>

#include "kaminpar-common/graph_compression/streamvbyte.h"

namespace {
using namespace kaminpar::streamvbyte;

template <typename Int> [[nodiscard]] Int generate_value(const std::size_t byte_width = 1) {
  return static_cast<Int>(1) << (byte_width * 7);
}

template <typename Int> [[nodiscard]] std::vector<Int> generate_values() {
  std::vector<Int> values;

  for (std::size_t control_byte = 0; control_byte < 256; ++control_byte) {
    for (std::size_t i = 0; i < 4; ++i) {
      const std::uint8_t header = (control_byte >> (2 * i)) & 0b11;

      std::uint8_t length;
      if constexpr (sizeof(Int) == 4) {
        length = header + 1;
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

        length = actual_length(header);
      }

      values.push_back(generate_value<Int>(length));
    };
  }

  return values;
}

template <typename Int> [[nodiscard]] std::vector<Int> generate_sorted_values() {
  std::vector<Int> values;

  for (std::size_t shift = 0; shift < sizeof(Int) * 8; ++shift) {
    values.push_back(static_cast<Int>(1) << shift);
  };

  return values;
}

template <typename Int, DifferentialCodingKind GapKind = DifferentialCodingKind::NONE>
void test_streamvbyte_codec(const std::vector<Int> &values) {
  auto ptr = std::make_unique<std::uint8_t[]>(values.size() * sizeof(Int) + values.size());

  {
    StreamVByteEncoder<Int, GapKind> encoder(values.size(), ptr.get());
    for (const Int value : values) {
      encoder.add(value);
    }
    encoder.flush();
  }

  std::size_t i = 0;
  {
    StreamVByteDecoder<Int, false, GapKind> decoder(values.size(), ptr.get());
    decoder.decode([&](const Int value) { EXPECT_EQ(values[i++], value); });
  }
  EXPECT_EQ(i, values.size());
}

template <typename Int> void test_streamvbyte_codec() {
  std::vector<Int> values = generate_values<Int>();

  for (std::size_t i = 0; i < 4; i++) {
    test_streamvbyte_codec(values);
    values.push_back(generate_value<Int>());
  }
}

template <typename Int, DifferentialCodingKind GapKind> void test_sorted_streamvbyte_codec() {
  std::vector<Int> values = generate_sorted_values<Int>();

  for (std::size_t i = 0; i < 4; i++) {
    test_streamvbyte_codec<Int, GapKind>(values);
    values.push_back(std::numeric_limits<Int>::max());
  }
}

TEST(StreamVByte32Test, Default) {
  test_streamvbyte_codec<std::uint32_t>();
}

TEST(StreamVByte32Test, GapKindD1) {
  test_sorted_streamvbyte_codec<std::uint32_t, DifferentialCodingKind::D1>();
}

TEST(StreamVByte32Test, GapKindD2) {
  test_sorted_streamvbyte_codec<std::uint32_t, DifferentialCodingKind::D2>();
}

TEST(StreamVByte32Test, GapKindD3) {
  test_sorted_streamvbyte_codec<std::uint32_t, DifferentialCodingKind::DM>();
}

TEST(StreamVByte32Test, GapKindD4) {
  test_sorted_streamvbyte_codec<std::uint32_t, DifferentialCodingKind::D4>();
}

#if defined(__x86_64__)
TEST(StreamVByte64Test, Default) {
  test_streamvbyte_codec<std::uint64_t>();
}
#endif

} // namespace
