#include <gmock/gmock.h>

#include "kaminpar-common/varint_stream_codec.h"

using namespace kaminpar;

template <typename Int> void test_varint_stream(const std::vector<Int> &values) {
  auto ptr = std::make_unique<std::uint8_t[]>(values.size() * sizeof(Int) + values.size());

  VarIntStreamEncoder<Int> encoder(ptr.get(), values.size());
  for (const Int value : values) {
    encoder.add(value);
  }
  encoder.flush();

  VarIntStreamDecoder<Int> decoder(ptr.get(), values.size());
  std::size_t i = 0;
  decoder.decode([&](const Int value) { EXPECT_EQ(values[i++], value); });
}

template <typename Int> void test_varint_stream() {
  std::vector<Int> values;

  for (std::size_t control_byte = 0; control_byte < 256; ++control_byte) {
    for (std::uint8_t i = 0; i < 4; ++i) {
      const std::uint8_t length = ((control_byte >> (2 * i)) & 0b11) + 1;
      const Int value = static_cast<Int>(1) << (length * 7);
      values.push_back(value);
    };
  }

  test_varint_stream(values);
}

TEST(VarIntStreamCodecTest, varint_stream) {
  test_varint_stream<std::uint32_t>();
}

template <typename Int> void test_varint_stream_remaining() {
  for (std::uint8_t i = 0; i < 3; ++i) {
    std::vector<Int> values;

    for (std::uint8_t j = 0; j <= i; ++j) {
      values.push_back(1);
    }

    test_varint_stream(values);
  };
}

TEST(VarIntStreamCodecTest, varint_stream_remaining) {
  test_varint_stream_remaining<std::uint32_t>();
}
