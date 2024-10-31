#include <gmock/gmock.h>

#include "kaminpar-common/graph_compression/varint.h"

namespace {
using namespace kaminpar;

template <typename Int> [[nodiscard]] std::vector<Int> generate_values() {
  std::vector<Int> values;

  values.push_back(static_cast<Int>(0));

  for (std::size_t i = 1; i < sizeof(Int) + 1; ++i) {
    values.push_back((static_cast<Int>(1) << (i * 7)) - 1);
    values.push_back(static_cast<Int>(1) << (i * 7));
  }

  values.push_back(std::numeric_limits<Int>::max());

  return values;
}

template <typename Int> [[nodiscard]] std::vector<Int> generate_signed_values() {
  std::vector<Int> values;

  values.push_back(static_cast<Int>(0));

  for (std::size_t i = 0; i < sizeof(Int); ++i) {
    values.push_back((static_cast<Int>(1) << (i * 7 + 6)) - 1);
    values.push_back(static_cast<Int>(1) << (i * 7 + 6));
    values.push_back(-(static_cast<Int>(1) << (i * 7 + 6)) + 1);
    values.push_back(-static_cast<Int>(1) << (i * 7 + 6));
  }

  values.push_back(std::numeric_limits<Int>::max());
  values.push_back(std::numeric_limits<Int>::min());

  return values;
}

template <
    typename Int,
    typename LengthFun,
    typename Encoder,
    typename Decoder,
    typename DecoderAndIncrementer>
void test_varint_codec(
    const auto &values,
    LengthFun &&length,
    Encoder &&encode,
    Decoder &&decode,
    DecoderAndIncrementer &&decode_and_increment
) {
  std::size_t total_length = 0;
  std::vector<std::size_t> value_lengths;
  for (const Int value : values) {
    const std::size_t value_len = length(value);
    value_lengths.push_back(value_len);
    total_length += value_len;
  }

  auto ptr = std::make_unique<std::uint8_t[]>(total_length);

  std::uint8_t *encode_ptr = ptr.get();
  for (std::size_t i = 0; i < values.size(); ++i) {
    const std::size_t value_len = encode(values[i], encode_ptr);
    EXPECT_EQ(value_lengths[i], value_len);

    encode_ptr += value_len;
  }

  const std::uint8_t *decode_ptr = ptr.get();
  for (std::size_t i = 0; i < values.size(); ++i) {
    const std::uint8_t *decode_start_ptr = decode_ptr;

    const Int decoded_value1 = decode(decode_ptr);
    EXPECT_EQ(values[i], decoded_value1);

    const Int decoded_value2 = decode_and_increment(&decode_ptr);
    EXPECT_EQ(values[i], decoded_value2);

    const std::size_t value_len = static_cast<std::size_t>(decode_ptr - decode_start_ptr);
    EXPECT_EQ(value_lengths[i], value_len);
  }
}

TEST(VarIntTest, Codec) {
  const auto values = generate_values<std::uint32_t>();

  test_varint_codec<std::uint32_t>(
      values,
      [](const std::uint32_t i) { return varint_length(i); },
      [](const std::uint32_t i, std::uint8_t *ptr) { return varint_encode(i, ptr); },
      [](const std::uint8_t *ptr) { return varint_decode<std::uint32_t>(ptr); },
      [](const std::uint8_t **ptr) { return varint_decode<std::uint32_t>(ptr); }
  );
}

TEST(SignedVarIntTest, Codec) {
  const auto values = generate_signed_values<std::int32_t>();

  test_varint_codec<std::int32_t>(
      values,
      [](const std::int32_t i) { return signed_varint_length(i); },
      [](const std::int32_t i, std::uint8_t *ptr) { return signed_varint_encode(i, ptr); },
      [](const std::uint8_t *ptr) { return signed_varint_decode<std::int32_t>(ptr); },
      [](const std::uint8_t **ptr) { return signed_varint_decode<std::int32_t>(ptr); }
  );
}

} // namespace
