#include <gmock/gmock.h>

#include "kaminpar-common/varint_codec.h"

using namespace kaminpar;

template <
    typename Int,
    bool marked_codec = false,
    bool marked = false,
    typename LengthFun,
    typename Encoder,
    typename Decoder>
void test_varint_codec(
    const std::vector<Int> &values, LengthFun &&length, Encoder &&encode, Decoder &&decode
) {
  std::size_t total_len = 0;
  std::vector<std::size_t> values_len;
  for (const Int value : values) {
    const std::size_t value_len = length(value);
    values_len.push_back(value_len);
    total_len += value_len;
  }

  auto ptr = std::make_unique<std::uint8_t[]>(total_len);

  std::size_t i = 0;
  std::uint8_t *encoded_ptr = ptr.get();
  for (const Int value : values) {
    const std::size_t value_len = encode(value, encoded_ptr);
    encoded_ptr += value_len;

    EXPECT_EQ(values_len[i++], value_len);
  }

  i = 0;
  const std::uint8_t *decoded_ptr = ptr.get();
  for (const Int value : values) {
    if constexpr (marked_codec) {
      const auto [decoded_value, marker_set, value_len] = decode(decoded_ptr);
      decoded_ptr += value_len;

      EXPECT_EQ(values_len[i++], value_len);
      EXPECT_EQ(value, decoded_value);
      EXPECT_EQ(marked, marker_set);
    } else {
      const auto [decoded_value, value_len] = decode(decoded_ptr);
      decoded_ptr += value_len;

      EXPECT_EQ(values_len[i++], value_len);
      EXPECT_EQ(value, decoded_value);
    }
  }
}

template <typename Int> std::vector<Int> generate_values() {
  std::vector<Int> values;

  values.push_back(static_cast<Int>(0));
  for (std::size_t i = 1; i < sizeof(Int) + 1; ++i) {
    values.push_back((static_cast<Int>(1) << (i * 7)) - 1);
    values.push_back(static_cast<Int>(1) << (i * 7));
  }
  values.push_back(std::numeric_limits<Int>::max());

  return values;
}

template <typename Int> std::vector<Int> generate_signed_values() {
  std::vector<Int> values;

  values.push_back(static_cast<Int>(0));

  values.push_back((static_cast<Int>(1) << 6) - 1);
  values.push_back((static_cast<Int>(1) << 6));

  values.push_back(-(static_cast<Int>(1) << 6) + 1);
  values.push_back(-(static_cast<Int>(1) << 6));

  for (std::size_t i = 1; i < sizeof(Int); ++i) {
    values.push_back((static_cast<Int>(1) << (i * 7 + 6)) - 1);
    values.push_back(static_cast<Int>(1) << (i * 7 + 6));
    values.push_back(-(static_cast<Int>(1) << (i * 7 + 6)) + 1);
    values.push_back(-static_cast<Int>(1) << (i * 7 + 6));
  }
  values.push_back(std::numeric_limits<Int>::max());
  values.push_back(-std::numeric_limits<Int>::max());

  return values;
}

template <typename Int, bool marked = false> void test_varint_codec() {
  if constexpr (marked) {
    std::vector<Int> values = generate_values<Int>();

    test_varint_codec<Int, true, false>(
        values,
        [](const Int value) { return marked_varint_length<Int>(value); },
        [](const Int value, std::uint8_t *ptr) {
          return marked_varint_encode<Int>(value, false, ptr);
        },
        [](const std::uint8_t *ptr) { return marked_varint_decode<Int>(ptr); }
    );

    test_varint_codec<Int, true, true>(
        values,
        [](const Int value) { return marked_varint_length<Int>(value); },
        [](const Int value, std::uint8_t *ptr) {
          return marked_varint_encode<Int>(value, true, ptr);
        },
        [](const std::uint8_t *ptr) { return marked_varint_decode<Int>(ptr); }
    );
  } else if constexpr (std::numeric_limits<Int>::is_signed) {
    test_varint_codec<Int>(
        generate_signed_values<Int>(),
        [](const Int value) { return signed_varint_length<Int>(value); },
        [](const Int value, std::uint8_t *ptr) { return signed_varint_encode<Int>(value, ptr); },
        [](const std::uint8_t *ptr) { return signed_varint_decode<Int>(ptr); }
    );
  } else {
    std::vector<Int> values = generate_values<Int>();

    test_varint_codec<Int>(
        values,
        [](const Int value) { return varint_length<Int>(value); },
        [](const Int value, std::uint8_t *ptr) { return varint_encode<Int>(value, ptr); },
        [](const std::uint8_t *ptr) { return varint_decode_general<Int>(ptr); }
    );

    test_varint_codec<Int>(
        values,
        [](const Int value) { return varint_length<Int>(value); },
        [](const Int value, std::uint8_t *ptr) { return varint_encode<Int>(value, ptr); },
        [](const std::uint8_t *ptr) { return varint_decode<Int>(ptr); }
    );
  }
}

TEST(VarIntCodecTest, varint_codec) {
  test_varint_codec<std::uint32_t>();
  test_varint_codec<std::uint64_t>();
}

TEST(VarIntCodecTest, signed_varint_codec) {
  test_varint_codec<std::int32_t>();
  test_varint_codec<std::int64_t>();
}

TEST(VarIntCodecTest, marked_varint_codec) {
  test_varint_codec<std::uint32_t, true>();
  test_varint_codec<std::uint64_t, true>();
}
