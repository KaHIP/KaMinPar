#include <gmock/gmock.h>

#include "kaminpar-common/variable_length_codec.h"

using namespace kaminpar;

template <typename VarLenCodec, typename Int> static void test_varlen_codec(Int value) {
  std::size_t len = VarLenCodec::length(value);
  auto ptr = std::make_unique<std::uint8_t[]>(len);

  std::size_t encoded_value_len = VarLenCodec::encode(value, ptr.get());
  auto [decoded_value, decoded_value_len] = VarLenCodec::template decode<Int>(ptr.get());

  EXPECT_EQ(value, decoded_value);
  EXPECT_EQ(len, encoded_value_len);
  EXPECT_EQ(len, decoded_value_len);
}

template <typename VarLenCodec, typename Int> static void test_signed_varlen_codec(Int value) {
  std::size_t len = VarLenCodec::length_signed(value);
  auto ptr = std::make_unique<std::uint8_t[]>(len);

  std::size_t encoded_value_len = VarLenCodec::encode_signed(value, ptr.get());
  auto [decoded_value, decoded_value_len] = VarLenCodec::template decode_signed<Int>(ptr.get());

  EXPECT_EQ(value, decoded_value);
  EXPECT_EQ(len, encoded_value_len);
  EXPECT_EQ(len, decoded_value_len);
}

template <typename VarLenCodec, typename Int>
static void test_marker_varlen_codec(Int value, bool set) {
  std::size_t len = VarLenCodec::length_marker(value);
  auto ptr = std::make_unique<std::uint8_t[]>(len);

  std::size_t encoded_value_len = VarLenCodec::encode_with_marker(value, set, ptr.get());
  auto [decoded_value, marker_set, decoded_value_len] =
      VarLenCodec::template decode_with_marker<Int>(ptr.get());

  EXPECT_EQ(value, decoded_value);
  EXPECT_EQ(set, marker_set);
  EXPECT_EQ(len, encoded_value_len);
  EXPECT_EQ(len, decoded_value_len);
}

TEST(VariableLengthCodecTest, varlen_codec) {
  test_varlen_codec<VarIntCodec>(0);
  test_varlen_codec<VarIntCodec>(std::numeric_limits<std::size_t>::max() - 1);
}

TEST(VariableLengthCodecTest, signed_varlen_codec) {
  test_signed_varlen_codec<VarIntCodec>(0);
  test_signed_varlen_codec<VarIntCodec>(std::numeric_limits<int>::min() + 1);
  test_signed_varlen_codec<VarIntCodec>(std::numeric_limits<int>::max() - 1);
}

TEST(VariableLengthCodecTest, marker_varlen_codec) {
  test_marker_varlen_codec<VarIntCodec>(0, true);
  test_marker_varlen_codec<VarIntCodec>(0, false);

  test_marker_varlen_codec<VarIntCodec>(std::numeric_limits<std::size_t>::max() - 1, true);
  test_marker_varlen_codec<VarIntCodec>(std::numeric_limits<std::size_t>::max() - 1, false);
}
