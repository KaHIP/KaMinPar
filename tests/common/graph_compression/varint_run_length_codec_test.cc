#include <gmock/gmock.h>

#include "kaminpar-common/graph_compression/varint_rle.h"

using namespace kaminpar;

template <typename Int> void test_run_length_codec() {
  const std::size_t len =
      (1 + sizeof(Int)) * (sizeof(Int) + sizeof(Int) * VarIntRunLengthEncoder<Int>::kBufferSize) +
      1;
  auto ptr = std::make_unique<std::uint8_t[]>(len);

  std::vector<Int> values;
  for (std::size_t i = 0; i < sizeof(Int); ++i) {
    values.push_back(static_cast<Int>(1) << (i * 8));
  }
  values.push_back(std::numeric_limits<Int>::max());
  for (std::size_t i = 0; i < sizeof(Int); ++i) {
    for (std::size_t j = 0; j < VarIntRunLengthEncoder<Int>::kBufferSize; ++j) {
      values.push_back(static_cast<Int>(1) << (i * 8));
    }
  }

  VarIntRunLengthEncoder<Int> rl_encoder(ptr.get());
  std::size_t written = 0;
  for (const Int value : values) {
    written += rl_encoder.add(value);
  }
  rl_encoder.flush();

  VarIntRunLengthDecoder<Int> rl_decoder(values.size(), ptr.get());
  std::size_t i = 0;
  rl_decoder.decode([&](const Int value) { EXPECT_EQ(values[i++], value); });
  EXPECT_EQ(i, values.size());
}

TEST(VarIntRunLengthCodecTest, run_length_codec) {
  test_run_length_codec<std::uint32_t>();
  test_run_length_codec<std::uint64_t>();
}
