#include "gmock/gmock.h"

#include "datastructure/fast_reset_array.h"

namespace kaminpar {
TEST(FastResetArrayTest, SizeWorks) {
  FastResetArray<int> array(128);
  EXPECT_THAT(array.size(), 128);
}

TEST(FastResetArrayTest, EmptyArrayWorks) {
  FastResetArray<int> array(0);
  EXPECT_THAT(array.size(), 0);
  EXPECT_TRUE(array.empty());
}

TEST(FastResetArrayTest, EmptyWorks) {
  FastResetArray<int> array(10);
  EXPECT_TRUE(array.empty());
  array.set(1, 1);
  EXPECT_FALSE(array.empty());
  array.set(1, 0);
  EXPECT_FALSE(array.empty());
  array.reset();
  EXPECT_TRUE(array.empty());
  array.set(1, 0);
  EXPECT_FALSE(array.empty());
  array.reset();
  EXPECT_TRUE(array.empty());
}

TEST(FastResetArrayTest, InitializationWorks) {
  FastResetArray<int> array(1);
  EXPECT_THAT(array.get(0), 0);
}

TEST(FastResetArrayTest, SettingElementsWorks) {
  FastResetArray<int> array(1);
  array.set(0, 42);
  EXPECT_THAT(array.get(0), 42);
}

TEST(FastResetArrayTest, ResettingElementsWorks) {
  FastResetArray<int> array(1);
  array.set(0, 42);
  array.reset();
  EXPECT_THAT(array.get(0), 0);
}

TEST(FastResetArrayTest, ResettingMultipleElementsWorks) {
  constexpr std::size_t kCapacity = 128;
  FastResetArray<int> array(kCapacity);
  for (std::size_t i = 0; i < kCapacity; ++i) { array.set(i, 128 * i); }
  array.reset();
  for (std::size_t i = 0; i < kCapacity; ++i) { EXPECT_THAT(array.get(i), 0); }
}

TEST(FastResetArrayTest, SettingElementsWithGapsWorks) {
  constexpr std::size_t kCapacity = 128;
  FastResetArray<int> array(kCapacity);
  array.set(0, 10);
  array.set(kCapacity / 2, 50);

  EXPECT_THAT(array.get(0), 10);
  EXPECT_THAT(array.get(kCapacity / 2), 50);
  for (std::size_t i = 1; i < kCapacity / 2; ++i) { EXPECT_THAT(array.get(i), 0); }
  for (std::size_t i = kCapacity + 1; i < kCapacity; ++i) { EXPECT_THAT(array.get(i), 0); }

  array.reset();
  for (std::size_t i = 0; i < kCapacity; ++i) { EXPECT_THAT(array.get(i), 0); }
}

TEST(FastResetArrayTest, HoldingAndResettingMultipleElementsWorks) {
  constexpr std::size_t kCapacity = 100;
  FastResetArray<int> array(kCapacity);
  for (int e = 0; e < static_cast<int>(kCapacity); e++) { array.set(e, 2 * e); }
  for (int e = 0; e < static_cast<int>(kCapacity); e++) { EXPECT_THAT(array.get(e), 2 * e); }
  array.reset();
  for (int e = 0; e < static_cast<int>(kCapacity); e++) { EXPECT_THAT(array.get(e), 0); }
}

TEST(FastResetArrayTest, ComplexDatatypeWorks) {
  FastResetArray<std::string> array(16);
  for (std::size_t i = 0; i < array.size(); ++i) { array.set(i, std::to_string(1000 + i)); }
  for (std::size_t i = 0; i < array.size(); ++i) { EXPECT_THAT(array.get(i), std::to_string(1000 + i)); }
  array.reset();
  for (std::size_t i = 0; i < array.size(); ++i) { EXPECT_THAT(array.get(i), ""); }
}
} // namespace kaminpar
