#include "common/utils/math.h"
#include "gmock/gmock.h"

using ::testing::AnyOf;
using ::testing::Eq;
using ::testing::Ne;

namespace kaminpar::math {
TEST(MathTest, FloorLog2) {
    EXPECT_EQ(0, floor_log2(1u));
    EXPECT_EQ(1, floor_log2(2u));
    EXPECT_EQ(1, floor_log2(3u));
    EXPECT_EQ(2, floor_log2(4u));
    EXPECT_EQ(2, floor_log2(5u));
    EXPECT_EQ(2, floor_log2(6u));
    EXPECT_EQ(2, floor_log2(7u));
    EXPECT_EQ(3, floor_log2(8u));
    EXPECT_EQ(4, floor_log2(16u));
}

TEST(MathTest, CeilLog2) {
    EXPECT_EQ(0, ceil_log2(1u));
    EXPECT_EQ(1, ceil_log2(2u));
    EXPECT_EQ(2, ceil_log2(3u));
    EXPECT_EQ(2, ceil_log2(4u));
    EXPECT_EQ(3, ceil_log2(5u));
    EXPECT_EQ(3, ceil_log2(6u));
    EXPECT_EQ(3, ceil_log2(7u));
    EXPECT_EQ(3, ceil_log2(8u));
}

TEST(MathTest, SplitNumberEvenlyWorks) {
    {
        const auto [k0, k1] = math::split_integral(0);
        EXPECT_THAT(k0, Eq(0));
        EXPECT_THAT(k1, Eq(0));
    }
    {
        const auto [k0, k1] = math::split_integral(1);
        EXPECT_THAT(k0, AnyOf(0, 1));
        EXPECT_THAT(k1, AnyOf(0, 1));
        EXPECT_THAT(k0, Ne(k1));
    }
    {
        const auto [k0, k1] = math::split_integral(2);
        EXPECT_THAT(k0, Eq(1));
        EXPECT_THAT(k1, Eq(1));
    }
}

TEST(MathTest, SplitNumberWithRatioWorks) {
    {
        const auto [k0, k1] = math::split_integral(0, 0.1);
        EXPECT_THAT(k0, Eq(0));
        EXPECT_THAT(k1, Eq(0));
    }
    {
        const auto [k0, k1] = math::split_integral(1, 0.0);
        EXPECT_THAT(k0, Eq(0));
        EXPECT_THAT(k1, Eq(1));
    }
    {
        const auto [k0, k1] = math::split_integral(1, 1.0);
        EXPECT_THAT(k0, Eq(1));
        EXPECT_THAT(k1, Eq(0));
    }
    {
        const auto [k0, k1] = math::split_integral(10, 0.1);
        EXPECT_THAT(k0, Eq(1));
        EXPECT_THAT(k1, Eq(9));
    }
}

TEST(MathTest, RoundDownToPowerOfTwo) {
    EXPECT_THAT(math::round_down_to_power_of_2(1u), Eq(1));
    EXPECT_THAT(math::round_down_to_power_of_2(2u), Eq(2));
    EXPECT_THAT(math::round_down_to_power_of_2(3u), Eq(2));
    EXPECT_THAT(math::round_down_to_power_of_2(4u), Eq(4));
    EXPECT_THAT(math::round_down_to_power_of_2(5u), Eq(4));
    EXPECT_THAT(math::round_down_to_power_of_2(6u), Eq(4));
    EXPECT_THAT(math::round_down_to_power_of_2(7u), Eq(4));
    EXPECT_THAT(math::round_down_to_power_of_2(8u), Eq(8));
    EXPECT_THAT(math::round_down_to_power_of_2(1023u), Eq(512));
    EXPECT_THAT(math::round_down_to_power_of_2(1024u), Eq(1024));
    EXPECT_THAT(math::round_down_to_power_of_2(1025u), Eq(1024));
}

TEST(MathTest, RoundUpToPowerOfTwo) {
    EXPECT_THAT(math::round_up_to_power_of_2(1u), Eq(1));
    EXPECT_THAT(math::round_up_to_power_of_2(2u), Eq(2));
    EXPECT_THAT(math::round_up_to_power_of_2(3u), Eq(4));
    EXPECT_THAT(math::round_up_to_power_of_2(4u), Eq(4));
    EXPECT_THAT(math::round_up_to_power_of_2(5u), Eq(8));
    EXPECT_THAT(math::round_up_to_power_of_2(6u), Eq(8));
    EXPECT_THAT(math::round_up_to_power_of_2(7u), Eq(8));
    EXPECT_THAT(math::round_up_to_power_of_2(8u), Eq(8));
    EXPECT_THAT(math::round_up_to_power_of_2(1023u), Eq(1024));
    EXPECT_THAT(math::round_up_to_power_of_2(1024u), Eq(1024));
    EXPECT_THAT(math::round_up_to_power_of_2(1025u), Eq(2048));
}
} // namespace kaminpar::math
