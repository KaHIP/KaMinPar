#include <gtest/gtest.h>

#include "common/utils/math.h"

using namespace kaminpar;

TEST(MathTest, encode_virtual_square_2x2elements) {
    const int num_elements = 4;
    EXPECT_EQ(encode_virtual_square_position(0, 0, num_elements), 0);
    EXPECT_EQ(encode_virtual_square_position(0, 1, num_elements), 1);
    EXPECT_EQ(encode_virtual_square_position(1, 0, num_elements), 2);
    EXPECT_EQ(encode_virtual_square_position(1, 1, num_elements), 3);
}

TEST(MathTest, decode_virtual_square_2x2elements) {
    const int num_elements = 4;
    EXPECT_EQ(decode_virtual_square_position(0, num_elements), std::make_pair(0, 0));
    EXPECT_EQ(decode_virtual_square_position(1, num_elements), std::make_pair(0, 1));
    EXPECT_EQ(decode_virtual_square_position(2, num_elements), std::make_pair(1, 0));
    EXPECT_EQ(decode_virtual_square_position(3, num_elements), std::make_pair(1, 1));
}

TEST(MathTest, encode_virtual_square_3x3elements) {
    const int num_elements = 9;
    EXPECT_EQ(encode_virtual_square_position(0, 0, num_elements), 0);
    EXPECT_EQ(encode_virtual_square_position(0, 1, num_elements), 1);
    EXPECT_EQ(encode_virtual_square_position(0, 2, num_elements), 2);
    EXPECT_EQ(encode_virtual_square_position(1, 0, num_elements), 3);
    EXPECT_EQ(encode_virtual_square_position(1, 1, num_elements), 4);
    EXPECT_EQ(encode_virtual_square_position(1, 2, num_elements), 5);
    EXPECT_EQ(encode_virtual_square_position(2, 0, num_elements), 6);
    EXPECT_EQ(encode_virtual_square_position(2, 1, num_elements), 7);
    EXPECT_EQ(encode_virtual_square_position(2, 2, num_elements), 8);
}

TEST(MathTest, decode_virtual_square_3x3elements) {
    const int num_elements = 9;
    EXPECT_EQ(decode_virtual_square_position(0, num_elements), std::make_pair(0, 0));
    EXPECT_EQ(decode_virtual_square_position(1, num_elements), std::make_pair(0, 1));
    EXPECT_EQ(decode_virtual_square_position(2, num_elements), std::make_pair(0, 2));
    EXPECT_EQ(decode_virtual_square_position(3, num_elements), std::make_pair(1, 0));
    EXPECT_EQ(decode_virtual_square_position(4, num_elements), std::make_pair(1, 1));
    EXPECT_EQ(decode_virtual_square_position(5, num_elements), std::make_pair(1, 2));
    EXPECT_EQ(decode_virtual_square_position(6, num_elements), std::make_pair(2, 0));
    EXPECT_EQ(decode_virtual_square_position(7, num_elements), std::make_pair(2, 1));
    EXPECT_EQ(decode_virtual_square_position(8, num_elements), std::make_pair(2, 2));
}

TEST(MathTest, encode_virtual_square_5elements) {
    const int num_elements = 5;
    EXPECT_EQ(encode_virtual_square_position(0, 0, num_elements), 0);
    EXPECT_EQ(encode_virtual_square_position(0, 1, num_elements), 1);
    EXPECT_EQ(encode_virtual_square_position(1, 0, num_elements), 2);
    EXPECT_EQ(encode_virtual_square_position(1, 1, num_elements), 3);
    EXPECT_EQ(encode_virtual_square_position(0, 2, num_elements), 4);
}

TEST(MathTest, decode_virtual_square_5elements) {
    const int num_elements = 5;
    EXPECT_EQ(decode_virtual_square_position(0, num_elements), std::make_pair(0, 0));
    EXPECT_EQ(decode_virtual_square_position(1, num_elements), std::make_pair(0, 1));
    EXPECT_EQ(decode_virtual_square_position(2, num_elements), std::make_pair(1, 0));
    EXPECT_EQ(decode_virtual_square_position(3, num_elements), std::make_pair(1, 1));
    EXPECT_EQ(decode_virtual_square_position(4, num_elements), std::make_pair(0, 2));
}

TEST(MathTest, encode_virtual_square_8elements) {
    const int num_elements = 8;
    EXPECT_EQ(encode_virtual_square_position(0, 0, num_elements), 0);
    EXPECT_EQ(encode_virtual_square_position(0, 1, num_elements), 1);
    EXPECT_EQ(encode_virtual_square_position(1, 0, num_elements), 2);
    EXPECT_EQ(encode_virtual_square_position(1, 1, num_elements), 3);
    EXPECT_EQ(encode_virtual_square_position(0, 2, num_elements), 4);
    EXPECT_EQ(encode_virtual_square_position(1, 2, num_elements), 5);
    EXPECT_EQ(encode_virtual_square_position(0, 3, num_elements), 6);
    EXPECT_EQ(encode_virtual_square_position(1, 3, num_elements), 7);
}

TEST(MathTest, decode_virtual_square_8elements) {
    const int num_elements = 8;
    EXPECT_EQ(decode_virtual_square_position(0, num_elements), std::make_pair(0, 0));
    EXPECT_EQ(decode_virtual_square_position(1, num_elements), std::make_pair(0, 1));
    EXPECT_EQ(decode_virtual_square_position(2, num_elements), std::make_pair(1, 0));
    EXPECT_EQ(decode_virtual_square_position(3, num_elements), std::make_pair(1, 1));
    EXPECT_EQ(decode_virtual_square_position(4, num_elements), std::make_pair(0, 2));
    EXPECT_EQ(decode_virtual_square_position(5, num_elements), std::make_pair(1, 2));
    EXPECT_EQ(decode_virtual_square_position(6, num_elements), std::make_pair(0, 3));
    EXPECT_EQ(decode_virtual_square_position(7, num_elements), std::make_pair(1, 3));
}
