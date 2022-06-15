#include <gtest/gtest.h>

#include "common/utils/math.h"

using namespace kaminpar;

TEST(MathTest, encode_virtual_square_1element) {
    EXPECT_EQ(encode_virtual_square_position(0, 0, 1), 0);
}

TEST(MathTest, decode_virtual_square_1element) {
    EXPECT_EQ(decode_virtual_square_position(0, 1), std::make_pair(0, 0));
}

TEST(MathTest, encode_virtual_square_2elements) {
    const int num_elements = 2;
    EXPECT_EQ(encode_virtual_square_position(0, 0, num_elements), 0);
    EXPECT_EQ(encode_virtual_square_position(0, 1, num_elements), 1);
}

TEST(MathTest, decode_virtual_square_2elements) {
    const int num_elements = 2;
    EXPECT_EQ(decode_virtual_square_position(0, num_elements), std::make_pair(0, 0));
    EXPECT_EQ(decode_virtual_square_position(1, num_elements), std::make_pair(0, 1));
}

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
    EXPECT_EQ(encode_virtual_square_position(0, 2, num_elements), 2);
    EXPECT_EQ(encode_virtual_square_position(1, 0, num_elements), 3);
    EXPECT_EQ(encode_virtual_square_position(1, 1, num_elements), 4);
}

TEST(MathTest, decode_virtual_square_5elements) {
    const int num_elements = 5;
    EXPECT_EQ(decode_virtual_square_position(0, num_elements), std::make_pair(0, 0));
    EXPECT_EQ(decode_virtual_square_position(1, num_elements), std::make_pair(0, 1));
    EXPECT_EQ(decode_virtual_square_position(2, num_elements), std::make_pair(0, 2));
    EXPECT_EQ(decode_virtual_square_position(3, num_elements), std::make_pair(1, 0));
    EXPECT_EQ(decode_virtual_square_position(4, num_elements), std::make_pair(1, 1));
}

TEST(MathTest, encode_virtual_square_8elements) {
    const int num_elements = 8;
    EXPECT_EQ(encode_virtual_square_position(0, 0, num_elements), 0);
    EXPECT_EQ(encode_virtual_square_position(0, 1, num_elements), 1);
    EXPECT_EQ(encode_virtual_square_position(0, 2, num_elements), 2);
    EXPECT_EQ(encode_virtual_square_position(1, 0, num_elements), 3);
    EXPECT_EQ(encode_virtual_square_position(1, 1, num_elements), 4);
    EXPECT_EQ(encode_virtual_square_position(1, 2, num_elements), 5);
    EXPECT_EQ(encode_virtual_square_position(2, 0, num_elements), 6);
    EXPECT_EQ(encode_virtual_square_position(2, 1, num_elements), 7);
}

TEST(MathTest, decode_virtual_square_8elements) {
    const int num_elements = 8;
    EXPECT_EQ(decode_virtual_square_position(0, num_elements), std::make_pair(0, 0));
    EXPECT_EQ(decode_virtual_square_position(1, num_elements), std::make_pair(0, 1));
    EXPECT_EQ(decode_virtual_square_position(2, num_elements), std::make_pair(0, 2));
    EXPECT_EQ(decode_virtual_square_position(3, num_elements), std::make_pair(1, 0));
    EXPECT_EQ(decode_virtual_square_position(4, num_elements), std::make_pair(1, 1));
    EXPECT_EQ(decode_virtual_square_position(5, num_elements), std::make_pair(1, 2));
    EXPECT_EQ(decode_virtual_square_position(6, num_elements), std::make_pair(2, 0));
    EXPECT_EQ(decode_virtual_square_position(7, num_elements), std::make_pair(2, 1));
}

TEST(MathTest, encode_virtual_square_10elements) {
    const int num_elements = 10;
    EXPECT_EQ(encode_virtual_square_position(0, 0, num_elements), 0);
    EXPECT_EQ(encode_virtual_square_position(0, 1, num_elements), 1);
    EXPECT_EQ(encode_virtual_square_position(0, 2, num_elements), 2);
    EXPECT_EQ(encode_virtual_square_position(0, 3, num_elements), 3);
    EXPECT_EQ(encode_virtual_square_position(1, 0, num_elements), 4);
    EXPECT_EQ(encode_virtual_square_position(1, 1, num_elements), 5);
    EXPECT_EQ(encode_virtual_square_position(1, 2, num_elements), 6);
    EXPECT_EQ(encode_virtual_square_position(2, 0, num_elements), 7);
    EXPECT_EQ(encode_virtual_square_position(2, 1, num_elements), 8);
    EXPECT_EQ(encode_virtual_square_position(2, 2, num_elements), 9);
}

TEST(MathTest, decode_virtual_square_10elements) {
    const int num_elements = 10;
    EXPECT_EQ(decode_virtual_square_position(0, num_elements), std::make_pair(0, 0));
    EXPECT_EQ(decode_virtual_square_position(1, num_elements), std::make_pair(0, 1));
    EXPECT_EQ(decode_virtual_square_position(2, num_elements), std::make_pair(0, 2));
    EXPECT_EQ(decode_virtual_square_position(3, num_elements), std::make_pair(0, 3));
    EXPECT_EQ(decode_virtual_square_position(4, num_elements), std::make_pair(1, 0));
    EXPECT_EQ(decode_virtual_square_position(5, num_elements), std::make_pair(1, 1));
    EXPECT_EQ(decode_virtual_square_position(6, num_elements), std::make_pair(1, 2));
    EXPECT_EQ(decode_virtual_square_position(7, num_elements), std::make_pair(2, 0));
    EXPECT_EQ(decode_virtual_square_position(8, num_elements), std::make_pair(2, 1));
    EXPECT_EQ(decode_virtual_square_position(9, num_elements), std::make_pair(2, 2));
}

TEST(MathTest, encode_virtual_square_18elements) {
    const int num_elements = 18;
    EXPECT_EQ(encode_virtual_square_position(0, 0, num_elements), 0);
    EXPECT_EQ(encode_virtual_square_position(0, 1, num_elements), 1);
    EXPECT_EQ(encode_virtual_square_position(0, 2, num_elements), 2);
    EXPECT_EQ(encode_virtual_square_position(0, 3, num_elements), 3);
    EXPECT_EQ(encode_virtual_square_position(0, 4, num_elements), 4);
    EXPECT_EQ(encode_virtual_square_position(1, 0, num_elements), 5);
    EXPECT_EQ(encode_virtual_square_position(1, 1, num_elements), 6);
    EXPECT_EQ(encode_virtual_square_position(1, 2, num_elements), 7);
    EXPECT_EQ(encode_virtual_square_position(1, 3, num_elements), 8);
    EXPECT_EQ(encode_virtual_square_position(1, 4, num_elements), 9);
    EXPECT_EQ(encode_virtual_square_position(2, 0, num_elements), 10);
    EXPECT_EQ(encode_virtual_square_position(2, 1, num_elements), 11);
    EXPECT_EQ(encode_virtual_square_position(2, 2, num_elements), 12);
    EXPECT_EQ(encode_virtual_square_position(2, 3, num_elements), 13);
    EXPECT_EQ(encode_virtual_square_position(3, 0, num_elements), 14);
    EXPECT_EQ(encode_virtual_square_position(3, 1, num_elements), 15);
    EXPECT_EQ(encode_virtual_square_position(3, 2, num_elements), 16);
    EXPECT_EQ(encode_virtual_square_position(3, 3, num_elements), 17);
}

TEST(MathTest, decode_virtual_square_18elements) {
    const int num_elements = 18;
    EXPECT_EQ(decode_virtual_square_position(0, num_elements), std::make_pair(0, 0));
    EXPECT_EQ(decode_virtual_square_position(1, num_elements), std::make_pair(0, 1));
    EXPECT_EQ(decode_virtual_square_position(2, num_elements), std::make_pair(0, 2));
    EXPECT_EQ(decode_virtual_square_position(3, num_elements), std::make_pair(0, 3));
    EXPECT_EQ(decode_virtual_square_position(4, num_elements), std::make_pair(0, 4));
    EXPECT_EQ(decode_virtual_square_position(5, num_elements), std::make_pair(1, 0));
    EXPECT_EQ(decode_virtual_square_position(6, num_elements), std::make_pair(1, 1));
    EXPECT_EQ(decode_virtual_square_position(7, num_elements), std::make_pair(1, 2));
    EXPECT_EQ(decode_virtual_square_position(8, num_elements), std::make_pair(1, 3));
    EXPECT_EQ(decode_virtual_square_position(9, num_elements), std::make_pair(1, 4));
    EXPECT_EQ(decode_virtual_square_position(10, num_elements), std::make_pair(2, 0));
    EXPECT_EQ(decode_virtual_square_position(11, num_elements), std::make_pair(2, 1));
    EXPECT_EQ(decode_virtual_square_position(12, num_elements), std::make_pair(2, 2));
    EXPECT_EQ(decode_virtual_square_position(13, num_elements), std::make_pair(2, 3));
    EXPECT_EQ(decode_virtual_square_position(14, num_elements), std::make_pair(3, 0));
    EXPECT_EQ(decode_virtual_square_position(15, num_elements), std::make_pair(3, 1));
    EXPECT_EQ(decode_virtual_square_position(16, num_elements), std::make_pair(3, 2));
    EXPECT_EQ(decode_virtual_square_position(17, num_elements), std::make_pair(3, 3));
}
