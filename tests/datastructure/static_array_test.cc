/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#include "kaminpar/datastructure/static_array.h"
#include "tests.h"

#include <gmock/gmock.h>
#include <ranges>

using ::testing::Eq;

namespace kaminpar {
TEST(StaticArrayTest, SimpleStorageTest) {
  StaticArray<int> array(10);
  for (std::size_t i = 0; i < 10; ++i) { array[i] = 10 * i; }
  for (std::size_t i = 0; i < 10; ++i) { EXPECT_THAT(array[i], Eq(10 * i)); }
  EXPECT_THAT(array.size(), Eq(10));
  EXPECT_FALSE(array.empty());
}

TEST(StaticArrayTest, IteratorTest) {
  StaticArray<int> array(10);
  for (std::size_t i = 0; i < 10; ++i) { array[i] = 10 * i; }
  std::size_t i{0};

  for (const int &v : array) {
    EXPECT_THAT(v, i * 10);
    ++i;
  }
}
} // namespace kaminpar