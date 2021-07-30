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
#include "kaminpar/parallel.h"

#include "gmock/gmock.h"

using ::testing::ElementsAre;

namespace kaminpar::parallel {
TEST(PrefixSumTest, ComputesCorrectPrefixSum) {
  std::vector<int> input{1, 2, 3, 4, 5};
  std::vector<int> output(input.size());
  prefix_sum(input.begin(), input.end(), output.begin());
  EXPECT_THAT(output, ElementsAre(1, 3, 6, 10, 15));
}

TEST(PrefixSumTest, ComputesCorrectPrefixSumInplace) {
  std::vector<int> data{1, 2, 3, 4, 5};
  prefix_sum(data.begin(), data.end(), data.begin());
  EXPECT_THAT(data, ElementsAre(1, 3, 6, 10, 15));
}

TEST(AccumulateTest, AccumulatesCorrectly) {
  std::vector<int> data{1, 2, 3, 4, 5};
  const int result = parallel::accumulate(data);
  EXPECT_THAT(result, 15);
}

TEST(AccumulateTest, AccumulatesSingleValueCorrectly) {
  std::vector<int> data{1};
  const int result = parallel::accumulate(data);
  EXPECT_THAT(result, 1);
}

TEST(MaxTest, FindsMaxInSingleValue) {
  std::vector<int> data{1};
  const int result = parallel::max_element(data);
  EXPECT_THAT(result, 1);
}

TEST(MaxTest, FindsMaxCorrectly) {
  std::vector<int> data{1, 5, 2, 10, 4};
  const int result = parallel::max_element(data);
  EXPECT_THAT(result, 10);
}
} // namespace kaminpar::parallel