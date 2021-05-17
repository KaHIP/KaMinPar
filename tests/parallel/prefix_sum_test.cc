#include "parallel.h"

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
}