#include <numeric>

#include <gmock/gmock.h>

#include "common/parallel/algorithm.h"

using ::testing::ElementsAre;
using namespace kaminpar;

TEST(ParallelAlgorithmTest, prefix_sum_with_multiple_elements) {
    std::vector<int> input{1, 2, 3, 4, 5};
    std::vector<int> output(input.size());
    parallel::prefix_sum(input.begin(), input.end(), output.begin());
    EXPECT_THAT(output, ElementsAre(1, 3, 6, 10, 15));
}

TEST(ParallelAlgorithmTest, prefix_sum_with_multiple_elements_inplace) {
    std::vector<int> data{1, 2, 3, 4, 5};
    parallel::prefix_sum(data.begin(), data.end(), data.begin());
    EXPECT_THAT(data, ElementsAre(1, 3, 6, 10, 15));
}

TEST(ParallelAlgorithmTest, max_element_with_single_element) {
    std::vector<int> data{1};
    const int        result = parallel::max_element(data);
    EXPECT_EQ(result, 1);
}

TEST(ParallelAlgorithmTest, max_element_with_multiple_elements) {
    std::vector<int> data{1, 5, 2, 10, 4};
    const int        result = parallel::max_element(data);
    EXPECT_EQ(result, 10);
}

TEST(ParallelAlgorithmTest, max_element_with_single_element_subset) {
    std::vector<int> data{1, 2, 3};
    const int        result = parallel::max_element(data.begin() + 1, data.begin() + 2);
    EXPECT_EQ(result, 2);
}

TEST(ParallelAlgorithmTest, max_element_with_iterators) {
    std::vector<int> data{1, 2, 3};
    const int        result = parallel::max_element(data.begin(), data.end());
    EXPECT_EQ(result, 3);
}

TEST(ParallelAlgorithmTest, accumulate_with_zero_elements) {
    std::vector<int> data;
    const int        result = parallel::accumulate(data, 0);
    EXPECT_EQ(result, 0);
}

TEST(ParallelAlgorithmTest, accumulate_with_zero_elements_initial_value) {
    std::vector<int> data;
    const int        result = parallel::accumulate(data, 42);
    EXPECT_EQ(result, 42);
}

TEST(ParallelAlgorithmTest, accumulate_with_sinle_element) {
    std::vector<int> data{1};
    const int        result = parallel::accumulate(data, 0);
    EXPECT_EQ(result, 1);
}

TEST(ParallelAlgorithmTest, accumulate_with_single_element_initial_value) {
    std::vector<int> data{1};
    const int        result = parallel::accumulate(data, 42);
    EXPECT_EQ(result, 43);
}

TEST(ParallelAlgorithmTest, accumulate_with_multiple_elements) {
    std::vector<int> data{1, 2, 3, 4, 5};
    const int        result = parallel::accumulate(data, 0);
    EXPECT_EQ(result, 15);
}

TEST(ParallelAlgorithmTest, accumulate_with_multiple_elements_initial_value) {
    std::vector<int> data{1, 2, 3, 4, 5};
    const int        result = parallel::accumulate(data, 42);
    EXPECT_EQ(result, 15 + 42);
}

TEST(ParallelAlgorithmTest, accumulate_with_many_elements) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);
    const int result = parallel::accumulate(data, 0);
    EXPECT_EQ(result, (1000 * 999) / 2);
}

TEST(ParallelAlgorithmTest, accumulate_empty_subset_with_initial_value) {
    std::vector<int> data{1, 2, 3};
    const int        result = parallel::accumulate(data.begin() + 1, data.begin() + 1, 42);
    EXPECT_EQ(result, 42);
}

TEST(ParallelAlgorithmTest, accumulate_single_element_subset) {
    std::vector<int> data{1, 2, 3};
    const int        result = parallel::accumulate(data.begin(), data.begin() + 1, 0);
    EXPECT_EQ(result, 1);
}

TEST(ParallelAlgorithmTest, accumulate_large_subset) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);
    const int result = parallel::accumulate(data.begin(), data.end(), 0);
    EXPECT_EQ(result, (1000 * 999) / 2);
}
