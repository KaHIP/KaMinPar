#include <gmock/internal/gmock-internal-utils.h>
#include <gtest/gtest.h>

#include "kaminpar-shm/coarsening/sparsification/sparsification_utils.h"
namespace kaminpar::shm::testing {
TEST(SparsificationUtils, QselectStupidTests) {
  std::vector<int> three_nums = {1, 2, 42};
  ASSERT_EQ(
      sparsification::utils::quickselect_k_smallest<int>(2, three_nums.begin(), three_nums.end()), 2
  );
}
TEST(SparsificationUtils, QselctOnPermutation) {
  std::vector<int> permutation_of_1_to_10 = {8, 1, 3, 5, 7, 9, 2, 10, 6, 4};
  ASSERT_EQ(permutation_of_1_to_10.size(), 10);
  ASSERT_EQ(std::distance(permutation_of_1_to_10.begin(), permutation_of_1_to_10.end()), 10);
  for (size_t k = 1; k <= permutation_of_1_to_10.size(); k++) {
    ASSERT_EQ(
        kaminpar::shm::sparsification::utils::quickselect_k_smallest<int>(
            k, permutation_of_1_to_10.begin(), permutation_of_1_to_10.end()
        ),
        k
    );
  }
}

TEST(SparsificationUtils, QselectOnRandomNumbers) {
  size_t times = 32;
  for (size_t i = 0; i < times; i++) {
    size_t size = 1024;
    StaticArray<double> numbers(size);
    StaticArray<double> sorted_numbers(size);
    for (size_t i = 0; i != size; i++) {
      double x = Random::instance().random_double();
      sorted_numbers[i] = x;
      numbers[i] = x;
    }
    std::sort(sorted_numbers.begin(), sorted_numbers.end());

    size_t number_of_ks = 42;
    std::vector<size_t> ks(number_of_ks);
    for (size_t i = 0; i != number_of_ks; i++)
      ks[i] = Random::instance().random_index(1, size + 1);
    for (size_t k : ks) {
      ASSERT_EQ(
          sparsification::utils::quickselect_k_smallest<double>(k, numbers.begin(), numbers.end()),
          sorted_numbers[k - 1]
      );
    }
  }
}

TEST(SparsificationUtils, Median) {
  std::vector<int> numbers_with_median_42 = {42, 1, 44, 2, 9, 99, 100000};
  ASSERT_EQ(
      sparsification::utils::median<int>(
          numbers_with_median_42.begin(), numbers_with_median_42.end()
      ),
      42
  );
  std::vector<double> numbers_with_median_three_halfs = {1, 42, 2, 0.1, 1e-42, 100};
  ASSERT_EQ(
      sparsification::utils::median<double>(
          numbers_with_median_three_halfs.begin(), numbers_with_median_three_halfs.end()
      ),
      1.5
  );
  std::vector<int> one_element = {42};
  ASSERT_EQ(sparsification::utils::median<int>(one_element.begin(), one_element.end()), 42);
  std::vector<double> two_elements = {123, 0.42};
  ASSERT_EQ(
      sparsification::utils::median<double>(two_elements.begin(), two_elements.end()),
      (two_elements[0] + two_elements[1]) / 2
  );
}
TEST(SparsificationUtils, MedianOfMedians) {
  std::vector<int> numbers_with_mom_2 = {
      2,
      1,
      3,
      0,
      -1, // median 1
      -42,
      2,
      1,
      42,
      100000, // median 2
      42      // median 42
  };
  ASSERT_EQ(
      sparsification::utils::medians_of_medians<int>(
          numbers_with_mom_2.begin(), numbers_with_mom_2.end()
      ),
      2
  );
}
} // namespace kaminpar::shm::testing