#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/parallel/quickselect.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm::testing {

TEST(SparsificationUtils, QselectStupidTests) {
  StaticArray<int> three_nums(3);
  three_nums[0] = 1;
  three_nums[1] = 2;
  three_nums[2] = 42;

  ASSERT_EQ(quickselect_k_smallest<int>(2, three_nums.begin(), three_nums.end()).value, 2);
}

TEST(SparsificationUtils, QselctOnPermutation) {
  StaticArray<int> permutation_of_1_to_10(10);
  permutation_of_1_to_10[0] = 8;
  permutation_of_1_to_10[1] = 1;
  permutation_of_1_to_10[2] = 3;
  permutation_of_1_to_10[3] = 5;
  permutation_of_1_to_10[4] = 7;
  permutation_of_1_to_10[5] = 9;
  permutation_of_1_to_10[6] = 2;
  permutation_of_1_to_10[7] = 10;
  permutation_of_1_to_10[8] = 6;
  permutation_of_1_to_10[9] = 4;

  ASSERT_EQ(permutation_of_1_to_10.size(), 10);
  ASSERT_EQ(std::distance(permutation_of_1_to_10.begin(), permutation_of_1_to_10.end()), 10);

  for (size_t k = 1; k <= permutation_of_1_to_10.size(); k++) {
    ASSERT_EQ(
        quickselect_k_smallest<int>(k, permutation_of_1_to_10.begin(), permutation_of_1_to_10.end())
            .value,
        k
    );
  }
}

TEST(SparsificationUtils, QselectOnRandomNumbers) {
  size_t times = 1 << 4;
  size_t size = 1 << 12;

  for (size_t i = 0; i < times; i++) {
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
      auto info = quickselect_k_smallest<double>(k, numbers.begin(), numbers.end());

      size_t number_eq = 0;
      size_t number_lt = 0;
      size_t number_gt = 0;
      for (double x : sorted_numbers) {
        if (x == info.value)
          number_eq++;
        else if (x < info.value)
          number_lt++;
        else
          number_gt++;
      }
      ASSERT_LE(number_gt, size - k);
      ASSERT_EQ(info.value, sorted_numbers[k - 1]);
      ASSERT_EQ(info.number_of_elements_equal, number_eq);
      ASSERT_EQ(info.number_of_elements_smaller, number_lt);
    }
  }
}

TEST(SparsificationUtils, QselectAllEqual) {
  size_t n = 1 << 20;
  StaticArray<int> numbers(n, 42);
  QuickselectResult<int> info =
      quickselect_k_smallest<int>(std::round(.23 * n), numbers.begin(), numbers.end());
  ASSERT_EQ(info.value, 42);
  ASSERT_EQ(info.number_of_elements_equal, n);
  ASSERT_EQ(info.number_of_elements_smaller, 0);
}

} // namespace kaminpar::shm::testing
