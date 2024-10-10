#include <gmock/internal/gmock-internal-utils.h>
#include <gtest/gtest.h>

#include "kaminpar-shm/coarsening/sparsification/sparsification_utils.h"
namespace kaminpar::shm::testing {
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
} // namespace kaminpar::shm::testing