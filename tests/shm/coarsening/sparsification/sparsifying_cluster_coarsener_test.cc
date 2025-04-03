#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "kaminpar-shm/coarsening/sparsifying_cluster_coarsener.h"

namespace kaminpar::shm::testing {

std::unique_ptr<SparsifyingClusterCoarsener>
make_scc(float density_factor, float reduction_factor) {
  Context ctx = kaminpar::shm::create_default_context();
  ctx.sparsification.density_target_factor = density_factor;
  ctx.sparsification.reduction_target_factor = reduction_factor;

  return std::make_unique<SparsifyingClusterCoarsener>(ctx, ctx.partition);
}

TEST(SparsifingClusterCoasener, SparsificationTargetTest) {
  const float infinity = std::numeric_limits<float>::infinity();

  auto both_infinity = make_scc(infinity, infinity);
  ASSERT_GE(both_infinity->sparsificationTarget(1000, 10000, 1), 1000);

  auto reduction_to_20_percent = make_scc(infinity, .2);
  ASSERT_EQ(reduction_to_20_percent->sparsificationTarget(1000, 123, 321), 200);

  auto no_density_inc = make_scc(1, infinity);
  ASSERT_EQ(no_density_inc->sparsificationTarget(10000, 12300, 123), 100);

  auto d2_r70_percent = make_scc(2, .7);
  ASSERT_EQ(d2_r70_percent->sparsificationTarget(1000, 100, 25), 500);
  ASSERT_EQ(d2_r70_percent->sparsificationTarget(1000, 100, 80), 700);
}

} // namespace kaminpar::shm::testing
