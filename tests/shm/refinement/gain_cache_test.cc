/*******************************************************************************
 * @file:   gain_cache_test.cc
 * @author: Daniel Seemaier
 * @date:   16.08.2024
 ******************************************************************************/
#include <gmock/gmock.h>

#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/gains/compact_hashing_gain_cache.h"
#include "kaminpar-shm/refinement/gains/dense_gain_cache.h"
#include "kaminpar-shm/refinement/gains/hashing_gain_cache.h"
#include "kaminpar-shm/refinement/gains/on_the_fly_gain_cache.h"
#include "kaminpar-shm/refinement/gains/sparse_gain_cache.h"

using namespace kaminpar::shm;
using namespace kaminpar::shm::testing;

namespace {

template <typename GainCacheType> class GainCacheTest : public ::testing::Test {
public:
  void init(const PartitionedGraph &p_graph) {
    this->gain_cache_ = std::make_unique<GainCacheType>(this->ctx_, p_graph.n(), p_graph.k());
    this->gain_cache_->initialize(p_graph.graph(), p_graph);
  }

  void free() {
    this->gain_cache_->free();
    this->gain_cache_ = nullptr;
  }

  std::unique_ptr<GainCacheType> gain_cache_ = nullptr;
  Context ctx_ = create_default_context();
};

using GainCacheTypes = ::testing::Types<
    DenseGainCache<Graph>,
    NormalHashingGainCache<Graph>,
    OnTheFlyGainCache<Graph>,
    NormalSparseGainCache<Graph>,
    NormalCompactHashingGainCache<Graph>>;

TYPED_TEST_SUITE(GainCacheTest, GainCacheTypes);

TYPED_TEST(GainCacheTest, ObjectConstructionWorks) {
  this->gain_cache_ = std::make_unique<TypeParam>(this->ctx_, 0, 0);
}

TYPED_TEST(GainCacheTest, InitWorksOnEmptyGraphWith0Nodes) {
  auto empty = make_empty_graph(0);
  auto p_empty = make_p_graph(empty, 2, {});

  this->init(p_empty);
  this->free();
}

TYPED_TEST(GainCacheTest, InitWorksOnEmptyGraphWith4Nodes) {
  auto empty = make_empty_graph(4);
  auto p_empty = make_p_graph(empty, 2, {0, 0, 1, 1});

  this->init(p_empty);

  EXPECT_EQ(this->gain_cache_->conn(0, 0), 0);
  EXPECT_EQ(this->gain_cache_->conn(0, 1), 0);
  EXPECT_EQ(this->gain_cache_->conn(1, 0), 0);
  EXPECT_EQ(this->gain_cache_->conn(1, 1), 0);
  EXPECT_EQ(this->gain_cache_->conn(2, 0), 0);
  EXPECT_EQ(this->gain_cache_->conn(2, 1), 0);
  EXPECT_EQ(this->gain_cache_->conn(3, 0), 0);
  EXPECT_EQ(this->gain_cache_->conn(3, 1), 0);

  EXPECT_EQ(this->gain_cache_->gain(0, 0, 1), 0);
  EXPECT_EQ(this->gain_cache_->gain(1, 0, 1), 0);
  EXPECT_EQ(this->gain_cache_->gain(2, 1, 0), 0);
  EXPECT_EQ(this->gain_cache_->gain(2, 1, 0), 0);

  this->gain_cache_->gains(0, 0, [&](BlockID, auto &&gain) { EXPECT_EQ(gain(), 0); });

  this->free();
}

} // namespace
