#include <sstream>
#include <stdexcept>

#include <gtest/gtest.h>

#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/refinement/fm/fm_refiner.h"
#include "kaminpar-shm/refinement/fm/unconstrained_fm_refiner.h"

namespace kaminpar::shm::testing {
namespace {

TEST(GainCacheStrategyTest, SharedValuesRetainTheirNamesAcrossAlgorithmOptionMaps) {
  const auto all = get_gain_cache_strategies();
  for (const auto &supported : {get_fm_gain_cache_strategies(), get_jet_gain_cache_strategies()}) {
    for (const auto &[name, strategy] : supported) {
      std::ostringstream formatted;
      formatted << strategy;
      EXPECT_EQ(formatted.str(), name);
      EXPECT_EQ(all.at(name), strategy);
    }
  }
}

TEST(GainCacheStrategyTest, AlgorithmOptionMapsRejectUnsupportedBackends) {
  EXPECT_FALSE(get_fm_gain_cache_strategies().contains("jet"));
  EXPECT_FALSE(get_jet_gain_cache_strategies().contains("compact-hashing"));
  EXPECT_EQ(get_jet_gain_cache_strategies().at("jet"), GainCacheStrategy::JET);
  EXPECT_EQ(get_jet_gain_cache_strategies().at("sparse"), GainCacheStrategy::SPARSE);
  EXPECT_EQ(get_jet_gain_cache_strategies().at("on-the-fly"), GainCacheStrategy::ON_THE_FLY);
}

TEST(GainCacheStrategyTest, FMRejectsJetCacheSelectedThroughTheLibrary) {
  auto graph = make_empty_graph(2);
  auto p_graph = make_p_graph(graph, 2, {0, 1});
  auto ctx = create_default_context();
  ctx.partition.setup(graph, 2, 0.03);
  ctx.refinement.kway_fm.gain_cache_strategy = GainCacheStrategy::JET;
  FMRefiner refiner(ctx);
  EXPECT_THROW(refiner.initialize(p_graph), std::invalid_argument);
}

TEST(GainCacheStrategyTest, UnconstrainedFMRejectsJetCacheSelectedThroughTheLibrary) {
  auto graph = make_empty_graph(2);
  auto p_graph = make_p_graph(graph, 2, {0, 1});
  auto ctx = create_default_context();
  ctx.partition.setup(graph, 2, 0.03);
  ctx.refinement.kway_fm.gain_cache_strategy = GainCacheStrategy::JET;
  UnconstrainedFMRefiner refiner(ctx);
  EXPECT_THROW(refiner.initialize(p_graph), std::invalid_argument);
}

} // namespace
} // namespace kaminpar::shm::testing
