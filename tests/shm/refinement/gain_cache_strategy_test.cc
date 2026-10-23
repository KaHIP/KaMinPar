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
  const auto fm = get_gain_cache_strategies();
  const auto jet = get_jet_gain_cache_strategies();
  for (const auto &supported : {fm, jet}) {
    for (const auto &[name, strategy] : supported) {
      std::ostringstream formatted;
      formatted << strategy;
      EXPECT_EQ(formatted.str(), name);
    }
  }
  for (const auto *name : {"sparse", "on-the-fly"}) {
    EXPECT_EQ(fm.at(name), jet.at(name));
  }
}

TEST(GainCacheStrategyTest, AlgorithmOptionMapsRejectUnsupportedBackends) {
  EXPECT_FALSE(get_gain_cache_strategies().contains("jet"));
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
