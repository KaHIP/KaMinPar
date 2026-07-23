/*******************************************************************************
 * @file: meta_refiner_test.cc
 ******************************************************************************/
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "kaminpar-cli/kaminpar_arguments.h"
#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/coarsening/clustering/ensemble_clusterer.h"
#include "kaminpar-shm/factories.h"
#include "kaminpar-shm/refinement/meta_refiner.h"

namespace kaminpar::shm::testing {
namespace {

struct ClustererState {
  std::size_t calls = 0;
  std::vector<bool> free_memory_afterwards;
};

class SequenceClusterer final : public Clusterer {
public:
  SequenceClusterer(
      std::shared_ptr<ClustererState> state, std::vector<std::vector<NodeID>> clusterings
  )
      : _state(std::move(state)),
        _clusterings(std::move(clusterings)) {}

  void compute_clustering(
      StaticArray<NodeID> &clustering,
      [[maybe_unused]] const Graph &graph,
      const bool free_memory_afterwards
  ) final {
    const std::vector<NodeID> &next = _clusterings[_state->calls % _clusterings.size()];
    std::copy(next.begin(), next.end(), clustering.begin());
    ++_state->calls;
    _state->free_memory_afterwards.push_back(free_memory_afterwards);
  }

private:
  std::shared_ptr<ClustererState> _state;
  std::vector<std::vector<NodeID>> _clusterings;
};

struct RefinerState {
  std::vector<NodeID> initialized_graph_sizes;
  std::vector<NodeID> refined_graph_sizes;
  std::vector<std::vector<BlockID>> input_partitions;
};

class RecordingRefiner final : public Refiner {
public:
  explicit RecordingRefiner(std::shared_ptr<RefinerState> state) : _state(std::move(state)) {}

  [[nodiscard]] std::string name() const final {
    return "Recording Refiner";
  }

  void initialize(const PartitionedGraph &p_graph) final {
    _state->initialized_graph_sizes.push_back(p_graph.n());
  }

  bool refine(PartitionedGraph &p_graph, [[maybe_unused]] const PartitionContext &p_ctx) final {
    _state->refined_graph_sizes.push_back(p_graph.n());
    _state->input_partitions.emplace_back(
        p_graph.raw_partition().begin(), p_graph.raw_partition().end()
    );

    if (_state->refined_graph_sizes.size() == 1) {
      for (NodeID u = 0; u < p_graph.n(); ++u) {
        if (p_graph.block(u) == 0) {
          p_graph.set_block(u, 1);
        }
      }
    } else {
      p_graph.set_block(p_graph.n() - 1, 0);
    }
    return true;
  }

private:
  std::shared_ptr<RefinerState> _state;
};

TEST(EnsembleClustererTest, OverlaysAnyNumberOfClusterings) {
  const Graph graph = make_path_graph(6);
  auto state = std::make_shared<ClustererState>();
  EnsembleClusterer clusterer(
      std::make_unique<SequenceClusterer>(
          state,
          std::vector<std::vector<NodeID>>{
              {0, 0, 0, 3, 3, 3},
              {0, 0, 2, 2, 4, 4},
              {0, 1, 1, 3, 4, 4},
          }
      ),
      3
  );

  StaticArray<NodeID> clustering(graph.n(), static_array::noinit);
  clusterer.compute_clustering(clustering, graph, true);

  EXPECT_EQ(state->calls, 3);
  EXPECT_EQ(state->free_memory_afterwards, std::vector<bool>({false, false, true}));
  EXPECT_EQ(clustering[4], clustering[5]);
  for (NodeID u = 0; u < 4; ++u) {
    for (NodeID v = u + 1; v < graph.n(); ++v) {
      EXPECT_NE(clustering[u], clustering[v]);
    }
  }
}

TEST(MetaRefinerTest, RefinesContractedGraphBeforeCurrentGraph) {
  const Graph graph = make_path_graph(4);
  PartitionedGraph p_graph = make_p_graph(graph, 2, {0, 0, 1, 1});

  Context ctx = create_default_context();
  ctx.refinement.meta.num_clusterings = 3;

  PartitionContext p_ctx;
  p_ctx.setup(graph, 2, 0.03);

  auto clusterer_state = std::make_shared<ClustererState>();
  auto refiner_state = std::make_shared<RefinerState>();
  MetaRefiner refiner(
      ctx,
      std::make_unique<SequenceClusterer>(
          clusterer_state, std::vector<std::vector<NodeID>>{{0, 0, 0, 0}}
      ),
      std::make_unique<RecordingRefiner>(refiner_state)
  );

  refiner.initialize(p_graph);
  EXPECT_TRUE(refiner.refine(p_graph, p_ctx));

  EXPECT_EQ(clusterer_state->calls, 3);
  EXPECT_EQ(refiner_state->initialized_graph_sizes, std::vector<NodeID>({2, 4}));
  EXPECT_EQ(refiner_state->refined_graph_sizes, std::vector<NodeID>({2, 4}));
  ASSERT_EQ(refiner_state->input_partitions[0].size(), 2);
  EXPECT_NE(refiner_state->input_partitions[0][0], refiner_state->input_partitions[0][1]);
  EXPECT_EQ(refiner_state->input_partitions[1], std::vector<BlockID>({1, 1, 1, 1}));
  EXPECT_EQ(
      std::vector<BlockID>(p_graph.raw_partition().begin(), p_graph.raw_partition().end()),
      std::vector<BlockID>({1, 1, 1, 0})
  );
}

TEST(MetaRefinerTest, FactoryUsesLPEnsembleAndConfiguredRefiner) {
  const Graph graph = make_path_graph(6);
  PartitionedGraph p_graph = make_p_graph(graph, 2, {0, 0, 0, 1, 1, 1});

  Context ctx = create_default_context();
  ctx.refinement.algorithms = {RefinementAlgorithm::META};
  ctx.refinement.meta.num_clusterings = 3;
  ctx.refinement.meta.refiner = RefinementAlgorithm::NOOP;

  PartitionContext p_ctx;
  p_ctx.setup(graph, 2, 0.03);

  auto refiner = factory::create_refiner(ctx);
  refiner->initialize(p_graph);
  EXPECT_FALSE(refiner->refine(p_graph, p_ctx));
  EXPECT_EQ(
      std::vector<BlockID>(p_graph.raw_partition().begin(), p_graph.raw_partition().end()),
      std::vector<BlockID>({0, 0, 0, 1, 1, 1})
  );
}

TEST(MetaRefinerTest, SkipsCoarseRefinementIfTheEnsembleDoesNotShrinkTheGraph) {
  const Graph graph = make_path_graph(4);
  PartitionedGraph p_graph = make_p_graph(graph, 2, {0, 0, 1, 1});

  Context ctx = create_default_context();
  ctx.refinement.meta.num_clusterings = 2;

  PartitionContext p_ctx;
  p_ctx.setup(graph, 2, 0.03);

  auto clusterer_state = std::make_shared<ClustererState>();
  auto refiner_state = std::make_shared<RefinerState>();
  MetaRefiner refiner(
      ctx,
      std::make_unique<SequenceClusterer>(
          clusterer_state, std::vector<std::vector<NodeID>>{{0, 1, 2, 3}}
      ),
      std::make_unique<RecordingRefiner>(refiner_state)
  );

  refiner.initialize(p_graph);
  EXPECT_TRUE(refiner.refine(p_graph, p_ctx));

  EXPECT_EQ(clusterer_state->calls, 2);
  EXPECT_EQ(refiner_state->initialized_graph_sizes, std::vector<NodeID>({4}));
  EXPECT_EQ(refiner_state->refined_graph_sizes, std::vector<NodeID>({4}));
}

TEST(MetaRefinerTest, FactorySupportsTwoWayFlowAsConfiguredRefiner) {
  const Graph graph = make_path_graph(6);
  PartitionedGraph p_graph = make_p_graph(graph, 2, {0, 0, 0, 1, 1, 1});

  Context ctx = create_default_context();
  ctx.refinement.algorithms = {RefinementAlgorithm::META};
  ctx.refinement.meta.num_clusterings = 2;
  ctx.refinement.meta.refiner = RefinementAlgorithm::TWOWAY_FLOW;

  PartitionContext p_ctx;
  p_ctx.setup(graph, 2, 0.03);

  auto refiner = factory::create_refiner(ctx);
  EXPECT_EQ(refiner->name(), "Meta Refiner (Two-Way Flow Refinement)");
  refiner->initialize(p_graph);
  EXPECT_NO_THROW(refiner->refine(p_graph, p_ctx));
}

TEST(MetaRefinerCLITest, ParsesNumberOfClusteringsAndRefinerByName) {
  Context ctx = create_default_context();
  EXPECT_EQ(ctx.refinement.meta.num_clusterings, 8);
  EXPECT_EQ(ctx.refinement.meta.refiner, RefinementAlgorithm::UNCONSTRAINED_FM);

  CLI::App app;
  create_refinement_options(&app, ctx);

  app.parse("--r-meta-num-clusterings 5 --r-meta-refiner lp", false);

  EXPECT_EQ(ctx.refinement.meta.num_clusterings, 5);
  EXPECT_EQ(ctx.refinement.meta.refiner, RefinementAlgorithm::LABEL_PROPAGATION);
}

TEST(MetaRefinerCLITest, ParsesDistinctUFMAndFlowMetaRefinersInOneChain) {
  Context ctx = create_default_context();
  CLI::App app;
  create_refinement_options(&app, ctx);

  app.parse(
      "--r-algorithms overload-balancer unconstrained-lp meta-ufm overload-balancer "
      "meta-flow overload-balancer",
      false
  );

  EXPECT_EQ(
      ctx.refinement.algorithms,
      std::vector<RefinementAlgorithm>({
          RefinementAlgorithm::OVERLOAD_BALANCER,
          RefinementAlgorithm::UNCONSTRAINED_LABEL_PROPAGATION,
          RefinementAlgorithm::META_UNCONSTRAINED_FM,
          RefinementAlgorithm::OVERLOAD_BALANCER,
          RefinementAlgorithm::META_TWOWAY_FLOW,
          RefinementAlgorithm::OVERLOAD_BALANCER,
      })
  );

  auto refiner = factory::create_refiner(ctx);
  EXPECT_EQ(refiner->name(), "Multi Refiner");
}

TEST(MetaEcoPresetTest, ReplacesUnconstrainedFMWithMetaRefiner) {
  const Context ctx = create_context_by_preset_name("meta-eco");

  EXPECT_TRUE(get_preset_names().contains("meta-eco"));
  EXPECT_EQ(
      ctx.refinement.algorithms,
      std::vector<RefinementAlgorithm>({
          RefinementAlgorithm::OVERLOAD_BALANCER,
          RefinementAlgorithm::UNCONSTRAINED_LABEL_PROPAGATION,
          RefinementAlgorithm::META,
          RefinementAlgorithm::OVERLOAD_BALANCER,
      })
  );
  EXPECT_EQ(ctx.refinement.meta.refiner, RefinementAlgorithm::UNCONSTRAINED_FM);
  EXPECT_EQ(ctx.refinement.meta.num_clusterings, 8);
}

} // namespace
} // namespace kaminpar::shm::testing
