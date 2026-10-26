#include <array>
#include <barrier>
#include <future>
#include <sstream>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

#include "tests/shm/graph_factories.h"

#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm::testing {
namespace {

template <typename T>
concept HasPartitionMember = requires(T value) { value.partition; };
static_assert(!HasPartitionMember<Context>);
static_assert(std::is_const_v<std::remove_reference_t<decltype(ExecutionContext::config)>>);

TEST(ShmRuntimeTest, CopiesForkSubproblemBoundsAndShareOnlyServices) {
  const auto graph = make_grid_graph(8, 8);
  const Context config = create_default_context();
  RuntimeContext runtime(42);
  ExecutionContext parent(config, runtime);
  parent.partition.setup(graph, 4, .1);
  ExecutionContext child = parent;
  child.partition.setup(graph, 2, .2);
  child.parallel.num_threads = 1;
  EXPECT_EQ(parent.partition.k, 4);
  EXPECT_EQ(child.partition.k, 2);
  EXPECT_EQ(&parent.config, &child.config);
  EXPECT_EQ(&parent.runtime, &child.runtime);
}

TEST(ShmRuntimeTest, RandomStreamsAreIsolatedOnTheSameThread) {
  RuntimeContext first(42), second(7), reference(42);
  for (int i = 0; i < 2000; ++i) {
    EXPECT_EQ(first.random.local().generator()(), reference.random.local().generator()());
    second.random.local().generator()();
    if (i % 71 == 0)
      second.random.reseed(i);
  }
}

TEST(ShmRuntimeTest, InactiveTimerScopeDoesNotPopAnActiveParent) {
  Timer timer("test");
  {
    auto parent = timer.start_scoped_timer("parent");
    timer.disable();
    {
      auto inactive = timer.start_scoped_timer("inactive");
      timer.enable();
    }
    auto child = timer.start_scoped_timer("child");
  }
  ASSERT_EQ(timer.tree().children.size(), 1);
  EXPECT_EQ(timer.tree().children.front()->name, "parent");
  ASSERT_EQ(timer.tree().children.front()->children.size(), 1);
  EXPECT_EQ(timer.tree().children.front()->children.front()->name, "child");
}

TEST(ShmRuntimeTest, TimedScopeOwnsItsDescription) {
  Timer timer("test");
  auto scope = timer::TimedScope<const std::string &>{&timer, "deferred"};
  auto described =
      timer::TimedScope<const std::string &>{&timer, "described", std::string(256, 'x')};
  std::vector<std::string> overwrite_stack(100, std::string(256, 'z'));
  EXPECT_EQ(scope + [] { return 7; }, 7);
  EXPECT_EQ(described + [] { return 9; }, 9);
#ifdef KAMINPAR_ENABLE_TIMERS
  ASSERT_EQ(timer.tree().children.size(), 2);
  EXPECT_TRUE(timer.tree().children[0]->description.empty());
  EXPECT_EQ(timer.tree().children[1]->description, std::string(256, 'x'));
#endif
}

TEST(ShmRuntimeTest, InstancesDoNotChangeTheProcessWorkerLimitOrGlobalLogger) {
  const auto limit =
      tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);
  const bool quiet = Logger::is_quiet();
  KaMinPar one(1, create_default_context());
  KaMinPar four(4, create_default_context());
  one.set_output_level(OutputLevel::QUIET);
  EXPECT_EQ(tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism), limit);
  EXPECT_EQ(Logger::is_quiet(), quiet);
  std::ostringstream hidden, visible;
  Logger(one.runtime().quiet.load(), hidden) << "hidden";
  Logger(four.runtime().quiet.load(), visible) << "visible";
  EXPECT_TRUE(hidden.str().empty());
  EXPECT_EQ(visible.str(), "visible\n");
}

struct Result {
  EdgeWeight cut;
  std::vector<BlockID> partition;
};

Result partition_graph(KaMinPar &partitioner, const Graph &graph, BlockID k = 4) {
  const auto &csr = graph.csr_graph();
  partitioner.copy_graph(csr.raw_nodes().view(), csr.raw_edges().view());
  partitioner.set_k(k);
  partitioner.set_uniform_max_block_weights(.25);
  Result result{0, std::vector<BlockID>(graph.n())};
  result.cut = partitioner.compute_partition(result.partition);
  EdgeWeight cut = 0;
  std::vector<BlockWeight> weights(k);
  for (NodeID u = 0; u < graph.n(); ++u) {
    EXPECT_LT(result.partition[u], k);
    if (result.partition[u] >= k)
      continue;
    ++weights[result.partition[u]];
    csr.adjacent_nodes(u, [&](NodeID v) { cut += result.partition[u] != result.partition[v]; });
  }
  EXPECT_EQ(result.cut, cut / 2);
  for (BlockID b = 0; b < k; ++b) {
    EXPECT_LE(weights[b], partitioner.partition_context().max_block_weight(b));
  }
  return result;
}

TEST(ShmConcurrencyTest, ConcurrentSeededCallsMatchTheirIndependentSerialRuns) {
  const auto graph = make_grid_graph(16, 24);
  constexpr std::size_t count = 3;
  constexpr std::array labels{"first", "second", "third"};
  std::array<Result, count> serial;
  for (std::size_t i = 0; i < count; ++i) {
    Context config = create_default_context();
    config.seed = 17 + i;
    config.coarsening.contraction_limit = 16;
    KaMinPar partitioner(1, config);
    partitioner.set_output_level(OutputLevel::QUIET);
    serial[i] = partition_graph(partitioner, graph);
  }

  std::barrier start(static_cast<std::ptrdiff_t>(count));
  std::array<std::future<void>, count> futures;
  for (std::size_t i = 0; i < count; ++i) {
    futures[i] = std::async(std::launch::async, [&, i] {
      Context config = create_default_context();
      config.seed = 17 + i;
      config.coarsening.contraction_limit = 16;
      KaMinPar partitioner(1, config);
      partitioner.set_output_level(OutputLevel::QUIET);
      {
        auto marker = partitioner.runtime().timer.start_scoped_timer(labels[i]);
      }
      start.arrive_and_wait();
      for (int repetition = 0; repetition < 3; ++repetition) {
        const auto result = partition_graph(partitioner, graph);
        EXPECT_EQ(result.cut, serial[i].cut);
        EXPECT_EQ(result.partition, serial[i].partition);
        EXPECT_EQ(partitioner.context().seed, config.seed);
        EXPECT_EQ(partitioner.context().node_ordering, config.node_ordering);
        EXPECT_EQ(partitioner.context().parallel.num_threads, config.parallel.num_threads);
        if (repetition == 0) {
          for (std::size_t other = 0; other < count; ++other) {
            EXPECT_EQ(
                partitioner.runtime().timer.tree().children_tbl.contains(labels[other]), i == other
            );
          }
        }
      }
    });
  }
  for (auto &future : futures)
    future.get();
}

TEST(ShmConcurrencyTest, ParallelSubproblemsAndPresetsCanOverlapInAHostArena) {
  const auto graph = make_grid_graph(32, 32);
  const std::array configs{
      create_default_context(),
      create_eco_context(),
      create_largek_context(),
      create_vcycle_context(),
      create_jet_context(2)
  };
  tbb::task_arena host(4);
  host.execute([&] {
    tbb::parallel_for(std::size_t{0}, configs.size(), [&](std::size_t i) {
      auto config = configs[i];
      config.seed = 91 + i;
      config.coarsening.contraction_limit = 16;
      config.refinement.kway_fm.num_iterations = 2;
      if (i == 3)
        config.partitioning.vcycles = {2};
      KaMinPar partitioner(2 + i % 2, config);
      partitioner.set_output_level(OutputLevel::QUIET);
      partition_graph(partitioner, graph, 8);
#ifdef KAMINPAR_ENABLE_TIMERS
      EXPECT_FALSE(partitioner.runtime().timer.tree().children.empty());
#endif
    });
  });
}

TEST(ShmConcurrencyTest, MovingAnIdleInstancePreservesItsRuntime) {
  const auto graph = make_grid_graph(8, 8);
  KaMinPar original(1, create_default_context());
  original.set_output_level(OutputLevel::QUIET);
  auto *runtime = &original.runtime();
  KaMinPar moved(std::move(original));
  EXPECT_EQ(&moved.runtime(), runtime);
  partition_graph(moved, graph);
  KaMinPar assigned;
  assigned = std::move(moved);
  EXPECT_EQ(&assigned.runtime(), runtime);
  partition_graph(assigned, graph);
}

TEST(ShmConcurrencyTest, AFailedCallDoesNotPoisonItsNextRunOrAnotherInstance) {
  const auto graph = make_grid_graph(8, 8);
  KaMinPar first(1, create_default_context()), second(2, create_default_context());
  first.set_output_level(OutputLevel::QUIET);
  second.set_output_level(OutputLevel::QUIET);
  std::vector<BlockID> partition(graph.n());
  EXPECT_THROW(first.compute_partition(partition), std::invalid_argument);
  partition_graph(second, graph);
  partition_graph(first, graph);
}

} // namespace
} // namespace kaminpar::shm::testing
