#include <algorithm>
#include <array>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <tbb/global_control.h>

#include "dist/distributed_graph_builder.h"

#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/coarsening/clustering/lp/global_lp_clusterer.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/distributed_label_propagation.h"

#include "kaminpar-common/random.h"

namespace kaminpar::dist {

class GlobalLPPhaseTest : public ::testing::TestWithParam<bool> {};

TEST_P(GlobalLPPhaseTest, WeightedMultiChunkLabelsMatchOwnersAfterEveryReuse) {
  tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 1);
  const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);
  constexpr NodeID kNodesPerRank = 16;
  const GlobalNodeID global_n = size * kNodesPerRank;
  graph::Builder builder(MPI_COMM_WORLD);
  builder.initialize(kNodesPerRank);
  for (NodeID u = 0; u < kNodesPerRank; ++u) {
    const GlobalNodeID global_u = rank * kNodesPerRank + u;
    builder.create_node(1 + global_u % 4);
    builder.create_edge(3, (global_u + 1) % global_n);
    builder.create_edge(3, (global_u + global_n - 1) % global_n);
    if (size > 1) {
      builder.create_edge(7, (global_u + kNodesPerRank) % global_n);
      if (size > 2) {
        builder.create_edge(7, (global_u + global_n - kNodesPerRank) % global_n);
      }
    }
  }
  DistributedGraph graph(std::make_unique<DistributedCSRGraph>(builder.finalize()));
  Context ctx = create_default_context();
  ctx.parallel.num_threads = 1;
  ctx.parallel.num_mpis = size;
  ctx.coarsening.global_lp.num_iterations = 6;
  ctx.coarsening.global_lp.chunks.fixed_num_chunks = 8;
  ctx.coarsening.global_lp.sync_cluster_weights = GetParam();
  ctx.coarsening.global_lp.enforce_cluster_weights = GetParam();
  ctx.coarsening.global_lp.prevent_cyclic_moves = true;
  GlobalLPClusterer clusterer(ctx);
  clusterer.set_max_cluster_weight(global_n * 4);

  std::unordered_set<GlobalNodeID> initial_foreign_ids;
  for (const NodeID u : graph.ghost_nodes()) {
    initial_foreign_ids.insert(graph.local_to_global_node(u));
  }
  GlobalNodeID local_new_foreign_labels = 0;
  for (const int seed : {7, 8, 9}) {
    Random::instance().reinit(seed);
    StaticArray<GlobalNodeID> labels(graph.total_n(), static_array::noinit);
    clusterer.cluster(labels, graph);
    std::vector<GlobalNodeID> owned(labels.begin(), labels.begin() + graph.n());
    std::vector<GlobalNodeID> all_labels(global_n);
    MPI_Allgather(
        owned.data(),
        kNodesPerRank,
        mpi::type::get<GlobalNodeID>(),
        all_labels.data(),
        kNodesPerRank,
        mpi::type::get<GlobalNodeID>(),
        MPI_COMM_WORLD
    );
    for (const NodeID u : graph.all_nodes()) {
      EXPECT_LT(labels[u], global_n);
      EXPECT_EQ(labels[u], all_labels[graph.local_to_global_node(u)]);
      local_new_foreign_labels +=
          !graph.is_owned_global_node(labels[u]) && !initial_foreign_ids.contains(labels[u]);
    }
    if (rank == 0 && std::getenv("KAMINPAR_TEST_DUMP_GLOBAL_LP") != nullptr) {
      std::cout << "LP_LABELS sync=" << GetParam() << " seed=" << seed;
      for (const auto label : all_labels) {
        std::cout << ' ' << label;
      }
      std::cout << " rng=";
      for (int i = 0; i < 32; ++i) {
        std::cout << Random::instance().random_bool();
      }
      std::cout << '\n';
    }
  }
  const auto global_new_foreign_labels =
      mpi::allreduce(local_new_foreign_labels, MPI_SUM, MPI_COMM_WORLD);
  if (size >= 4) {
    EXPECT_GT(global_new_foreign_labels, 0u);
  }
}

INSTANTIATE_TEST_SUITE_P(SynchronizedAndLocal, GlobalLPPhaseTest, ::testing::Bool());

namespace {

struct PrefilterGraph {
  using NodeID = dist::NodeID;
  using NodeWeight = dist::NodeWeight;
  using EdgeID = dist::EdgeID;
  using EdgeWeight = dist::EdgeWeight;
  std::vector<std::pair<NodeID, EdgeWeight>> neighbors;

  NodeWeight node_weight(NodeID) const {
    return 1;
  }
  NodeID degree(NodeID) const {
    return static_cast<NodeID>(neighbors.size());
  }
  template <typename F> void adjacent_nodes(NodeID, F &&f) const {
    for (const auto [v, w] : neighbors) {
      if constexpr (std::is_invocable_v<F, NodeID, EdgeWeight>) {
        f(v, w);
      } else {
        f(v);
      }
    }
  }
  template <typename F> void adjacent_nodes(NodeID, NodeID limit, F &&f) const {
    for (std::size_t i = 0; i < std::min<std::size_t>(limit, neighbors.size()); ++i) {
      if constexpr (std::is_invocable_v<F, NodeID, EdgeWeight>) {
        f(neighbors[i].first, neighbors[i].second);
      } else {
        f(neighbors[i].first);
      }
    }
  }
};

// Fixed insertion order makes the actual Base selection loop's tie calls observable.
struct PrefilterRatings {
  std::vector<std::pair<BlockID, EdgeWeight>> values;
  std::function<void()> after_selection;
  EdgeWeight &operator[](BlockID block) {
    for (auto &[b, value] : values) {
      if (b == block) {
        return value;
      }
    }
    return values.emplace_back(block, 0).second;
  }
  const auto &entries() const {
    return values;
  }
  void clear() {
    values.clear();
    if (after_selection) {
      after_selection();
    }
  }
  template <typename F> auto execute(std::size_t, F &&f) {
    return f(*this);
  }
};

struct PrefilterGenericConfig : LabelPropagationConfig {
  using ClusterID = BlockID;
  using ClusterWeight = BlockWeight;
  using RatingMap = ::kaminpar::RatingMap<EdgeWeight, BlockID>;
  static constexpr bool kUseActiveSetStrategy = false;
};

template <bool Enabled, bool ActualGain = false> struct PrefilterConfig : PrefilterGenericConfig {
  static constexpr bool kSkipLosingClusterWeights = Enabled;
  static constexpr bool kUseActualGain = ActualGain;
};

template <typename Config>
struct PrefilterLP : LabelPropagation<PrefilterLP<Config>, Config, PrefilterGraph> {
  using Base = LabelPropagation<PrefilterLP<Config>, Config, PrefilterGraph>;
  using Base::find_best_cluster;
  using Base::handle_node;
  explicit PrefilterLP(const PrefilterGraph &graph) {
    this->_graph = &graph;
    this->_initial_num_clusters = 8;
  }
  std::array<BlockWeight, 8> weights{};
  std::array<BlockWeight, 8> last_weight{};
  std::array<int, 8> queries{};
  std::function<void(BlockID)> before_weight_read;
  BlockID label = 0;
  int move_attempts = 0;
  bool weight_only_acceptance = false;

  BlockID cluster(NodeID node) {
    return node == 0 ? label : node;
  }
  BlockWeight cluster_weight(BlockID block) {
    ++queries[block];
    if (before_weight_read) {
      before_weight_read(block);
    }
    return last_weight[block] = __atomic_load_n(&weights[block], __ATOMIC_RELAXED);
  }
  BlockWeight max_cluster_weight(BlockID) {
    return 20;
  }
  bool accept_cluster(const typename Base::ClusterSelectionState &state) {
    if (weight_only_acceptance) {
      return state.current_cluster != 0 && state.current_cluster_weight + state.u_weight <= 20;
    }
    return (state.current_gain > state.best_gain ||
            (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
           ((Config::kUseActualGain ? state.current_cluster_weight + state.u_weight < 20
                                    : state.current_cluster_weight + state.u_weight <= 20) ||
            state.current_cluster == state.initial_cluster);
  }
  bool move_cluster_weight(BlockID from, BlockID to, BlockWeight delta, BlockWeight maximum) {
    ++move_attempts;
    if (cluster_weight(to) + delta > maximum) {
      return false;
    }
    __atomic_fetch_sub(&weights[from], delta, __ATOMIC_RELAXED);
    __atomic_fetch_add(&weights[to], delta, __ATOMIC_RELAXED);
    return true;
  }
  void move_node(NodeID, BlockID block) {
    label = block;
  }
};

template <bool ActualGain> void expect_same_selection_and_rng() {
  const PrefilterGraph graph{{{0, 3}, {1, 8}, {2, 2}, {3, 8}, {4, 1}, {5, 12}, {6, 12}, {7, 4}}};
  for (int seed = 0; seed < 32; ++seed) {
    PrefilterLP<PrefilterConfig<false, ActualGain>> off(graph);
    PrefilterLP<PrefilterConfig<true, ActualGain>> on(graph);
    Random before(random::thread_independent_seeding), after(random::thread_independent_seeding);
    before.reinit(seed);
    after.reinit(seed);
    PrefilterRatings off_ratings, on_ratings;
    EXPECT_EQ(
        off.find_best_cluster(0, 1, 0, before, off_ratings),
        on.find_best_cluster(0, 1, 0, after, on_ratings)
    );
    for (int i = 0; i < 32; ++i) {
      EXPECT_EQ(before.random_bool(), after.random_bool());
    }
    EXPECT_EQ(off.queries[2], 1);
    EXPECT_EQ(on.queries[2], 0);
    EXPECT_EQ(off.queries[3], 1);
    EXPECT_EQ(on.queries[3], 1); // Equal-gain candidate stays live.
  }
}

TEST(LPGainPrefilterTest, LosingQueriesDisappearWithoutChangingTiesGainOrNextRng) {
  expect_same_selection_and_rng<false>();
  expect_same_selection_and_rng<true>();
}

template <bool Enabled, bool ActualGain> void expect_fresh_weights_and_capacity_recheck() {
  const PrefilterGraph graph{{{1, 4}, {2, 9}}};
  PrefilterLP<PrefilterConfig<Enabled, ActualGain>> lp(graph);
  lp.weights[2] = 20;
  lp.before_weight_read = [&](BlockID block) {
    if (block == 1) {
      __atomic_store_n(&lp.weights[2], 0, __ATOMIC_RELAXED);
    }
  };
  Random rand(random::thread_independent_seeding);
  rand.reinit(7);
  PrefilterRatings ratings;
  EXPECT_EQ(lp.find_best_cluster(0, 1, 0, rand, ratings).first, BlockID{2});
  EXPECT_EQ(lp.last_weight[2], 0); // Winner changed after the first competitive query.
  ratings.after_selection = [&] {
    __atomic_store_n(&lp.weights[2], 20, __ATOMIC_RELAXED);
  };
  EXPECT_FALSE(lp.handle_node(0, rand, ratings).first);
  EXPECT_EQ(lp.move_attempts, 1);
  EXPECT_EQ(lp.last_weight[2], 20); // Move still rechecks the now-full target.
  EXPECT_EQ(lp.label, 0u);
}

TEST(LPGainPrefilterTest, PotentialWinnersStayFreshAndMovesRecheckCapacity) {
  expect_fresh_weights_and_capacity_recheck<false, false>();
  expect_fresh_weights_and_capacity_recheck<true, false>();
  expect_fresh_weights_and_capacity_recheck<false, true>();
  expect_fresh_weights_and_capacity_recheck<true, true>();
}

TEST(LPGainPrefilterTest, GenericDefaultKeepsNonGainAcceptancePolicy) {
  static_assert(!PrefilterGenericConfig::kSkipLosingClusterWeights);
  const PrefilterGraph graph{{{1, 8}, {2, 2}}};
  PrefilterLP<PrefilterGenericConfig> lp(graph);
  lp.weight_only_acceptance = true;
  Random rand(random::thread_independent_seeding);
  rand.reinit(3);
  PrefilterRatings ratings;
  EXPECT_EQ(lp.find_best_cluster(0, 1, 0, rand, ratings).first, BlockID{2});
  EXPECT_EQ(lp.queries[2], 1); // A custom non-gain predicate must still see this candidate.
}

} // namespace

} // namespace kaminpar::dist
