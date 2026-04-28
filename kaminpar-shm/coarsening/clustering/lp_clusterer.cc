/******************************************************************************
 * Label propagation for graph coarsening / clustering.
 *
 * @file:   lp_clusterer.cc
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#include "kaminpar-shm/coarsening/clustering/lp_clusterer.h"

#include <functional>
#include <span>

#include "kaminpar-common/algorithms/label_propagation.h"
#include "kaminpar-common/datastructures/concurrent_fast_reset_array.h"
#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/parallel/iteration.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

//
// Actual implementation -- not exposed in header
//

namespace {

constexpr NodeID kMinChunkSize = 1024;
constexpr NodeID kPermutationSize = 64;
constexpr std::size_t kNumberOfNodePermutations = 64;
constexpr std::size_t kRatingMapThreshold = 10000;

using LPClusterRatingMap = ::kaminpar::RatingMap<EdgeWeight, NodeID>;
using LPClusterGrowingRatingMap = DynamicRememberingFlatMap<NodeID, EdgeWeight>;
using LPClusterConcurrentRatingMap = ConcurrentFastResetArray<EdgeWeight, NodeID>;
using LPClusterWorkspace = lp::Workspace<
    NodeID,
    NodeID,
    EdgeWeight,
    LPClusterRatingMap,
    LPClusterGrowingRatingMap,
    LPClusterConcurrentRatingMap,
    true>;
using LPClusterOrderWorkspace =
    iteration::ChunkRandomNodeOrderWorkspace<NodeID, kPermutationSize, kNumberOfNodePermutations>;

lp::RatingMapStrategy map_rating_map_strategy(const LabelPropagationImplementation impl) {
  switch (impl) {
  case LabelPropagationImplementation::SINGLE_PHASE:
    return lp::RatingMapStrategy::SINGLE_PHASE;
  case LabelPropagationImplementation::TWO_PHASE:
    return lp::RatingMapStrategy::TWO_PHASE;
  case LabelPropagationImplementation::GROWING_HASH_TABLES:
    return lp::RatingMapStrategy::GROWING_HASH_TABLES;
  }
  __builtin_unreachable();
}

lp::TieBreakingStrategy map_tie_breaking_strategy(const TieBreakingStrategy strategy) {
  switch (strategy) {
  case TieBreakingStrategy::GEOMETRIC:
    return lp::TieBreakingStrategy::GEOMETRIC;
  case TieBreakingStrategy::UNIFORM:
    return lp::TieBreakingStrategy::UNIFORM;
  }
  __builtin_unreachable();
}

class LPClusterWeights : public lp::RelaxedClusterWeightVector<NodeID, NodeWeight> {
public:
  void set_max_cluster_weight(const NodeWeight max_cluster_weight) {
    _max_cluster_weight = max_cluster_weight;
  }

  [[nodiscard]] NodeWeight max_cluster_weight(const NodeID) const {
    return _max_cluster_weight;
  }

  void set_initial_cluster_weight(std::function<NodeWeight(NodeID)> initial_cluster_weight) {
    _initial_cluster_weight = std::move(initial_cluster_weight);
  }

  [[nodiscard]] NodeWeight initial_cluster_weight(const NodeID cluster) const {
    return _initial_cluster_weight(cluster);
  }

private:
  NodeWeight _max_cluster_weight = kInvalidBlockWeight;
  std::function<NodeWeight(NodeID)> _initial_cluster_weight = [](NodeID) {
    return 0;
  };
};

struct LPClusteringNeighborPolicy {
  std::span<const NodeID> communities;

  [[nodiscard]] bool accept(const NodeID u, const NodeID v) const {
    return communities.empty() || communities[u] == communities[v];
  }

  [[nodiscard]] bool activate(const NodeID) const {
    return true;
  }

  [[nodiscard]] bool skip(const NodeID) const {
    return false;
  }
};

class LPClusteringSelector {
public:
  LPClusteringSelector(LPClusterWeights &weights, std::span<const NodeID> &communities)
      : _weights(weights),
        _communities(communities) {}

  template <lp::TieBreakingStrategy TieBreaking, typename Context, typename RatingMap>
  [[nodiscard]] KAMINPAR_LP_INLINE auto select(
      const Context &context,
      RatingMap &map,
      ScalableVector<NodeID> &tie_breaking_clusters,
      ScalableVector<NodeID> &tie_breaking_favored_clusters
  ) {
    auto choice = lp::make_initial_choice(context);
    const NodeWeight max_cluster_weight = _weights.max_cluster_weight(context.initial_cluster);
    const bool use_communities = !_communities.empty();
    const NodeID initial_community =
        use_communities ? _communities[context.initial_cluster] : kInvalidNodeID;

    for (const auto [cluster, rating] : map.entries()) {
      const NodeID current_cluster = cluster;
      const EdgeWeight current_gain = rating - context.gain_delta;
      const NodeWeight current_cluster_weight = _weights.cluster_weight(current_cluster);

      if (context.track_favored_cluster) {
        if constexpr (TieBreaking == lp::TieBreakingStrategy::UNIFORM) {
          if (current_gain > choice.favored_gain) {
            choice.favored_gain = current_gain;
            choice.favored_cluster = current_cluster;

            tie_breaking_favored_clusters.clear();
            tie_breaking_favored_clusters.push_back(current_cluster);
          } else if (current_gain == choice.favored_gain) {
            tie_breaking_favored_clusters.push_back(current_cluster);
          }
        } else {
          if (current_gain > choice.favored_gain) {
            choice.favored_gain = current_gain;
            choice.favored_cluster = current_cluster;
          }
        }
      }

      if constexpr (TieBreaking == lp::TieBreakingStrategy::UNIFORM) {
        if (current_gain > choice.best_gain) {
          if ((current_cluster_weight + context.node_weight <= max_cluster_weight ||
               current_cluster == context.initial_cluster) &&
              (!use_communities || _communities[current_cluster] == initial_community)) {
            choice.best_cluster = current_cluster;
            choice.best_gain = current_gain;
            choice.best_cluster_weight = current_cluster_weight;

            tie_breaking_clusters.clear();
            tie_breaking_clusters.push_back(current_cluster);
          }
        } else if (current_gain == choice.best_gain) {
          if ((current_cluster_weight + context.node_weight <= max_cluster_weight ||
               current_cluster == context.initial_cluster) &&
              (!use_communities || _communities[current_cluster] == initial_community)) {
            tie_breaking_clusters.push_back(current_cluster);
          }
        }
      } else {
        if ((current_gain > choice.best_gain ||
             (current_gain == choice.best_gain && context.rand.random_bool())) &&
            (current_cluster_weight + context.node_weight <= max_cluster_weight ||
             current_cluster == context.initial_cluster) &&
            (!use_communities || _communities[current_cluster] == initial_community)) {
          choice.best_cluster = current_cluster;
          choice.best_gain = current_gain;
          choice.best_cluster_weight = current_cluster_weight;
        }
      }
    }

    if constexpr (TieBreaking == lp::TieBreakingStrategy::UNIFORM) {
      if (tie_breaking_clusters.size() > 1) {
        const NodeID i = context.rand.random_index(0, tie_breaking_clusters.size());
        choice.best_cluster = tie_breaking_clusters[i];
      }
      tie_breaking_clusters.clear();

      if (tie_breaking_favored_clusters.size() > 1) {
        const NodeID i = context.rand.random_index(0, tie_breaking_favored_clusters.size());
        choice.favored_cluster = tie_breaking_favored_clusters[i];
      }
      tie_breaking_favored_clusters.clear();
    }

    return choice;
  }

private:
  LPClusterWeights &_weights;
  std::span<const NodeID> &_communities;
};

} // namespace

template <typename Graph> class LPClusteringImpl final {
  SET_DEBUG(false);

public:
  LPClusteringImpl(
      const CoarseningContext &c_ctx,
      LPClusterWorkspace &workspace,
      LPClusterOrderWorkspace &order_workspace,
      LPClusterWeights &weights
  )
      : _lp_ctx(c_ctx.clustering.lp),
        _workspace(workspace),
        _order_workspace(order_workspace),
        _weights(weights),
        _selector(_weights, _communities),
        _relabel_before_second_phase(c_ctx.clustering.lp.relabel_before_second_phase) {}

  void set_max_cluster_weight(const NodeWeight max_cluster_weight) {
    _weights.set_max_cluster_weight(max_cluster_weight);
  }

  void set_communities(const std::span<const NodeID> communities) {
    _communities = communities;
  }

  void reset_communities() {
    _communities = {};
  }

  void preinitialize(const NodeID num_nodes) {
    _num_nodes = num_nodes;
  }

  void allocate(const NodeID num_clusters) {
    SCOPED_HEAP_PROFILER("Allocation");
    SCOPED_TIMER("Allocation");

    _weights.allocate(num_clusters);
  }

  void free() {
    SCOPED_HEAP_PROFILER("Free");
    SCOPED_TIMER("Free");

    _workspace.free();
    _order_workspace.free();
    _weights.free();
  }

  void compute_clustering(StaticArray<NodeID> &clustering, const Graph &graph) {
    _labels.init(clustering);
    _weights.set_initial_cluster_weight([&](const NodeID cluster) {
      return graph.node_weight(cluster);
    });
    _order_workspace.clear_order();

    lp::PassConfig<NodeID, NodeID> config{
        .nodes =
            {.max_degree = _lp_ctx.large_degree_threshold,
             .max_neighbors = _lp_ctx.max_num_neighbors},
        .rating =
            {.strategy = map_rating_map_strategy(_lp_ctx.impl),
             .large_map_threshold = kRatingMapThreshold,
             .relabel_before_second_phase = _relabel_before_second_phase},
        .active_set = {.strategy = lp::ActiveSetStrategy::GLOBAL},
        .selection =
            {.tie_breaking_strategy = map_tie_breaking_strategy(_lp_ctx.tie_breaking_strategy),
             .use_actual_gain = false,
             .track_favored_clusters = true},
        .stopping = {.desired_clusters = _desired_num_clusters, .track_cluster_count = true},
    };

    LPClusteringNeighborPolicy neighbors{.communities = _communities};
    lp::LabelPropagationCore core(
        graph, _labels, _weights, _selector, neighbors, _workspace, config
    );
    core.initialize(
        {.num_nodes = _num_nodes, .num_active_nodes = graph.n(), .num_clusters = graph.n()}
    );

    for (std::size_t iteration = 0; iteration < _lp_ctx.num_iterations; ++iteration) {
      SCOPED_TIMER("Iteration", std::to_string(iteration));
      iteration::ChunkRandomNodeOrder order(
          graph,
          _order_workspace,
          iteration::NodeRange<NodeID>{0, graph.n()},
          static_cast<EdgeID>(kMinChunkSize),
          iteration::bucket_limit_for_max_degree(graph, config.nodes.max_degree)
      );
      const auto result = lp::run_iteration(order, core);
      if (result.moved_nodes == 0) {
        break;
      }

      // Only relabel during the first iteration because afterwards the memory for the second phase
      // is already allocated.
      if (iteration == 0) {
        config.rating.relabel_before_second_phase = false;
        core.set_config(config);
        _relabel_before_second_phase = false;
      }
    }

    cluster_isolated_nodes(core, graph);
    cluster_two_hop_nodes(core, graph);
  }

  void set_desired_num_clusters(const NodeID count) {
    _desired_num_clusters = count;
  }

  void set_relabel_before_second_phase(const bool relabel) {
    _relabel_before_second_phase = relabel;
  }

private:
  template <typename Core> void cluster_two_hop_nodes(Core &core, const Graph &graph) {
    SCOPED_HEAP_PROFILER("Handle two-hop nodes");
    SCOPED_TIMER("Handle two-hop nodes");

    if (!should_handle_two_hop_nodes(core, graph)) {
      return;
    }

    switch (_lp_ctx.two_hop_strategy) {
    case TwoHopStrategy::MATCH:
      core.match_two_hop_nodes();
      break;
    case TwoHopStrategy::MATCH_THREADWISE:
      core.match_two_hop_nodes_threadwise();
      break;
    case TwoHopStrategy::CLUSTER:
      core.cluster_two_hop_nodes();
      break;
    case TwoHopStrategy::CLUSTER_THREADWISE:
      core.cluster_two_hop_nodes_threadwise();
      break;
    case TwoHopStrategy::DISABLE:
      break;
    }
  }

  template <typename Core> void cluster_isolated_nodes(Core &core, const Graph &graph) {
    SCOPED_HEAP_PROFILER("Handle isolated nodes");
    SCOPED_TIMER("Handle isolated nodes");

    switch (_lp_ctx.isolated_nodes_strategy) {
    case IsolatedNodesClusteringStrategy::MATCH:
      core.match_isolated_nodes();
      break;
    case IsolatedNodesClusteringStrategy::CLUSTER:
      core.cluster_isolated_nodes();
      break;
    case IsolatedNodesClusteringStrategy::MATCH_DURING_TWO_HOP:
      if (should_handle_two_hop_nodes(core, graph)) {
        core.match_isolated_nodes();
      }
      break;
    case IsolatedNodesClusteringStrategy::CLUSTER_DURING_TWO_HOP:
      if (should_handle_two_hop_nodes(core, graph)) {
        core.cluster_isolated_nodes();
      }
      break;
    case IsolatedNodesClusteringStrategy::KEEP:
      break;
    }
  }

  template <typename Core>
  [[nodiscard]] bool should_handle_two_hop_nodes(Core &core, const Graph &graph) const {
    return (1.0 - 1.0 * core.current_num_clusters() / graph.n()) <= _lp_ctx.two_hop_threshold;
  }

  const LabelPropagationCoarseningContext &_lp_ctx;
  LPClusterWorkspace &_workspace;
  LPClusterOrderWorkspace &_order_workspace;
  LPClusterWeights &_weights;
  lp::ExternalLabelArray<NodeID, NodeID> _labels;
  LPClusteringSelector _selector;

  std::span<const NodeID> _communities;
  NodeID _num_nodes = 0;
  NodeID _desired_num_clusters = 0;
  bool _relabel_before_second_phase;
};

class LPClusteringImplWrapper {
public:
  LPClusteringImplWrapper(const CoarseningContext &c_ctx)
      : _csr_impl(
            std::make_unique<LPClusteringImpl<CSRGraph>>(
                c_ctx, _workspace, _order_workspace, _weights
            )
        ),
        _compressed_impl(
            std::make_unique<LPClusteringImpl<CompressedGraph>>(
                c_ctx, _workspace, _order_workspace, _weights
            )
        ) {}

  void set_max_cluster_weight(const NodeWeight max_cluster_weight) {
    _csr_impl->set_max_cluster_weight(max_cluster_weight);
    _compressed_impl->set_max_cluster_weight(max_cluster_weight);
  }

  void set_desired_cluster_count(const NodeID count) {
    _csr_impl->set_desired_num_clusters(count);
    _compressed_impl->set_desired_num_clusters(count);
  }

  void set_communities(std::span<const NodeID> communities) {
    _csr_impl->set_communities(communities);
    _compressed_impl->set_communities(communities);
  }

  void compute_clustering(
      StaticArray<NodeID> &clustering, const Graph &graph, const bool free_memory_afterwards
  ) {
    const auto compute_clustering = [&](auto &core, auto &graph) {
      if (_freed) {
        _freed = false;
        core.allocate(graph.n());
      }

      core.compute_clustering(clustering, graph);

      if (free_memory_afterwards) {
        _freed = true;
        core.free();
      }
    };

    const NodeID num_nodes = graph.n();
    _csr_impl->preinitialize(num_nodes);
    _compressed_impl->preinitialize(num_nodes);

    reified(
        graph,
        [&](const auto &csr_graph) {
          LPClusteringImpl<CSRGraph> &impl = *_csr_impl;
          compute_clustering(impl, csr_graph);
        },
        [&](const auto &compressed_graph) {
          LPClusteringImpl<CompressedGraph> &impl = *_compressed_impl;
          compute_clustering(impl, compressed_graph);
        }
    );

    // Only relabel clusters for the first iteration
    _csr_impl->set_relabel_before_second_phase(false);
    _compressed_impl->set_relabel_before_second_phase(false);
  }

private:
  std::unique_ptr<LPClusteringImpl<CSRGraph>> _csr_impl;
  std::unique_ptr<LPClusteringImpl<CompressedGraph>> _compressed_impl;

  bool _freed = true;
  LPClusterWorkspace _workspace;
  LPClusterOrderWorkspace _order_workspace;
  LPClusterWeights _weights;
};

//
// Exposed wrapper
//

LPClustering::LPClustering(const CoarseningContext &c_ctx)
    : _impl_wrapper(std::make_unique<LPClusteringImplWrapper>(c_ctx)) {}

// we must declare the destructor explicitly here, otherwise, it is implicitly
// generated before LPClusteringImplWrapper is complete
LPClustering::~LPClustering() = default;

void LPClustering::set_max_cluster_weight(const NodeWeight max_cluster_weight) {
  _impl_wrapper->set_max_cluster_weight(max_cluster_weight);
}

void LPClustering::set_desired_cluster_count(const NodeID count) {
  _impl_wrapper->set_desired_cluster_count(count);
}

void LPClustering::set_communities(std::span<const NodeID> communities) {
  _impl_wrapper->set_communities(communities);
}

void LPClustering::compute_clustering(
    StaticArray<NodeID> &clustering, const Graph &graph, const bool free_memory_afterwards
) {
  return _impl_wrapper->compute_clustering(clustering, graph, free_memory_afterwards);
}

} // namespace kaminpar::shm
