/*******************************************************************************
 * Label propagation clustering that only clusters node within a PE (i.e.,
 * ignores ghost nodes).
 *
 * @file:   local_lp_clusterer.cc
 * @author: Daniel Seemaier
 * @date:   30.09.2021
 ******************************************************************************/
#include "kaminpar-dist/coarsening/clustering/lp/local_lp_clusterer.h"

#include <functional>

#include "kaminpar-common/algorithms/label_propagation.h"
#include "kaminpar-common/datastructures/concurrent_fast_reset_array.h"
#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/parallel/iteration.h"

namespace kaminpar::dist {

namespace {

constexpr NodeID kMinChunkSize = 1024;
constexpr NodeID kPermutationSize = 64;
constexpr std::size_t kNumberOfNodePermutations = 64;

using LocalLPRatingMap = ::kaminpar::RatingMap<EdgeWeight, NodeID>;
using LocalLPGrowingRatingMap = DynamicRememberingFlatMap<NodeID, EdgeWeight>;
using LocalLPConcurrentRatingMap = ConcurrentFastResetArray<EdgeWeight, NodeID>;
using LocalLPWorkspace = lp::Workspace<
    NodeID,
    NodeID,
    EdgeWeight,
    LocalLPRatingMap,
    LocalLPGrowingRatingMap,
    LocalLPConcurrentRatingMap,
    false>;
using LocalLPOrderWorkspace =
    iteration::ChunkRandomNodeOrderWorkspace<NodeID, kPermutationSize, kNumberOfNodePermutations>;

class LocalLPWeights : public lp::RelaxedClusterWeightVector<NodeID, NodeWeight> {
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
  NodeWeight _max_cluster_weight = std::numeric_limits<NodeWeight>::max();
  std::function<NodeWeight(NodeID)> _initial_cluster_weight = [](NodeID) {
    return 0;
  };
};

template <typename Graph> class TypedLocalLPNeighborPolicy {
public:
  TypedLocalLPNeighborPolicy(
      const Graph &graph, const BlockID *partition, const bool ignore_ghost_nodes
  )
      : _graph(graph),
        _partition(partition),
        _ignore_ghost_nodes(ignore_ghost_nodes) {}

  [[nodiscard]] bool accept(const NodeID u, const NodeID v) const {
    return (_partition == nullptr || _partition[u] == _partition[v]) &&
           (!_ignore_ghost_nodes || _graph.is_owned_node(v));
  }

  [[nodiscard]] bool activate(const NodeID u) const {
    return _graph.is_owned_node(u);
  }

  [[nodiscard]] bool skip(const NodeID) const {
    return false;
  }

private:
  const Graph &_graph;
  const BlockID *_partition;
  bool _ignore_ghost_nodes;
};

class LocalLPSelector {
public:
  explicit LocalLPSelector(LocalLPWeights &weights) : _weights(weights) {}

  template <::kaminpar::lp::TieBreakingStrategy TieBreaking, typename Context, typename RatingMap>
  [[nodiscard]] KAMINPAR_LP_INLINE auto select(
      const Context &context,
      RatingMap &map,
      ScalableVector<NodeID> &tie_breaking_clusters,
      ScalableVector<NodeID> &tie_breaking_favored_clusters
  ) {
    return ::kaminpar::lp::choose_cluster<TieBreaking>(
        context, map, *this, tie_breaking_clusters, tie_breaking_favored_clusters
    );
  }

  [[nodiscard]] KAMINPAR_LP_INLINE NodeWeight cluster_weight(const NodeID cluster) const {
    return _weights.cluster_weight(cluster);
  }

  template <typename Context, typename Candidate, typename Choice>
  [[nodiscard]] KAMINPAR_LP_INLINE bool
  is_feasible(const Context &context, const Candidate &candidate, const Choice &) const {
    return candidate.weight + context.node_weight <=
               _weights.max_cluster_weight(candidate.cluster) ||
           candidate.cluster == context.initial_cluster;
  }

  template <typename Context, typename Candidate, typename Choice>
  [[nodiscard]] KAMINPAR_LP_INLINE ::kaminpar::lp::CandidateComparison
  compare(const Context &, const Candidate &candidate, const Choice &choice) const {
    return ::kaminpar::lp::compare_by_gain(candidate.gain, choice.best_gain);
  }

private:
  LocalLPWeights &_weights;
};

} // namespace

template <typename Graph> class LocalLPClusteringImpl final {
  SET_DEBUG(false);

public:
  LocalLPClusteringImpl(
      NodeID,
      const CoarseningContext &c_ctx,
      LocalLPWorkspace &workspace,
      LocalLPOrderWorkspace &order_workspace,
      LocalLPWeights &weights
  )
      : _workspace(workspace),
        _order_workspace(order_workspace),
        _weights(weights),
        _selector(_weights),
        _ignore_ghost_nodes(c_ctx.local_lp.ignore_ghost_nodes),
        _keep_ghost_clusters(c_ctx.local_lp.keep_ghost_clusters) {
    set_max_num_iterations(c_ctx.local_lp.num_iterations);
    _max_degree = c_ctx.local_lp.active_high_degree_threshold;
    _max_num_neighbors = c_ctx.local_lp.max_num_neighbors;
  }

  void preinitialize(const NodeID num_nodes) {
    _num_nodes = num_nodes;
  }

  void initialize(const Graph &graph) {
    _graph = &graph;
    _weights.allocate(graph.n());
    _weights.set_initial_cluster_weight([&](const NodeID cluster) {
      return graph.node_weight(cluster);
    });
  }

  void set_max_cluster_weight(const GlobalNodeWeight max_cluster_weight) {
    _weights.set_max_cluster_weight(max_cluster_weight);
  }

  void compute_clustering(StaticArray<NodeID> &clustering, const Graph &graph) {
    _labels.init(clustering);
    initialize(graph);

    lp::PassConfig<NodeID, NodeID> config{
        .nodes = {.max_degree = _max_degree, .max_neighbors = _max_num_neighbors},
        .rating = {.strategy = lp::RatingMapStrategy::SINGLE_PHASE},
        .active_set = {.strategy = lp::ActiveSetStrategy::NONE},
        .selection = {.tie_breaking_strategy = lp::TieBreakingStrategy::GEOMETRIC},
    };
    TypedLocalLPNeighborPolicy neighbors(graph, _partition, _ignore_ghost_nodes);
    lp::LabelPropagationCore core(
        graph, _labels, _weights, _selector, neighbors, _workspace, config
    );
    core.initialize(
        {.num_nodes = _num_nodes, .num_active_nodes = graph.n(), .num_clusters = graph.n()}
    );
    _order_workspace.clear_order();

    // initialize ghost nodes
    if (!_ignore_ghost_nodes) {
      init_ghost_nodes();
    }

    DBG << "Computing clustering on graph with " << graph.global_n()
        << " nodes (local: " << graph.n() << ", ghost: " << graph.ghost_n()
        << "), max cluster weight " << _weights.max_cluster_weight(0) << ", and at most "
        << _max_num_iterations << " iterations";

    std::size_t iteration;
    for (iteration = 0; iteration < _max_num_iterations; ++iteration) {
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
    }
    DBG << "Converged / stopped after " << iteration << " iterations";

    // dissolve all clusters owned by ghost nodes
    if (!_ignore_ghost_nodes) {
      if (_keep_ghost_clusters) {
        for (NodeID u : _graph->nodes()) {
          const NodeID u_cluster = _labels.cluster(u);
          if (_graph->is_ghost_node(u_cluster)) {
            // abuse cluster(u_cluster) to remap the whole cluster
            if (_graph->is_ghost_node(_labels.cluster(u_cluster))) {
              _labels.move_node(u_cluster, u);
            }

            _labels.move_node(u, _labels.cluster(u_cluster));
          }
        }
      } else {
        graph.pfor_nodes([&](const NodeID u) {
          if (_graph->is_ghost_node(_labels.cluster(u))) {
            _labels.move_node(u, u);
          }
        });
      }
    }
  }

  void set_max_num_iterations(const std::size_t max_num_iterations) {
    _max_num_iterations =
        (max_num_iterations == 0) ? std::numeric_limits<std::size_t>::max() : max_num_iterations;
  }

  void init_ghost_nodes() {
    tbb::parallel_for(_graph->n(), _graph->total_n(), [&](const NodeID ghost) {
      const NodeID cluster = ghost;
      _labels.init_cluster(ghost, cluster);
      _weights.init_cluster_weight(ghost, _graph->node_weight(cluster));
    });
  }

  const Graph *_graph = nullptr;
  LocalLPWorkspace &_workspace;
  LocalLPOrderWorkspace &_order_workspace;
  LocalLPWeights &_weights;
  lp::ExternalLabelArray<NodeID, NodeID> _labels;
  LocalLPSelector _selector;
  NodeID _max_degree = std::numeric_limits<NodeID>::max();
  NodeID _max_num_neighbors = std::numeric_limits<NodeID>::max();
  NodeID _num_nodes = 0;
  std::size_t _max_num_iterations;
  bool _ignore_ghost_nodes;
  bool _keep_ghost_clusters;

  const BlockID *_partition = nullptr;
};

class LocalLPClusteringImplWrapper {
public:
  LocalLPClusteringImplWrapper(const NodeID max_n, const CoarseningContext &c_ctx)
      : _csr_impl(
            std::make_unique<LocalLPClusteringImpl<DistributedCSRGraph>>(
                max_n, c_ctx, _workspace, _order_workspace, _weights
            )
        ),
        _compressed_impl(
            std::make_unique<LocalLPClusteringImpl<DistributedCompressedGraph>>(
                max_n, c_ctx, _workspace, _order_workspace, _weights
            )
        ) {}

  void set_communities(const StaticArray<BlockID> &communities) {
    _csr_impl->_partition = communities.data();
    _compressed_impl->_partition = communities.data();
  }

  void clear_communities() {
    _csr_impl->_partition = nullptr;
    _compressed_impl->_partition = nullptr;
  }

  void set_max_cluster_weight(const GlobalNodeWeight weight) {
    _csr_impl->set_max_cluster_weight(weight);
    _compressed_impl->set_max_cluster_weight(weight);
  }

  void compute_clustering(StaticArray<NodeID> &clustering, const DistributedGraph &graph) {
    const auto compute_clustering = [&](auto &impl, const auto &graph) {
      impl.compute_clustering(clustering, graph);
    };

    const NodeID num_nodes = graph.total_n();
    _csr_impl->preinitialize(num_nodes);
    _compressed_impl->preinitialize(num_nodes);

    graph.reified(
        [&](const DistributedCSRGraph &csr_graph) {
          LocalLPClusteringImpl<DistributedCSRGraph> &impl = *_csr_impl;
          compute_clustering(impl, csr_graph);
        },
        [&](const DistributedCompressedGraph &compressed_graph) {
          LocalLPClusteringImpl<DistributedCompressedGraph> &impl = *_compressed_impl;
          compute_clustering(impl, compressed_graph);
        }
    );
  }

private:
  LocalLPWorkspace _workspace;
  LocalLPOrderWorkspace _order_workspace;
  LocalLPWeights _weights;
  std::unique_ptr<LocalLPClusteringImpl<DistributedCSRGraph>> _csr_impl;
  std::unique_ptr<LocalLPClusteringImpl<DistributedCompressedGraph>> _compressed_impl;
};

//
// Interface
//

LocalLPClusterer::LocalLPClusterer(const Context &ctx)
    : _impl(
          std::make_unique<LocalLPClusteringImplWrapper>(
              ctx.coarsening.local_lp.ignore_ghost_nodes ? ctx.partition.n : ctx.partition.total_n,
              ctx.coarsening
          )
      ) {}

LocalLPClusterer::~LocalLPClusterer() = default;

void LocalLPClusterer::set_communities(const StaticArray<BlockID> &communities) {
  _impl->set_communities(communities);
}

void LocalLPClusterer::clear_communities() {
  _impl->clear_communities();
}

void LocalLPClusterer::set_max_cluster_weight(GlobalNodeWeight weight) {
  _impl->set_max_cluster_weight(weight);
}

void LocalLPClusterer::cluster(
    StaticArray<GlobalNodeID> &global_clustering, const DistributedGraph &p_graph
) {
  StaticArray<NodeID> local_clustering(
      p_graph.n(), reinterpret_cast<NodeID *>(global_clustering.data())
  );
  return _impl->compute_clustering(local_clustering, p_graph);
}

} // namespace kaminpar::dist
