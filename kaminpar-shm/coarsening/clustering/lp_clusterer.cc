/******************************************************************************
 * Label propagation for graph coarsening / clustering.
 *
 * @file:   lp_clusterer.cc
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#include "kaminpar-shm/coarsening/clustering/lp_clusterer.h"

#include <span>

#include "kaminpar-shm/label_propagation.h"

#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

//
// Actual implementation -- not exposed in header
//

template <typename NeighborhoodSampler_> struct LPClusteringConfig : public LabelPropagationConfig {
  using ClusterID = NodeID;
  using ClusterWeight = BlockWeight;
  static constexpr bool kTrackClusterCount = true;
  static constexpr bool kUseTwoHopClustering = true;
  using NeighborhoodSampler = NeighborhoodSampler_;
};

struct AllNeighborsSampler {
  void init(const Context &, const CSRGraph &) {}

  bool accept(NodeID, EdgeID, NodeID) {
    return true;
  }

  NodeID skip(NodeID, EdgeID, NodeID) {
    return 1;
  }
};

struct AvgDegreeSampler {
  constexpr static std::size_t kPeriode = 1023;

  AvgDegreeSampler() {
    _precomputed_doubles.resize(kPeriode + 1);
    for (double &d : _precomputed_doubles) {
      d = _rand.random_real();
    }
  }

  void init(const Context &ctx, const CSRGraph &graph) {
    _graph = &graph;
    _avg_deg = 1.0 * _graph->m() / _graph->n();
    _target = _avg_deg * ctx.coarsening.clustering.lp.neighborhood_sampling_avg_degree_threshold;
    _next = 0;
  }

  bool accept(NodeID, EdgeID, NodeID) {
    /*
    const EdgeID degree = _graph->degree(u);
    if (degree <= _target) {
      return true;
    }

    // return _rand.random_bool(1.0 * _target / degree);
    return _precomputed_doubles[_next++ & kPeriode)] <= 1.0 * _target / degree;
    */

    return true;
  }

  NodeID skip(const NodeID u, EdgeID, NodeID) {
    const EdgeID degree = _graph->degree(u);

    if (degree <= _target) {
      return 1;
    }

    const double p = 1.0 * _target / degree; 
    if (p == 0.0) {
      return _graph->n();
    }

    const double logp = std::log(1.0 - p); 
    return 1 + static_cast<NodeID>(std::ceil(std::log(1.0 - next()) / logp));
  }

  double next() {
    return _precomputed_doubles[_next++ & kPeriode];
  }

  // Make *this a random generator:
  double min() {
    return 0;
  }
  double max() {
    return 1;
  }
  double operator()() {
    return next();
  }

  std::vector<double> _precomputed_doubles;
  std::size_t _next = 0;

  Random &_rand = Random::instance();

  const CSRGraph *_graph;
  double _avg_deg = kInvalidEdgeID;
  double _target = kInvalidEdgeID;
  //NodeID _last_deg = kInvalidNodeID;
  //NodeID _last_u = kInvalidNodeID;
};

template <typename Graph, typename NeighborhoodSampler = void>
class LPClusteringImpl final : public ChunkRandomLabelPropagation<
                                   LPClusteringImpl<Graph, NeighborhoodSampler>,
                                   LPClusteringConfig<NeighborhoodSampler>,
                                   Graph>,
                               public OwnedRelaxedClusterWeightVector<NodeID, NodeWeight>,
                               public NonatomicClusterVectorRef<NodeID, NodeID> {
  SET_DEBUG(false);
  SET_STATISTICS_FROM_GLOBAL();

  using Base =
      ChunkRandomLabelPropagation<LPClusteringImpl, LPClusteringConfig<NeighborhoodSampler>, Graph>;
  using ClusterWeightBase = OwnedRelaxedClusterWeightVector<NodeID, NodeWeight>;
  using ClusterBase = NonatomicClusterVectorRef<NodeID, NodeID>;

  using Config = LPClusteringConfig<NeighborhoodSampler>;
  using ClusterID = Config::ClusterID;

public:
  using Permutations = Base::Permutations;

  LPClusteringImpl(const Context &ctx, Permutations &permutations) : Base(permutations), _ctx(ctx) {
    Base::set_max_degree(_ctx.coarsening.clustering.lp.large_degree_threshold);
    Base::set_max_num_neighbors(_ctx.coarsening.clustering.lp.max_num_neighbors);
    Base::set_implementation(_ctx.coarsening.clustering.lp.impl);
    Base::set_tie_breaking_strategy(_ctx.coarsening.clustering.lp.tie_breaking_strategy);
    Base::set_relabel_before_second_phase(_ctx.coarsening.clustering.lp.relabel_before_second_phase
    );
  }

  void set_max_cluster_weight(const NodeWeight max_cluster_weight) {
    _max_cluster_weight = max_cluster_weight;
  }

  void set_communities(const std::span<const NodeID> communities) {
    _communities = communities;
  }

  void reset_communities() {
    _communities = {};
  }

  void preinitialize(const NodeID num_nodes) {
    Base::preinitialize(num_nodes, num_nodes);
  }

  void allocate(const NodeID num_clusters) {
    SCOPED_HEAP_PROFILER("Allocation");
    SCOPED_TIMER("Allocation");

    Base::allocate();
    ClusterWeightBase::allocate_cluster_weights(num_clusters);
  }

  void free() {
    SCOPED_HEAP_PROFILER("Free");
    SCOPED_TIMER("Free");

    Base::free();
    ClusterWeightBase::free();
  }

  void compute_clustering(StaticArray<NodeID> &clustering, const Graph &graph) {
    ClusterWeightBase::reset_cluster_weights();
    ClusterBase::init_clusters_ref(clustering);
    Base::initialize(_ctx, &graph, graph.n());

    for (std::size_t iteration = 0; iteration < _ctx.coarsening.clustering.lp.num_iterations;
         ++iteration) {
      SCOPED_TIMER("Iteration", std::to_string(iteration));
      if (Base::perform_iteration() == 0) {
        break;
      }

      // Only relabel during the first iteration because afterwards the memory for the second phase
      // is already allocated.
      if (iteration == 0) {
        Base::set_relabel_before_second_phase(false);
      }
    }

    cluster_isolated_nodes();
    cluster_two_hop_nodes();

    // STATS << "Visited " << Base::num_accessed_neighbors() << " neighbors and skipped "
    //       << Base::num_skipped_neighbors();
  }

  NodeID num_skipped() {
    return Base::num_skipped_neighbors();
  }

  NodeID num_visited() {
    return Base::num_accessed_neighbors();
  }

private:
  void cluster_two_hop_nodes() {
    SCOPED_HEAP_PROFILER("Handle two-hop nodes");
    SCOPED_TIMER("Handle two-hop nodes");

    if (!should_handle_two_hop_nodes()) {
      return;
    }

    switch (_ctx.coarsening.clustering.lp.two_hop_strategy) {
    case TwoHopStrategy::MATCH:
      Base::match_two_hop_nodes();
      break;
    case TwoHopStrategy::MATCH_THREADWISE:
      Base::match_two_hop_nodes_threadwise();
      break;
    case TwoHopStrategy::CLUSTER:
      Base::cluster_two_hop_nodes();
      break;
    case TwoHopStrategy::CLUSTER_THREADWISE:
      Base::cluster_two_hop_nodes_threadwise();
      break;
    case TwoHopStrategy::DISABLE:
      break;
    }
  }

  void cluster_isolated_nodes() {
    SCOPED_HEAP_PROFILER("Handle isolated nodes");
    SCOPED_TIMER("Handle isolated nodes");

    switch (_ctx.coarsening.clustering.lp.isolated_nodes_strategy) {
    case IsolatedNodesClusteringStrategy::MATCH:
      Base::match_isolated_nodes();
      break;
    case IsolatedNodesClusteringStrategy::CLUSTER:
      Base::cluster_isolated_nodes();
      break;
    case IsolatedNodesClusteringStrategy::MATCH_DURING_TWO_HOP:
      if (should_handle_two_hop_nodes()) {
        Base::match_isolated_nodes();
      }
      break;
    case IsolatedNodesClusteringStrategy::CLUSTER_DURING_TWO_HOP:
      if (should_handle_two_hop_nodes()) {
        Base::cluster_isolated_nodes();
      }
      break;
    case IsolatedNodesClusteringStrategy::KEEP:
      break;
    }
  }

  [[nodiscard]] bool should_handle_two_hop_nodes() const {
    return (1.0 - 1.0 * _current_num_clusters / _graph->n()) <=
           _ctx.coarsening.clustering.lp.two_hop_threshold;
  }

public:
  [[nodiscard]] NodeID initial_cluster(const NodeID u) {
    return u;
  }

  [[nodiscard]] NodeWeight initial_cluster_weight(const NodeID cluster) {
    return _graph->node_weight(cluster);
  }

  [[nodiscard]] NodeWeight max_cluster_weight(const NodeID /* cluster */) {
    return _max_cluster_weight;
  }

  template <typename RatingMap>
  [[nodiscard]] ClusterID select_best_cluster(
      const bool store_favored_cluster,
      const EdgeWeight gain_delta,
      Base::ClusterSelectionState &state,
      RatingMap &map,
      ScalableVector<ClusterID> &tie_breaking_clusters,
      ScalableVector<ClusterID> &tie_breaking_favored_clusters
  ) {
    const bool use_uniform_tie_breaking = _tie_breaking_strategy == TieBreakingStrategy::UNIFORM;

    const auto accept_cluster_community = [&] {
      return _communities.empty() ||
             _communities[state.current_cluster] == _communities[state.initial_cluster];
    };

    ClusterID favored_cluster = state.initial_cluster;
    if (use_uniform_tie_breaking) {
      const auto accept_cluster = [&] {
        return (state.current_cluster_weight + state.u_weight <=
                    max_cluster_weight(state.current_cluster) ||
                state.current_cluster == state.initial_cluster) &&
               accept_cluster_community();
      };

      for (const auto [cluster, rating] : map.entries()) {
        state.current_cluster = cluster;
        state.current_gain = rating - gain_delta;
        state.current_cluster_weight = cluster_weight(cluster);

        if (store_favored_cluster) {
          if (state.current_gain > state.overall_best_gain) {
            state.overall_best_gain = state.current_gain;
            favored_cluster = state.current_cluster;

            tie_breaking_favored_clusters.clear();
            tie_breaking_favored_clusters.push_back(state.current_cluster);
          } else if (state.current_gain == state.overall_best_gain) {
            tie_breaking_favored_clusters.push_back(state.current_cluster);
          }
        }

        if (state.current_gain > state.best_gain) {
          if (accept_cluster()) {
            tie_breaking_clusters.clear();
            tie_breaking_clusters.push_back(state.current_cluster);

            state.best_cluster = state.current_cluster;
            state.best_gain = state.current_gain;
          }
        } else if (state.current_gain == state.best_gain) {
          if (accept_cluster()) {
            tie_breaking_clusters.push_back(state.current_cluster);
          }
        }
      }

      if (tie_breaking_clusters.size() > 1) {
        const ClusterID i = state.local_rand.random_index(0, tie_breaking_clusters.size());
        state.best_cluster = tie_breaking_clusters[i];
      }
      tie_breaking_clusters.clear();

      if (tie_breaking_favored_clusters.size() > 1) {
        const ClusterID i = state.local_rand.random_index(0, tie_breaking_favored_clusters.size());
        favored_cluster = tie_breaking_favored_clusters[i];
      }
      tie_breaking_favored_clusters.clear();

      return favored_cluster;
    } else {
      const auto accept_cluster = [&] {
        return (state.current_gain > state.best_gain ||
                (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
               (state.current_cluster_weight + state.u_weight <=
                    max_cluster_weight(state.current_cluster) ||
                state.current_cluster == state.initial_cluster) &&
               accept_cluster_community();
      };

      for (const auto [cluster, rating] : map.entries()) {
        state.current_cluster = cluster;
        state.current_gain = rating - gain_delta;
        state.current_cluster_weight = cluster_weight(cluster);

        if (store_favored_cluster && state.current_gain > state.overall_best_gain) {
          state.overall_best_gain = state.current_gain;
          favored_cluster = state.current_cluster;
        }

        if (accept_cluster()) {
          state.best_cluster = state.current_cluster;
          state.best_cluster_weight = state.current_cluster_weight;
          state.best_gain = state.current_gain;
        }
      }

      return favored_cluster;
    }
  }

  using Base::_current_num_clusters;
  using Base::_graph;
  using Base::_tie_breaking_strategy;

  const Context &_ctx;
  NodeWeight _max_cluster_weight = kInvalidBlockWeight;

  std::span<const NodeID> _communities;
};

class LPClusteringImplWrapper {
public:
  LPClusteringImplWrapper(const Context &ctx)
      : _ctx(ctx),
        _csr_impl(std::make_unique<LPClusteringImpl<CSRGraph>>(ctx, _permutations)),
        _csr_all_impl(
            std::make_unique<LPClusteringImpl<CSRGraph, AllNeighborsSampler>>(ctx, _permutations)
        ),
        _csr_avg_degree_impl(
            std::make_unique<LPClusteringImpl<CSRGraph, AvgDegreeSampler>>(ctx, _permutations)
        ) {}

  void set_max_cluster_weight(const NodeWeight max_cluster_weight) {
    _csr_impl->set_max_cluster_weight(max_cluster_weight);
    _csr_all_impl->set_max_cluster_weight(max_cluster_weight);
    _csr_avg_degree_impl->set_max_cluster_weight(max_cluster_weight);
  }

  void set_desired_cluster_count(const NodeID count) {
    _csr_impl->set_desired_num_clusters(count);
  }

  void set_communities(std::span<const NodeID> communities) {
    _csr_impl->set_communities(communities);
  }

  void compute_clustering(
      StaticArray<NodeID> &clustering, const Graph &graph, const bool free_memory_afterwards
  ) {
    // Compute a clustering and setup/release the data structures used by the core, so that they can
    // be shared by all implementations.
    const auto compute_clustering = [&](auto &core, auto &graph) {
      if (_freed) {
        _freed = false;
        core.allocate(graph.n());
      } else {
        core.setup(std::move(_structs));
        core.setup_cluster_weights(std::move(_cluster_weights));
      }

      core.compute_clustering(clustering, graph);

      if (free_memory_afterwards) {
        _freed = true;
        core.free();
      } else {
        _structs = core.release();
        _cluster_weights = core.take_cluster_weights();
      }
    };

    const NodeID num_nodes = graph.n();
    _csr_impl->preinitialize(num_nodes);
    _csr_all_impl->preinitialize(num_nodes);
    _csr_avg_degree_impl->preinitialize(num_nodes);
    //_compressed_impl->preinitialize(num_nodes);

    graph.reified(
        [&](const auto &csr_graph) {
          switch (_ctx.coarsening.clustering.lp.neighborhood_sampling_strategy) {
          case NeighborhoodSamplingStrategy::DISABLED:
            compute_clustering(*_csr_impl, csr_graph);
            break;

          case NeighborhoodSamplingStrategy::ALL:
            compute_clustering(*_csr_all_impl, csr_graph);
            break;

          case NeighborhoodSamplingStrategy::AVG_DEGREE:
            compute_clustering(*_csr_avg_degree_impl, csr_graph);
            break;
          }
        },
        [&](const auto &) {}
    );

    // Only relabel clusters for the first iteration
    _csr_impl->set_relabel_before_second_phase(false);
  }

  NodeID num_skipped() {
    return _csr_avg_degree_impl->num_skipped() + _csr_impl->num_skipped() +
           _csr_all_impl->num_skipped();
  }

  NodeID num_visited() {
    return _csr_avg_degree_impl->num_visited() + _csr_impl->num_visited() +
           _csr_all_impl->num_visited();
  }

private:
  const Context &_ctx;

  std::unique_ptr<LPClusteringImpl<CSRGraph>> _csr_impl;
  std::unique_ptr<LPClusteringImpl<CSRGraph, AllNeighborsSampler>> _csr_all_impl;
  std::unique_ptr<LPClusteringImpl<CSRGraph, AvgDegreeSampler>> _csr_avg_degree_impl;

  // The data structures that are used by the LP clusterer and are shared between the
  // different implementations.
  bool _freed = true;
  LPClusteringImpl<Graph>::Permutations _permutations;
  LPClusteringImpl<Graph>::DataStructures _structs;
  LPClusteringImpl<Graph>::ClusterWeights _cluster_weights;
};

//
// Exposed wrapper
//

LPClustering::LPClustering(const Context &ctx)
    : _impl_wrapper(std::make_unique<LPClusteringImplWrapper>(ctx)) {}

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

NodeID LPClustering::num_skipped() {
  return _impl_wrapper->num_skipped();
}

NodeID LPClustering::num_visited() {
  return _impl_wrapper->num_visited();
}

void LPClustering::compute_clustering(
    StaticArray<NodeID> &clustering, const Graph &graph, const bool free_memory_afterwards
) {
  return _impl_wrapper->compute_clustering(clustering, graph, free_memory_afterwards);
}

} // namespace kaminpar::shm
