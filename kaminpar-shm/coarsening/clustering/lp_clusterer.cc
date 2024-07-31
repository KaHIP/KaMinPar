/******************************************************************************
 * Label propagation for graph coarsening / clustering.
 *
 * @file:   lp_clusterer.cc
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#include "kaminpar-shm/coarsening/clustering/lp_clusterer.h"

#include "kaminpar-shm/label_propagation.h"

#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

//
// Actual implementation -- not exposed in header
//

struct LPClusteringConfig : public LabelPropagationConfig {
  using ClusterID = NodeID;
  using ClusterWeight = BlockWeight;
  static constexpr bool kTrackClusterCount = true;
  static constexpr bool kUseTwoHopClustering = true;
};

template <typename Graph>
class LPClusteringImpl final
    : public ChunkRandomLabelPropagation<LPClusteringImpl<Graph>, LPClusteringConfig, Graph>,
      public OwnedRelaxedClusterWeightVector<NodeID, NodeWeight>,
      public NonatomicClusterVectorRef<NodeID, NodeID> {
  SET_DEBUG(false);

  using Base = ChunkRandomLabelPropagation<LPClusteringImpl, LPClusteringConfig, Graph>;
  using ClusterWeightBase = OwnedRelaxedClusterWeightVector<NodeID, NodeWeight>;
  using ClusterBase = NonatomicClusterVectorRef<NodeID, NodeID>;

  using Config = LPClusteringConfig;
  using ClusterID = Config::ClusterID;

public:
  using Permutations = Base::Permutations;

  LPClusteringImpl(const CoarseningContext &c_ctx, Permutations &permutations)
      : Base(permutations),
        ClusterWeightBase(c_ctx.clustering.lp.cluster_weights_structure),
        _lp_ctx(c_ctx.clustering.lp) {
    Base::set_max_degree(_lp_ctx.large_degree_threshold);
    Base::set_max_num_neighbors(_lp_ctx.max_num_neighbors);
    Base::set_implementation(_lp_ctx.impl);
    Base::set_tie_breaking_strategy(_lp_ctx.tie_breaking_strategy);
    Base::set_second_phase_selection_strategy(_lp_ctx.second_phase_selection_strategy);
    Base::set_second_phase_aggregation_strategy(_lp_ctx.second_phase_aggregation_strategy);
    Base::set_relabel_before_second_phase(_lp_ctx.relabel_before_second_phase);
  }

  void set_max_cluster_weight(const NodeWeight max_cluster_weight) {
    _max_cluster_weight = max_cluster_weight;
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
    Base::initialize(&graph, graph.n());

    for (std::size_t iteration = 0; iteration < _lp_ctx.num_iterations; ++iteration) {
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
  }

private:
  void cluster_two_hop_nodes() {
    SCOPED_HEAP_PROFILER("Handle two-hop nodes");
    SCOPED_TIMER("Handle two-hop nodes");

    if (!should_handle_two_hop_nodes()) {
      return;
    }

    switch (_lp_ctx.two_hop_strategy) {
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
    case TwoHopStrategy::LEGACY:
      handle_two_hop_clustering_legacy();
      break;
    case TwoHopStrategy::DISABLE:
      break;
    }
  }

  void cluster_isolated_nodes() {
    SCOPED_HEAP_PROFILER("Handle isolated nodes");
    SCOPED_TIMER("Handle isolated nodes");

    switch (_lp_ctx.isolated_nodes_strategy) {
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
    return (1.0 - 1.0 * _current_num_clusters / _graph->n()) <= _lp_ctx.two_hop_threshold;
  }

  // @todo: old implementation that should no longer be used
  void handle_two_hop_clustering_legacy() {
    // Reset _favored_clusters entries for nodes that are not considered for
    // 2-hop clustering, i.e., nodes that are already clustered with at least one other node or
    // nodes that have more weight than max_weight/2.
    // Set _favored_clusters to dummy entry _graph->n() for isolated nodes
    tbb::parallel_for<NodeID>(0, _graph->n(), [&](const NodeID u) {
      if (u != cluster(u)) {
        Base::_favored_clusters[u] = u;
      } else {
        const auto initial_weight = initial_cluster_weight(u);
        const auto current_weight = ClusterWeightBase::cluster_weight(u);
        const auto max_weight = max_cluster_weight(u);
        if (current_weight != initial_weight || current_weight > max_weight / 2) {
          Base::_favored_clusters[u] = u;
        }
      }
    });

    tbb::parallel_for<NodeID>(0, _graph->n(), [&](const NodeID u) {
      // Abort once we have merged enough clusters to achieve the configured minimum shrink factor
      if (Base::should_stop()) {
        return;
      }

      // Skip nodes that should not be considered during 2-hop clustering
      const NodeID favored_leader = Base::_favored_clusters[u];
      if (favored_leader == u) {
        return;
      }

      do {
        // If this works, we set ourself as clustering partners for nodes that have the same favored
        // cluster we have
        NodeID expected_value = favored_leader;
        if (__atomic_compare_exchange_n(
                &Base::_favored_clusters[favored_leader],
                &expected_value,
                u,
                false,
                __ATOMIC_SEQ_CST,
                __ATOMIC_SEQ_CST
            )) {
          break;
        }

        // If this did not work, there is another node that has the same favored cluster
        // Try to join the cluster of that node
        const NodeID partner = expected_value;
        if (__atomic_compare_exchange_n(
                &Base::_favored_clusters[favored_leader],
                &expected_value,
                favored_leader,
                false,
                __ATOMIC_SEQ_CST,
                __ATOMIC_SEQ_CST
            )) {
          if (ClusterWeightBase::move_cluster_weight(
                  u, partner, ClusterWeightBase::cluster_weight(u), max_cluster_weight(partner)
              )) {
            move_node(u, partner);
            --_current_num_clusters;
          }

          break;
        }
      } while (true);
    });
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

    ClusterID favored_cluster = state.initial_cluster;
    if (use_uniform_tie_breaking) {
      const auto accept_cluster = [&] {
        return state.current_cluster_weight + state.u_weight <=
                   max_cluster_weight(state.current_cluster) ||
               state.current_cluster == state.initial_cluster;
      };

      for (const auto [cluster, rating] : map.entries()) {
        state.current_cluster = cluster;
        state.current_gain = rating - gain_delta;
        state.current_cluster_weight = cluster_weight(cluster);

        if (state.current_gain > state.best_gain) {
          if (store_favored_cluster) {
            tie_breaking_favored_clusters.clear();
            tie_breaking_favored_clusters.push_back(state.current_cluster);

            favored_cluster = state.current_cluster;
          }

          if (accept_cluster()) {
            tie_breaking_clusters.clear();
            tie_breaking_clusters.push_back(state.current_cluster);

            state.best_cluster = state.current_cluster;
            state.best_gain = state.current_gain;
          }
        } else if (state.current_gain == state.best_gain) {
          if (store_favored_cluster) {
            tie_breaking_favored_clusters.push_back(state.current_cluster);
          }

          if (accept_cluster()) {
            tie_breaking_clusters.push_back(state.current_cluster);
          }
        }
      }

      if (tie_breaking_clusters.size() > 1) {
        const ClusterID i = state.local_rand.random_index(0, tie_breaking_clusters.size());
        const ClusterID best_cluster = tie_breaking_clusters[i];
        state.best_cluster = best_cluster;
      }
      tie_breaking_clusters.clear();

      if (tie_breaking_favored_clusters.size() > 1) {
        const ClusterID i = state.local_rand.random_index(0, tie_breaking_favored_clusters.size());
        const ClusterID best_favored_cluster = tie_breaking_favored_clusters[i];
        favored_cluster = best_favored_cluster;
      }
      tie_breaking_favored_clusters.clear();

      return favored_cluster;
    } else {
      const auto accept_cluster = [&] {
        return (state.current_gain > state.best_gain ||
                (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
               (state.current_cluster_weight + state.u_weight <=
                    max_cluster_weight(state.current_cluster) ||
                state.current_cluster == state.initial_cluster);
      };

      for (const auto [cluster, rating] : map.entries()) {
        state.current_cluster = cluster;
        state.current_gain = rating - gain_delta;
        state.current_cluster_weight = cluster_weight(cluster);

        if (store_favored_cluster && state.current_gain > state.best_gain) {
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

  const LabelPropagationCoarseningContext &_lp_ctx;
  NodeWeight _max_cluster_weight = kInvalidBlockWeight;
};

class LPClusteringImplWrapper {
public:
  LPClusteringImplWrapper(const CoarseningContext &c_ctx)
      : _csr_impl(std::make_unique<LPClusteringImpl<CSRGraph>>(c_ctx, _permutations)),
        _compressed_impl(std::make_unique<LPClusteringImpl<CompressedGraph>>(c_ctx, _permutations)
        ) {}

  void set_max_cluster_weight(const NodeWeight max_cluster_weight) {
    _csr_impl->set_max_cluster_weight(max_cluster_weight);
    _compressed_impl->set_max_cluster_weight(max_cluster_weight);
  }

  void set_desired_cluster_count(const NodeID count) {
    _csr_impl->set_desired_num_clusters(count);
    _compressed_impl->set_desired_num_clusters(count);
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
    _compressed_impl->preinitialize(num_nodes);

    graph.reified(
        [&](const auto &csr_graph) {
          LPClusteringImpl<CSRGraph> &impl = *_csr_impl;
          compute_clustering(impl, csr_graph);
        },
        [&](const auto &compressed_graph) {
          LPClusteringImpl<CompressedGraph> &impl = *_compressed_impl;
          compute_clustering(impl, compressed_graph);
        }
    );

    // Only relabel clusters during the first iteration
    _csr_impl->set_relabel_before_second_phase(false);
    _compressed_impl->set_relabel_before_second_phase(false);

    // Only use the initially small cluster weight vector for the first lp implementation
    _csr_impl->set_use_small_vector_initially(false);
    _compressed_impl->set_use_small_vector_initially(false);
  }

private:
  std::unique_ptr<LPClusteringImpl<CSRGraph>> _csr_impl;
  std::unique_ptr<LPClusteringImpl<CompressedGraph>> _compressed_impl;

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

void LPClustering::compute_clustering(
    StaticArray<NodeID> &clustering, const Graph &graph, const bool free_memory_afterwards
) {
  return _impl_wrapper->compute_clustering(clustering, graph, free_memory_afterwards);
}

} // namespace kaminpar::shm
