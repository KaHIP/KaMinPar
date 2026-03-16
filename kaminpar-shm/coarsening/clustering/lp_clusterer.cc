/******************************************************************************
 * Label propagation for graph coarsening / clustering.
 *
 * @file:   lp_clusterer.cc
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#include "kaminpar-shm/coarsening/clustering/lp_clusterer.h"

#include <span>

#include "kaminpar-shm/label_propagation/active_set.h"
#include "kaminpar-shm/label_propagation/chunk_random_iteration.h"
#include "kaminpar-shm/label_propagation/cluster_ops.h"
#include "kaminpar-shm/label_propagation/config.h"
#include "kaminpar-shm/label_propagation/node_processor.h"
#include "kaminpar-shm/label_propagation/simple_gain_selection.h"
#include "kaminpar-shm/label_propagation/two_hop_clustering.h"

#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

//
// Configuration
//

struct LPClusteringConfig : public LabelPropagationConfig {
  using ClusterID = NodeID;
  using ClusterWeight = BlockWeight;
  static constexpr bool kTrackClusterCount = true;
  static constexpr bool kUseTwoHopClustering = true;
};

//
// ClusterOps: the plain struct that satisfies the ClusterOps concept.
// Replaces all CRTP hooks from the old LPClusteringImpl.
//

template <typename Graph> struct LPClusteringOps {
  NonatomicClusterVectorRef<NodeID, NodeID> *clusters = nullptr;
  OwnedRelaxedClusterWeightVector<NodeID, NodeWeight> *weights = nullptr;
  const Graph *graph = nullptr;
  NodeWeight max_weight = kInvalidBlockWeight;
  std::span<const NodeID> communities;

  NodeID cluster(const NodeID u) {
    return clusters->cluster(u);
  }

  void move_node(const NodeID u, const NodeID c) {
    clusters->move_node(u, c);
  }

  NodeWeight cluster_weight(const NodeID c) {
    return weights->cluster_weight(c);
  }

  bool move_cluster_weight(
      const NodeID old_c, const NodeID new_c, const NodeWeight d, const NodeWeight max
  ) {
    return weights->move_cluster_weight(old_c, new_c, d, max);
  }

  NodeWeight max_cluster_weight(const NodeID /* c */) {
    return max_weight;
  }

  NodeID initial_cluster(const NodeID u) {
    return u;
  }

  NodeWeight initial_cluster_weight(const NodeID c) {
    return graph->node_weight(c);
  }

  bool accept_neighbor(const NodeID u, const NodeID v) {
    return communities.empty() || communities[u] == communities[v];
  }

  // Used by the selection strategy to check community constraint.
  bool accept_cluster(const NodeID current_cluster, const NodeID initial_cluster) {
    return communities.empty() || communities[current_cluster] == communities[initial_cluster];
  }

  void init_cluster(const NodeID u, const NodeID c) {
    clusters->move_node(u, c);
  }

  void init_cluster_weight(const NodeID c, const NodeWeight w) {
    weights->init_cluster_weight(c, w);
  }

  void reassign_cluster_weights(const StaticArray<NodeID> &mapping, const NodeID num_new_clusters) {
    weights->reassign_cluster_weights(mapping, num_new_clusters);
  }

  bool skip_node(const NodeID /* u */) {
    return false;
  }

  void reset_node_state(const NodeID /* u */) {}

  bool activate_neighbor(const NodeID /* v */) {
    return true;
  }

  NodeWeight min_cluster_weight(const NodeID /* c */) {
    return 0;
  }
};

//
// Actual implementation -- composition, no CRTP
//

template <typename Graph> class LPClusteringImpl {
  SET_DEBUG(false);

  using Config = LPClusteringConfig;
  using ClusterID = Config::ClusterID;
  using Ops = LPClusteringOps<Graph>;
  using Selection = lp::SimpleGainClusterSelection<Ops>;
  using Processor = lp::LPNodeProcessor<Graph, Ops, Selection, Config>;
  using Iterator = lp::ChunkRandomIterator<Config>;

  static constexpr bool kUseActiveSet =
      Config::kUseActiveSetStrategy || Config::kUseLocalActiveSetStrategy;

public:
  using Permutations = Iterator::Permutations;

  // Data structures for memory reuse between calls. Combines processor + iterator state.
  using DataStructures = std::tuple<typename Processor::DataStructures, typename Iterator::DataStructures>;

  LPClusteringImpl(const CoarseningContext &c_ctx, Permutations &permutations)
      : _lp_ctx(c_ctx.clustering.lp),
        _selection(_ops, _lp_ctx.tie_breaking_strategy),
        _processor(_ops, _selection, _active_set),
        _iterator(permutations),
        _max_degree(_lp_ctx.large_degree_threshold),
        _impl(_lp_ctx.impl),
        _relabel_before_second_phase(_lp_ctx.relabel_before_second_phase) {
    _processor.set_max_num_neighbors(_lp_ctx.max_num_neighbors);
  }

  void set_max_cluster_weight(const NodeWeight max_cluster_weight) {
    _ops.max_weight = max_cluster_weight;
  }

  void set_desired_num_clusters(const NodeID count) {
    _processor.set_desired_num_clusters(count);
  }

  void set_communities(const std::span<const NodeID> communities) {
    _ops.communities = communities;
  }

  void set_relabel_before_second_phase(const bool relabel) {
    _relabel_before_second_phase = relabel;
  }

  void preinitialize(const NodeID num_nodes) {
    _num_nodes = num_nodes;
  }

  void allocate(const NodeID num_clusters) {
    SCOPED_HEAP_PROFILER("Allocation");
    SCOPED_TIMER("Allocation");

    _processor.allocate(_num_nodes, _num_nodes, num_clusters);
    _cluster_weights.allocate_cluster_weights(num_clusters);
  }

  void free() {
    SCOPED_HEAP_PROFILER("Free");
    SCOPED_TIMER("Free");

    _processor.free();
    _iterator.free();
    _cluster_weights.free();
  }

  void setup(DataStructures structs) {
    auto [proc_structs, iter_structs] = std::move(structs);
    _processor.setup(std::move(proc_structs));
    _iterator.setup(std::move(iter_structs));
  }

  DataStructures release() {
    return std::make_tuple(_processor.release(), _iterator.release());
  }

  void setup_cluster_weights(OwnedRelaxedClusterWeightVector<NodeID, NodeWeight>::ClusterWeights cw) {
    _cluster_weights.setup_cluster_weights(std::move(cw));
  }

  OwnedRelaxedClusterWeightVector<NodeID, NodeWeight>::ClusterWeights take_cluster_weights() {
    return _cluster_weights.take_cluster_weights();
  }

  void compute_clustering(StaticArray<NodeID> &clustering, const Graph &graph) {
    _ops.graph = &graph;
    _ops.clusters->init_clusters_ref(clustering);
    _cluster_weights.reset_cluster_weights();

    _processor.initialize(&graph, graph.n());
    _iterator.clear();

    for (std::size_t iteration = 0; iteration < _lp_ctx.num_iterations; ++iteration) {
      SCOPED_TIMER("Iteration", std::to_string(iteration));
      if (perform_iteration(graph) == 0) {
        break;
      }

      // Only relabel during the first iteration because afterwards the memory for the second
      // phase is already allocated.
      if (iteration == 0) {
        _relabel_before_second_phase = false;
      }
    }

    handle_isolated_nodes(graph);
    handle_two_hop_nodes(graph);
  }

private:
  NodeID perform_iteration(const Graph &graph) {
    if (_iterator.empty()) {
      _iterator.init_chunks(graph, 0, graph.n(), _max_degree);
    }
    _iterator.shuffle_chunks();

    auto &current_num_clusters = _processor.current_num_clusters();

    auto handler = [&](const NodeID u) {
      auto &rand = Random::instance();
      auto &rating_map = _processor.rating_map_ets().local();
      auto &tb = _processor.tie_breaking_clusters_ets().local();
      auto &tbf = _processor.tie_breaking_favored_clusters_ets().local();
      return _processor.handle_node(u, rand, rating_map, tb, tbf);
    };

    auto first_phase_handler = [&](const NodeID u) {
      auto &rand = Random::instance();
      auto &rating_map = _processor.rating_map_ets().local();
      auto &tb = _processor.tie_breaking_clusters_ets().local();
      auto &tbf = _processor.tie_breaking_favored_clusters_ets().local();
      auto result = _processor.handle_first_phase_node(u, rand, rating_map, tb, tbf);
      if (result.first && _processor.relabeled()) {
        _processor.moved()[u] = 1;
      }
      return result;
    };

    auto should_stop = [&] { return _processor.should_stop(); };
    auto is_active = [&](const NodeID u) { return _processor.is_active(u); };

    switch (_impl) {
    case LabelPropagationImplementation::GROWING_HASH_TABLES: {
      // Use growing rating maps
      auto growing_handler = [&](const NodeID u) {
        auto &rand = Random::instance();
        auto &rating_map = _processor.rating_map_ets().local(); // @todo use growing map
        auto &tb = _processor.tie_breaking_clusters_ets().local();
        auto &tbf = _processor.tie_breaking_favored_clusters_ets().local();
        return _processor.handle_node(u, rand, rating_map, tb, tbf);
      };
      return _iterator.iterate(graph, _max_degree, growing_handler, should_stop, is_active, current_num_clusters);
    }
    case LabelPropagationImplementation::SINGLE_PHASE:
      return _iterator.iterate(graph, _max_degree, handler, should_stop, is_active, current_num_clusters);
    case LabelPropagationImplementation::TWO_PHASE: {
      const NodeID initial_num_clusters = _processor.initial_num_clusters();
      const auto [num_processed, num_moved_first] = _iterator.iterate_first_phase(
          graph, _max_degree, first_phase_handler, should_stop, is_active, current_num_clusters
      );

      auto &second_phase_nodes = _processor.second_phase_nodes();
      const NodeID num_second_phase_nodes = second_phase_nodes.size();
      NodeID total_moved = num_moved_first;

      if (num_second_phase_nodes > 0) {
        if (_relabel_before_second_phase) {
          _processor.relabel_clusters();
        }

        // Second phase: process deferred high-degree nodes sequentially
        const std::size_t num_clusters = _processor.initial_num_clusters();
        auto &concurrent_map = _processor.concurrent_rating_map();
        if (concurrent_map.capacity() < num_clusters) {
          concurrent_map.resize(num_clusters);
        }

        auto &rand = Random::instance();
        for (const NodeID u : second_phase_nodes) {
          const auto [moved_node, emptied_cluster] =
              _processor.handle_second_phase_node(u, rand, concurrent_map);

          if (moved_node) {
            ++total_moved;
            if (_processor.relabeled()) {
              _processor.moved()[u] = 1;
            }
          }
          if (emptied_cluster) {
            --current_num_clusters;
          }
        }

        second_phase_nodes.clear();
      }

      return total_moved;
    }
    }

    __builtin_unreachable();
  }

  void handle_isolated_nodes(const Graph &graph) {
    SCOPED_HEAP_PROFILER("Handle isolated nodes");
    SCOPED_TIMER("Handle isolated nodes");

    switch (_lp_ctx.isolated_nodes_strategy) {
    case IsolatedNodesClusteringStrategy::MATCH:
      lp::two_hop::match_isolated_nodes(graph, _ops);
      break;
    case IsolatedNodesClusteringStrategy::CLUSTER:
      lp::two_hop::cluster_isolated_nodes(graph, _ops);
      break;
    case IsolatedNodesClusteringStrategy::MATCH_DURING_TWO_HOP:
      if (should_handle_two_hop_nodes(graph)) {
        lp::two_hop::match_isolated_nodes(graph, _ops);
      }
      break;
    case IsolatedNodesClusteringStrategy::CLUSTER_DURING_TWO_HOP:
      if (should_handle_two_hop_nodes(graph)) {
        lp::two_hop::cluster_isolated_nodes(graph, _ops);
      }
      break;
    case IsolatedNodesClusteringStrategy::KEEP:
      break;
    }
  }

  void handle_two_hop_nodes(const Graph &graph) {
    SCOPED_HEAP_PROFILER("Handle two-hop nodes");
    SCOPED_TIMER("Handle two-hop nodes");

    if (!should_handle_two_hop_nodes(graph)) {
      return;
    }

    auto &favored_clusters = _processor.favored_clusters();
    auto &current_num_clusters = _processor.current_num_clusters();
    const NodeID desired_num_clusters = 0; // clusterer doesn't use early stopping for 2-hop

    switch (_lp_ctx.two_hop_strategy) {
    case TwoHopStrategy::MATCH:
      lp::two_hop::match_two_hop_nodes(
          graph, _ops, favored_clusters, current_num_clusters, desired_num_clusters
      );
      break;
    case TwoHopStrategy::MATCH_THREADWISE:
      lp::two_hop::match_two_hop_nodes_threadwise(
          graph, _ops, favored_clusters, &_processor.moved(), _processor.relabeled()
      );
      break;
    case TwoHopStrategy::CLUSTER:
      lp::two_hop::cluster_two_hop_nodes(
          graph, _ops, favored_clusters, current_num_clusters, desired_num_clusters
      );
      break;
    case TwoHopStrategy::CLUSTER_THREADWISE:
      lp::two_hop::cluster_two_hop_nodes_threadwise(
          graph, _ops, favored_clusters, &_processor.moved(), _processor.relabeled()
      );
      break;
    case TwoHopStrategy::DISABLE:
      break;
    }
  }

  [[nodiscard]] bool should_handle_two_hop_nodes(const Graph &graph) const {
    return (1.0 - 1.0 * _processor.current_num_clusters() / graph.n()) <=
           _lp_ctx.two_hop_threshold;
  }

  // --- Members ---

  const LabelPropagationCoarseningContext &_lp_ctx;

  // Building blocks (composition)
  NonatomicClusterVectorRef<NodeID, NodeID> _cluster_storage;
  OwnedRelaxedClusterWeightVector<NodeID, NodeWeight> _cluster_weights;
  ActiveSet<kUseActiveSet> _active_set;
  Ops _ops = {&_cluster_storage, &_cluster_weights, nullptr, kInvalidBlockWeight, {}};
  Selection _selection;
  Processor _processor;
  Iterator _iterator;

  NodeID _num_nodes = 0;
  NodeID _max_degree;
  LabelPropagationImplementation _impl;
  bool _relabel_before_second_phase;
};

//
// Wrapper that handles CSR vs Compressed graph dispatch + memory reuse
//

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

  // The data structures that are used by the LP clusterer and are shared between the
  // different implementations.
  bool _freed = true;
  LPClusteringImpl<Graph>::Permutations _permutations;
  LPClusteringImpl<Graph>::DataStructures _structs;
  OwnedRelaxedClusterWeightVector<NodeID, NodeWeight>::ClusterWeights _cluster_weights;
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
