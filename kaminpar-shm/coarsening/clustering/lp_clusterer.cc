/******************************************************************************
 * Label propagation for graph coarsening / clustering.
 *
 * @file:   lp_clusterer.cc
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#include "kaminpar-shm/coarsening/clustering/lp_clusterer.h"

#include <cstddef>
#include <span>

#include <tbb/concurrent_vector.h>
#include <tbb/task_arena.h>

#include "kaminpar-shm/algorithms/iteration_order.h"
#include "kaminpar-shm/algorithms/label_propagation/clustering_node_processor.h"
#include "kaminpar-shm/algorithms/label_propagation/clustering_postprocessor.h"
#include "kaminpar-shm/algorithms/label_propagation/clustering_selector.h"
#include "kaminpar-shm/algorithms/label_propagation/clustering_state.h"
#include "kaminpar-shm/algorithms/label_propagation/node_processing.h"
#include "kaminpar-shm/algorithms/label_propagation/parallel_rating_map.h"
#include "kaminpar-shm/algorithms/label_propagation/rating_map_pool.h"

#include "kaminpar-common/datastructures/cache_aligned_vector.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

using ClusteringRatingMaps = lp::RatingMapPool<NodeID, EdgeWeight>;

class LPClusteringWorkspace {
public:
  void ensure_capacity(const NodeID num_nodes) {
    rating_maps.ensure_capacity(num_nodes);
    const std::size_t concurrency = tbb::this_task_arena::max_concurrency();
    if (local_selections.size() < concurrency) {
      local_selections.resize(
          concurrency,
          lp::ClusteringSelector::RatedSelection{
              .cluster = 0,
              .rating = -1,
              .favored_cluster = 0,
              .favored_rating = -1,
          }
      );
    }
  }

  void begin_round() {
    statistics.clear();
  }

  void free() {
    state.free();
    rating_maps.free();
    parallel_ratings.free();
    deferred_nodes.clear();
    deferred_nodes.shrink_to_fit();
    local_selections.clear();
    local_selections.shrink_to_fit();
    statistics.clear();
  }

  lp::ClusteringState state;
  ClusteringRatingMaps rating_maps;
  lp::ParallelRatingMap<NodeID, EdgeWeight> parallel_ratings;
  tbb::concurrent_vector<NodeID> deferred_nodes;
  CacheAlignedVector<lp::ClusteringSelector::RatedSelection> local_selections;
  lp::RoundStatistics statistics;
};

namespace {

template <typename Graph> class LPClusteringRunner {
public:
  LPClusteringRunner(
      const Graph &graph,
      const LabelPropagationCoarseningContext &lp_ctx,
      LPClusteringWorkspace &workspace,
      InOrderIterationOrder &in_order,
      ChunkShuffledIterationOrder &chunk_shuffled
  )
      : _graph(graph),
        _lp_ctx(lp_ctx),
        _workspace(workspace),
        _in_order(in_order),
        _chunk_shuffled(chunk_shuffled) {}

  void run(StaticArray<NodeID> &clustering) {
    _workspace.ensure_capacity(_graph.n());
    _workspace.state.reset(clustering, _graph);
    initialize_iteration_order();

    lp::ClusteringNodeProcessor processor(
        _graph,
        _workspace.state,
        _workspace.rating_maps,
        _workspace.parallel_ratings,
        _workspace.deferred_nodes,
        _workspace.local_selections,
        _workspace.statistics
    );

    for (std::size_t iteration = 0; iteration < _lp_ctx.num_iterations; ++iteration) {
      SCOPED_TIMER(iteration == 0 ? "Initial iteration" : "Remaining iterations");
      _workspace.begin_round();

      if (_lp_ctx.rating_aggregation == LabelPropagationRatingAggregation::DEFERRED_PARALLEL) {
        {
          SCOPED_HEAP_PROFILER("First phase");
          SCOPED_TIMER("First phase");
          auto kernel = lp::make_first_phase_kernel(processor);
          for_each(kernel);
        }
        if (!_workspace.deferred_nodes.empty()) {
          SCOPED_HEAP_PROFILER("Second phase");
          SCOPED_TIMER("Second phase");
          processor.process_deferred_nodes();
        }
      } else {
        auto kernel = lp::make_single_phase_kernel(processor);
        for_each(kernel);
      }

      if (_workspace.statistics.totals().moved == 0) {
        break;
      }
    }

    SCOPED_HEAP_PROFILER("Handle two-hop nodes");
    SCOPED_TIMER("Handle two-hop nodes");
    lp::ClusteringPostprocessor(
        _graph, _workspace.state, _lp_ctx.two_hop_strategy, _lp_ctx.two_hop_threshold
    )
        .run();
  }

private:
  void initialize_iteration_order() {
    if (_lp_ctx.iteration_order == LabelPropagationIterationOrder::IN_ORDER) {
      _in_order.initialize(_graph.n());
    } else {
      _chunk_shuffled.initialize(_graph);
    }
  }

  template <typename Kernel> void for_each(Kernel &kernel) {
    if (_lp_ctx.iteration_order == LabelPropagationIterationOrder::IN_ORDER) {
      _in_order.for_each(kernel);
    } else {
      _chunk_shuffled.for_each(kernel);
    }
  }

  const Graph &_graph;
  const LabelPropagationCoarseningContext &_lp_ctx;
  LPClusteringWorkspace &_workspace;
  InOrderIterationOrder &_in_order;
  ChunkShuffledIterationOrder &_chunk_shuffled;
};

} // namespace

class LPClusteringImplWrapper {
public:
  explicit LPClusteringImplWrapper(const CoarseningContext &c_ctx)
      : _lp_ctx(c_ctx.clustering.lp),
        _chunk_shuffled(_permutations) {}

  void set_max_cluster_weight(const NodeWeight max_cluster_weight) {
    _max_cluster_weight = max_cluster_weight;
  }

  void set_desired_cluster_count(const NodeID count) {
    _desired_cluster_count = count;
  }

  void set_communities(const std::span<const NodeID> communities) {
    _communities = communities;
  }

  void compute_clustering(
      StaticArray<NodeID> &clustering, const Graph &graph, const bool free_memory_afterwards
  ) {
    _workspace.state.set_max_cluster_weight(_max_cluster_weight);
    _workspace.state.set_desired_num_clusters(_desired_cluster_count);
    _workspace.state.set_communities(_communities);

    reified(graph, [&]<typename ConcreteGraph>(const ConcreteGraph &concrete_graph) {
      LPClusteringRunner(concrete_graph, _lp_ctx, _workspace, _in_order, _chunk_shuffled)
          .run(clustering);
    });

    if (free_memory_afterwards) {
      SCOPED_HEAP_PROFILER("Deallocation");
      SCOPED_TIMER("Deallocation");
      _workspace.free();
      _chunk_shuffled.free();
    }
  }

private:
  const LabelPropagationCoarseningContext &_lp_ctx;
  LPClusteringWorkspace _workspace;

  ChunkShuffledIterationOrder::Permutations _permutations;
  InOrderIterationOrder _in_order;
  ChunkShuffledIterationOrder _chunk_shuffled;

  NodeWeight _max_cluster_weight = kInvalidBlockWeight;
  NodeID _desired_cluster_count = 0;
  std::span<const NodeID> _communities;
};

LPClustering::LPClustering(const CoarseningContext &c_ctx)
    : _impl_wrapper(std::make_unique<LPClusteringImplWrapper>(c_ctx)) {}

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
  _impl_wrapper->compute_clustering(clustering, graph, free_memory_afterwards);
}

} // namespace kaminpar::shm
