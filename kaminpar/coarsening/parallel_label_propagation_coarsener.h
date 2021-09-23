/*******************************************************************************
 * @file:   parallel_label_propagation_coarsener.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Coarsening algorithm that uses parallel label propagation.
 ******************************************************************************/
#pragma once

#include "algorithm/graph_contraction.h"
#include "algorithm/parallel_label_propagation.h"
#include "coarsening/i_coarsener.h"
#include "context.h"
#include "datastructure/graph.h"
#include "parallel.h"
#include "utility/timer.h"

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

namespace kaminpar {
struct LabelPropagationClusteringConfig : public LabelPropagationConfig {
  using ClusterID = BlockID;
  using ClusterWeight = BlockWeight;
  static constexpr bool kTrackClusterCount = true;
  static constexpr bool kUseTwoHopClustering = true;
};

class LabelPropagationClustering final
    : public ChunkRandomizedLabelPropagation<LabelPropagationClustering, LabelPropagationClusteringConfig>,
      public OwnedRelaxedClusterWeightVector<NodeID, NodeWeight>,
      public OwnedClusterVector<NodeID, NodeID> {
  SET_DEBUG(false);

  using Base = ChunkRandomizedLabelPropagation<LabelPropagationClustering, LabelPropagationClusteringConfig>;
  using ClusterWeightBase = OwnedRelaxedClusterWeightVector<NodeID, NodeWeight>;
  using ClusterBase = OwnedClusterVector<NodeID, NodeID>;

  static constexpr std::size_t kInfiniteIterations{std::numeric_limits<std::size_t>::max()};

public:
  using Base::set_desired_num_clusters;

  using ClusterBase::cluster;
  using ClusterBase::init_cluster;
  using ClusterBase::move_node;
  using ClusterWeightBase::cluster_weight;
  using ClusterWeightBase::init_cluster_weight;
  using ClusterWeightBase::move_cluster_weight;

  LabelPropagationClustering(const NodeID max_n, const LabelPropagationCoarseningContext &lp_ctx)
      : Base{max_n},
        ClusterWeightBase{max_n},
        ClusterBase{max_n},
        _lp_ctx{lp_ctx} {
    set_max_degree(lp_ctx.large_degree_threshold);
    set_max_num_neighbors(lp_ctx.max_num_neighbors);
  }

  const auto &cluster(const Graph &graph, const NodeWeight max_cluster_weight,
                      const std::size_t max_iterations = kInfiniteIterations) {
    initialize(&graph, graph.n());
    _max_cluster_weight = max_cluster_weight;

    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
      SCOPED_TIMER("Iteration", std::to_string(iteration), TIMER_BENCHMARK);
      if (perform_iteration() == 0) { break; }
    }

    if (_lp_ctx.should_merge_nonadjacent_clusters(_graph->n(), _current_num_clusters)) {
      TIMED_SCOPE("2-hop Clustering") { perform_two_hop_clustering(); };
    }

    return clusters();
  }

public:
  [[nodiscard]] NodeID initial_cluster(const NodeID u) const { return u; }

  [[nodiscard]] NodeWeight initial_cluster_weight(const NodeID cluster) const { return _graph->node_weight(cluster); }

  [[nodiscard]] NodeWeight max_cluster_weight(const NodeID /* cluster */) const { return _max_cluster_weight; }

  [[nodiscard]] bool accept_cluster(const Base::ClusterSelectionState &state) const {
    return (state.current_gain > state.best_gain ||
            (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
           (state.current_cluster_weight + state.u_weight < max_cluster_weight(state.current_cluster) ||
            state.current_cluster == state.initial_cluster);
  }

  using Base::_current_num_clusters;
  using Base::_graph;

  const LabelPropagationCoarseningContext &_lp_ctx;
  NodeWeight _max_cluster_weight{kInvalidBlockWeight};
};

class ParallelLabelPropagationCoarsener : public Coarsener {
public:
  ParallelLabelPropagationCoarsener(const Graph &input_graph, const CoarseningContext &c_ctx)
      : _input_graph{input_graph},
        _current_graph{&input_graph},
        _label_propagation_core{input_graph.n(), c_ctx.lp},
        _c_ctx{c_ctx} {}

  ParallelLabelPropagationCoarsener(const ParallelLabelPropagationCoarsener &) = delete;
  ParallelLabelPropagationCoarsener &operator=(const ParallelLabelPropagationCoarsener) = delete;
  ParallelLabelPropagationCoarsener(ParallelLabelPropagationCoarsener &&) = delete;
  ParallelLabelPropagationCoarsener &operator=(ParallelLabelPropagationCoarsener &&) = delete;

  using Coarsener::coarsen;

  std::pair<const Graph *, bool> coarsen(const std::function<NodeWeight(NodeID)> &cb_max_cluster_weight,
                                         const NodeID to_size) final {
    SCOPED_TIMER("Level", std::to_string(_hierarchy.size()), TIMER_BENCHMARK);

    _label_propagation_core.set_desired_num_clusters(to_size);

    const NodeWeight max_cluster_weight = cb_max_cluster_weight(_current_graph->n());

    const auto &clustering = TIMED_SCOPE("Label Propagation") {
      return _label_propagation_core.cluster(*_current_graph, max_cluster_weight, _c_ctx.lp.num_iterations);
    };

    auto [c_graph, c_mapping, m_ctx] = TIMED_SCOPE("Contract graph") {
      return graph::contract(*_current_graph, clustering, std::move(_contraction_m_ctx));
    };
    _contraction_m_ctx = std::move(m_ctx);

    const bool converged = _c_ctx.should_converge(_current_graph->n(), c_graph.n());

    _hierarchy.push_back(std::move(c_graph));
    _mapping.push_back(std::move(c_mapping));
    _current_graph = &_hierarchy.back();

    return {_current_graph, !converged};
  };

  PartitionedGraph uncoarsen(PartitionedGraph &&p_graph) final {
    ASSERT(&p_graph.graph() == _current_graph);
    ASSERT(!empty()) << size();
    SCOPED_TIMER("Level", std::to_string(_hierarchy.size()), TIMER_BENCHMARK);

    START_TIMER("Allocation");
    auto mapping{std::move(_mapping.back())};
    _mapping.pop_back();
    _hierarchy.pop_back(); // destroys the graph wrapped in p_graph, but partition access is still ok
    _current_graph = empty() ? &_input_graph : &_hierarchy.back();
    ASSERT(mapping.size() == _current_graph->n()) << V(mapping.size()) << V(_current_graph->n());

    StaticArray<parallel::IntegralAtomicWrapper<BlockID>> partition(_current_graph->n());
    STOP_TIMER();

    START_TIMER("Copy partition");
    tbb::parallel_for(static_cast<NodeID>(0), _current_graph->n(),
                      [&](const NodeID u) { partition[u] = p_graph.block(mapping[u]); });
    STOP_TIMER();

    SCOPED_TIMER("Create graph");
    PartitionedGraph new_p_graph(*_current_graph, p_graph.k(), std::move(partition), std::move(p_graph.take_final_k()));
#ifdef KAMINPAR_ENABLE_DEBUG_FEATURES
    new_p_graph.set_block_names(p_graph.block_names());
#endif // KAMINPAR_ENABLE_DEBUG_FEATURES

    return new_p_graph;
  }

  [[nodiscard]] const Graph *coarsest_graph() const final { return _current_graph; }
  [[nodiscard]] std::size_t size() const final { return _hierarchy.size(); }
  void set_community_structure(std::vector<BlockID>) final {}
  void initialize(const Graph *) final {}
  [[nodiscard]] const CoarseningContext &context() const { return _c_ctx; }

private:
  const Graph &_input_graph;
  const Graph *_current_graph;
  std::vector<Graph> _hierarchy;
  std::vector<scalable_vector<NodeID>> _mapping;
  LabelPropagationClustering _label_propagation_core;
  const CoarseningContext &_c_ctx;
  graph::contraction::MemoryContext _contraction_m_ctx{};
};
} // namespace kaminpar
