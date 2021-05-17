/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2020 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/

#pragma once

#include "algorithm/graph_utils.h"
#include "algorithm/parallel_label_propagation.h"
#include "coarsening/i_coarsener.h"
#include "context.h"
#include "datastructure/graph.h"
#include "parallel.h"
#include "utility/timer.h"

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

namespace kaminpar {
class LabelPropagationClustering final : public LabelPropagation<LabelPropagationClustering, NodeID, NodeWeight> {
  using Base = LabelPropagation<LabelPropagationClustering, NodeID, NodeWeight>;
  friend Base;

  SET_DEBUG(false);
  SET_OUTPUT(false);

  static constexpr std::size_t kInfiniteIterations{std::numeric_limits<std::size_t>::max()};

public:
  LabelPropagationClustering(const NodeID max_n, const double nonadjacent_clustering_fraction_threshold,
                             const bool randomize_chunk_order, const bool merge_singleton_clusters)
      : Base{max_n, max_n}, _nonadjacent_clustering_fraction_threshold{nonadjacent_clustering_fraction_threshold},
        _randomize_chunk_order{randomize_chunk_order}, _merge_singleton_clusters{merge_singleton_clusters} {
    _clustering.resize(max_n);
    _favored_clustering.resize(max_n);
  }

  const scalable_vector<NodeID> &cluster(const Graph &graph, const NodeWeight max_cluster_weight,
                                         const std::size_t max_iterations = kInfiniteIterations) {
    ASSERT(_clustering.size() >= graph.n());

    initialize(&graph);
    _max_cluster_weight = max_cluster_weight;
    NodeID total_num_emptied_clusters = 0;

    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
      const auto [num_moved_nodes, num_emptied_clusters] = label_propagation_iteration();
      total_num_emptied_clusters += num_emptied_clusters;
      if (num_moved_nodes == 0) { break; }
    }

    if (total_num_emptied_clusters < _graph->n() / _nonadjacent_clustering_fraction_threshold) {
      SCOPED_TIMER("Cluster merging");
      CLOG << "Label propagation emptied roughly " << total_num_emptied_clusters << " clusters"
           << " (of " << _graph->n()
           << ") -> join clusters by favored cluster ID ... max_cluster_weight=" << max_cluster_weight
           << " total_node_weight=" << graph.total_node_weight() << " max_node_weight=" << graph.max_node_weight()
           << " n=" << graph.n() << " m=" << graph.m();
      join_singleton_clusters_by_favored_cluster(total_num_emptied_clusters);
    }

    return _clustering;
  }

private:
  void join_singleton_clusters_by_favored_cluster(const NodeID emptied_clusters) {
    const NodeID desired_emptied_clusters = _graph->n() / _nonadjacent_clustering_fraction_threshold;
    parallel::IntegralAtomicWrapper<NodeID> current_emptied_clusters = emptied_clusters;

    tbb::parallel_for(static_cast<NodeID>(0), _graph->n(), [&](const NodeID u) {
      if (current_emptied_clusters >= desired_emptied_clusters) { return; }

      const NodeID leader = _clustering[u]; // first && part avoids _cluster_weights cache miss
      const bool singleton = leader == u && _cluster_weights[u] == _graph->node_weight(u);

      if (singleton) {
        NodeID favored_leader = _favored_clustering[u];
        if (_merge_singleton_clusters && u == favored_leader) { favored_leader = 0; }
        do {
          NodeID expected_value = favored_leader;
          if (_favored_clustering[favored_leader].compare_exchange_strong(expected_value, u)) {
            break; // if this worked, we replaced favored_leader with u
          }

          // if this didn't work, there is another node that has the same favored leader -> try to join that nodes
          // cluster
          const NodeID partner = expected_value;
          if (_favored_clustering[favored_leader].compare_exchange_strong(expected_value, favored_leader)) {
            _clustering[u] = partner;
            ++current_emptied_clusters;
            break;
          }
        } while (true);
      }
    });
  }

  // used in Base class
  static constexpr bool kUseHardWeightConstraint = false;
  static constexpr bool kUseFavoredCluster = true;
  static constexpr bool kReportEmptyClusters = true;

  void reset_node_state(const NodeID u) {
    _clustering[u] = u;
    _favored_clustering[u] = u;
  }

  NodeID cluster(const NodeID u) { return _clustering[u]; }
  void set_cluster(const NodeID u, const NodeID cluster) { _clustering[u] = cluster; }
  void set_favored_cluster(const NodeID u, const NodeID cluster) { _favored_clustering[u] = cluster; }
  NodeID num_clusters() { return _graph->n(); }
  NodeWeight initial_cluster_weight(const NodeID cluster) { return _graph->node_weight(cluster); }
  NodeWeight max_cluster_weight(const NodeID) { return _max_cluster_weight; }

  bool accept_cluster(const Base::ClusterSelectionState &state) {
    return (state.current_gain > state.best_gain ||
            (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
           (state.current_cluster_weight + state.u_weight < max_cluster_weight(state.current_cluster) ||
            state.current_cluster == state.initial_cluster);
  }

  scalable_vector<NodeID> _clustering;
  scalable_vector<parallel::IntegralAtomicWrapper<NodeID>> _favored_clustering;
  NodeWeight _max_cluster_weight;
  double _nonadjacent_clustering_fraction_threshold;
  bool _randomize_chunk_order;
  bool _merge_singleton_clusters;
};

class ParallelLabelPropagationCoarsener : public Coarsener {
public:
  ParallelLabelPropagationCoarsener(const Graph &input_graph, const CoarseningContext &c_ctx)
      : _input_graph{input_graph}, _current_graph{&input_graph},
        _label_propagation_core{input_graph.n(), c_ctx.nonadjacent_clustering_fraction_threshold,
                                c_ctx.randomize_chunk_order, c_ctx.merge_singleton_clusters},
        _c_ctx{c_ctx} {
    if (c_ctx.large_degree_threshold > 0) {
      _label_propagation_core.set_large_degree_threshold(c_ctx.large_degree_threshold);
    }
  }

  ParallelLabelPropagationCoarsener(const ParallelLabelPropagationCoarsener &) = delete;
  ParallelLabelPropagationCoarsener &operator=(const ParallelLabelPropagationCoarsener) = delete;
  ParallelLabelPropagationCoarsener(ParallelLabelPropagationCoarsener &&) = delete;
  ParallelLabelPropagationCoarsener &operator=(ParallelLabelPropagationCoarsener &&) = delete;

  using Coarsener::coarsen;

  std::pair<const Graph *, bool> coarsen(const std::function<NodeWeight(NodeID)> &cb_max_cluster_weight) final {
    const NodeWeight max_cluster_weight{cb_max_cluster_weight(_current_graph->n())};

    START_TIMER(TIMER_LABEL_PROPAGATION);
    const auto &clustering = _label_propagation_core.cluster(*_current_graph, max_cluster_weight,
                                                             _c_ctx.num_iterations);
    STOP_TIMER();
    START_TIMER(TIMER_CONTRACT_GRAPH);
    auto [c_graph, c_mapping, m_ctx] = contract(*_current_graph, clustering, false, std::move(_contraction_m_ctx));
    STOP_TIMER();
    _contraction_m_ctx = std::move(m_ctx);

    const bool shrunk{1.0 * c_graph.n() < (1.0 - _c_ctx.shrink_factor_abortion_threshold) * _current_graph->n()};

    _hierarchy.push_back(std::move(c_graph));
    _mapping.push_back(std::move(c_mapping));
    _current_graph = &_hierarchy.back();

    return {_current_graph, shrunk};
  };

  PartitionedGraph uncoarsen(PartitionedGraph &&p_graph) final {
    ASSERT(&p_graph.graph() == _current_graph);
    ASSERT(!empty()) << size();

    START_TIMER(TIMER_ALLOCATION);
    auto mapping{std::move(_mapping.back())}; // TODO idea: re-use mapping as partition vector
    _mapping.pop_back();
    _hierarchy.pop_back(); // destroys the graph wrapped in p_graph, but partition access is still ok
    _current_graph = empty() ? &_input_graph : &_hierarchy.back();
    ASSERT(mapping.size() == _current_graph->n()) << V(mapping.size()) << V(_current_graph->n());

    StaticArray<BlockID> partition(_current_graph->n());
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

  const Graph *coarsest_graph() const final { return _current_graph; }
  std::size_t size() const final { return _hierarchy.size(); }
  void set_community_structure(std::vector<BlockID>) final {}
  void initialize(const Graph *) final {}
  const CoarseningContext &context() const { return _c_ctx; }

private:
  const Graph &_input_graph;
  const Graph *_current_graph;
  std::vector<Graph> _hierarchy;
  std::vector<scalable_vector<NodeID>> _mapping;
  LabelPropagationClustering _label_propagation_core;
  const CoarseningContext &_c_ctx;
  ContractionMemoryContext _contraction_m_ctx{};
};
} // namespace kaminpar
