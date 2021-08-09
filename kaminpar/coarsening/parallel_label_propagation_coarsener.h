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

#include "algorithm/graph_contraction.h"
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
struct LabelPropagationClusteringConfig : public LabelPropagationConfig {
  using ClusterID = BlockID;
  using ClusterWeight = BlockWeight;
  static constexpr bool kUseHardWeightConstraint = false;
  static constexpr bool kReportEmptyClusters = true;
};

class LabelPropagationClustering final
    : public LabelPropagation<LabelPropagationClustering, LabelPropagationClusteringConfig> {
  SET_DEBUG(false);

  using Base = LabelPropagation<LabelPropagationClustering, LabelPropagationClusteringConfig>;
  friend Base;

  static constexpr std::size_t kInfiniteIterations{std::numeric_limits<std::size_t>::max()};

public:
  LabelPropagationClustering(const NodeID max_n, const double shrink_factor,
                             const LabelPropagationCoarseningContext &lp_ctx)
      : Base{max_n, max_n},
        _shrink_factor{shrink_factor},
        _lp_ctx{lp_ctx},
        _max_cluster_weight{kInvalidBlockWeight} {
    _clustering.resize(max_n);
    _favored_clustering.resize(max_n);
    set_max_degree(lp_ctx.large_degree_threshold);
    set_max_num_neighbors(lp_ctx.max_num_neighbors);
  }

  const scalable_vector<NodeID> &cluster(const Graph &graph, const NodeWeight max_cluster_weight,
                                         const std::size_t max_iterations = kInfiniteIterations) {
    ASSERT(_clustering.size() >= graph.n());

    initialize(&graph);
    _max_cluster_weight = max_cluster_weight;
    _current_size = graph.n();
    _target_size = static_cast<NodeID>(_shrink_factor * _current_size);
    NodeID total_num_emptied_clusters = 0;

    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
      //SCOPED_FINE_TIMER(std::string("Iteration ") + std::to_string(iteration)); // TODO

      const auto [num_moved_nodes, num_emptied_clusters] = randomized_iteration();
      _current_size -= num_emptied_clusters;
      total_num_emptied_clusters += num_emptied_clusters;
      if (num_moved_nodes == 0) { break; }
    }

    if (_lp_ctx.should_merge_nonadjacent_clusters(_graph->n(), _graph->n() - total_num_emptied_clusters)) {
      DBG << "Empty clusters after LP: " << total_num_emptied_clusters << " of " << _graph->n();
      TIMED_SCOPE("2-hop Clustering") { join_singleton_clusters_by_favored_cluster(total_num_emptied_clusters); };
    }

    return _clustering;
  }

private:
  void join_singleton_clusters_by_favored_cluster(const NodeID emptied_clusters) {
    const auto desired_no_of_coarse_nodes = _graph->n() * (1.0 - _lp_ctx.merge_nonadjacent_clusters_threshold);
    parallel::IntegralAtomicWrapper<NodeID> current_no_of_coarse_nodes = _graph->n() - emptied_clusters;

    tbb::parallel_for(static_cast<NodeID>(0), _graph->n(), [&](const NodeID u) {
      if (current_no_of_coarse_nodes <= desired_no_of_coarse_nodes) { return; }

      const NodeID leader = _clustering[u]; // first && part avoids _cluster_weights cache miss
      const bool singleton = leader == u && _cluster_weights[u] == _graph->node_weight(u);

      if (singleton) {
        NodeID favored_leader = _favored_clustering[u];
        if (_lp_ctx.merge_isolated_clusters && u == favored_leader) { favored_leader = 0; }

        do {
          NodeID expected_value = favored_leader;
          if (_favored_clustering[favored_leader].compare_exchange_strong(expected_value, u)) {
            break; // if this worked, we replaced favored_leader with u
          }

          // if this didn't work, there is another node that has the same favored leader -> try to join that nodes
          // cluster
          const NodeID partner = expected_value;
          if (_favored_clustering[favored_leader].compare_exchange_strong(expected_value, favored_leader)) {
            if (_cluster_weights[partner] + _graph->node_weight(u) < _max_cluster_weight) {
              _clustering[u] = partner;
              _cluster_weights[partner] += _graph->node_weight(u);
              --current_no_of_coarse_nodes;
            }
            break;
          }
        } while (true);
      }
    });
  }

  // called from Base class
  void reset_node_state(const NodeID u) {
    _clustering[u] = u;
    _favored_clustering[u] = u;
  }

  [[nodiscard]] NodeID cluster(const NodeID u) const { return _clustering[u]; }
  void set_cluster(const NodeID u, const NodeID cluster) { _clustering[u] = cluster; }
  [[nodiscard]] NodeID num_clusters() const { return _graph->n(); }
  [[nodiscard]] NodeWeight initial_cluster_weight(const NodeID cluster) const { return _graph->node_weight(cluster); }
  [[nodiscard]] NodeWeight max_cluster_weight(const NodeID) const { return _max_cluster_weight; }

  [[nodiscard]] bool accept_cluster(const Base::ClusterSelectionState &state) const {
    return (state.current_gain > state.best_gain ||
            (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
           (state.current_cluster_weight + state.u_weight < max_cluster_weight(state.current_cluster) ||
            state.current_cluster == state.initial_cluster);
  }

  void set_favored_cluster(const NodeID u, const NodeID cluster) { _favored_clustering[u] = cluster; }

  [[nodiscard]] bool should_stop(const NodeID num_emptied_clusters) const {
    return _current_size - num_emptied_clusters < _target_size;
  }

  double _shrink_factor;
  const LabelPropagationCoarseningContext &_lp_ctx;
  scalable_vector<NodeID> _clustering;
  scalable_vector<parallel::IntegralAtomicWrapper<NodeID>> _favored_clustering;
  NodeWeight _max_cluster_weight;

  NodeID _current_size;
  NodeID _target_size;
};

class ParallelLabelPropagationCoarsener : public Coarsener {
public:
  ParallelLabelPropagationCoarsener(const Graph &input_graph, const CoarseningContext &c_ctx)
      : _input_graph{input_graph},
        _current_graph{&input_graph},
        _label_propagation_core{input_graph.n(), c_ctx.enforce_contraction_limit ? 0.5 : 0.0, c_ctx.lp},
        _c_ctx{c_ctx} {}

  ParallelLabelPropagationCoarsener(const ParallelLabelPropagationCoarsener &) = delete;
  ParallelLabelPropagationCoarsener &operator=(const ParallelLabelPropagationCoarsener) = delete;
  ParallelLabelPropagationCoarsener(ParallelLabelPropagationCoarsener &&) = delete;
  ParallelLabelPropagationCoarsener &operator=(ParallelLabelPropagationCoarsener &&) = delete;

  using Coarsener::coarsen;

  std::pair<const Graph *, bool> coarsen(const std::function<NodeWeight(NodeID)> &cb_max_cluster_weight) final {
    //SCOPED_FINE_TIMER(std::string("Level ") + std::to_string(_hierarchy.size())); // TODO

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
    //SCOPED_FINE_TIMER(std::string("Level ") + std::to_string(_hierarchy.size())); // TODO

    START_TIMER("Allocation");
    auto mapping{std::move(_mapping.back())};
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
