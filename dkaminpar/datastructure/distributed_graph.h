/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
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

#include "dkaminpar/distributed_definitions.h"
#include "kaminpar/datastructure/graph.h"
#include "kaminpar/parallel.h"

#include <definitions.h>
#include <ranges>
#include <tbb/parallel_for.h>
#include <vector>

namespace dkaminpar {
class DistributedGraph {
  SET_DEBUG(true);

public:
  DistributedGraph() = default;

  DistributedGraph(const DNodeID global_n, const DEdgeID global_m, const DNodeID ghost_n, const DNodeID offset_n,
                   const DEdgeID offset_m, scalable_vector<DNodeID> node_distribution,
                   scalable_vector<DEdgeID> edge_distribution, scalable_vector<DEdgeID> nodes,
                   scalable_vector<DNodeID> edges, scalable_vector<DNodeWeight> node_weights,
                   scalable_vector<DEdgeWeight> edge_weights, scalable_vector<PEID> ghost_owner,
                   scalable_vector<DNodeID> ghost_to_global, std::unordered_map<DNodeID, DNodeID> global_to_ghost)
      : _global_n{global_n},
        _global_m{global_m},
        _ghost_n{ghost_n},
        _offset_n{offset_n},
        _offset_m{offset_m},
        _node_distribution{std::move(node_distribution)},
        _edge_distribution{std::move(edge_distribution)},
        _nodes{std::move(nodes)},
        _edges{std::move(edges)},
        _node_weights{std::move(node_weights)},
        _edge_weights{std::move(edge_weights)},
        _ghost_owner{std::move(ghost_owner)},
        _ghost_to_global{std::move(ghost_to_global)},
        _global_to_ghost{std::move(global_to_ghost)} {
    init_total_node_weight();
  }

  DistributedGraph(const DistributedGraph &) = delete;
  DistributedGraph &operator=(const DistributedGraph &) = delete;
  DistributedGraph(DistributedGraph &&) noexcept = default;
  DistributedGraph &operator=(DistributedGraph &&) noexcept = default;

  // Graph size
  [[nodiscard]] inline DNodeID global_n() const { return _global_n; }
  [[nodiscard]] inline DNodeID ghost_n() const { return _ghost_n; }
  [[nodiscard]] inline DNodeID total_n() const { return ghost_n() + n(); }
  [[nodiscard]] inline DNodeID n() const { return _nodes.size() - 1; }
  [[nodiscard]] inline DEdgeID global_m() const { return _global_m; }
  [[nodiscard]] inline DEdgeID m() const { return _edges.size(); }

  [[nodiscard]] inline DNodeID offset_n() const { return _offset_n; }
  [[nodiscard]] inline DEdgeID offset_m() const { return _offset_m; }

  [[nodiscard]] inline DNodeWeight total_node_weight() const { return _total_node_weight; }
  [[nodiscard]] inline DNodeWeight max_node_weight() const { return _max_node_weight; }

  // Node type
  [[nodiscard]] inline bool is_ghost_node(const DNodeID u) const {
    ASSERT(u < total_n());
    return u >= n();
  }
  [[nodiscard]] inline bool is_owned_node(const DNodeID u) const {
    ASSERT(u < total_n());
    return u < n();
  }

  // Distributed info
  [[nodiscard]] inline PEID ghost_owner(const DNodeID u) const {
    ASSERT(is_ghost_node(u));
    return _ghost_owner[u - n()];
  }

  [[nodiscard]] inline DNodeID global_node(const DNodeID local_u) const {
    ASSERT(local_u < total_n()) << V(local_u) << V(total_n()) << V(n()) << V(ghost_n());

    if (is_owned_node(local_u)) {
      return _offset_n + local_u;
    } else {
      return _ghost_to_global[local_u - n()];
    }
  }

  [[nodiscard]] inline DNodeID local_node(const DNodeID global_u) const {
    if (offset_n() <= global_u && global_u < offset_n() + n()) {
      return global_u - offset_n();
    } else {
      ASSERT(_global_to_ghost.contains(global_u));
      return _global_to_ghost[global_u];
    }
  }

  // Access methods
  [[nodiscard]] inline DNodeWeight node_weight(const DNodeID u) const {
    ASSERT(u < total_n());
    ASSERT(u < _node_weights.size());
    return _node_weights[u];
  }

  [[nodiscard]] inline DEdgeWeight edge_weight(const DEdgeID e) const {
    ASSERT(e < m());
    ASSERT(e < _edge_weights.size());
    return _edge_weights[e];
  }

  // Graph structure
  [[nodiscard]] inline DEdgeID first_edge(const DNodeID u) const {
    ASSERT(u < n());
    return _nodes[u];
  }

  [[nodiscard]] inline DEdgeID first_invalid_edge(const DNodeID u) const {
    ASSERT(u < n());
    return _nodes[u + 1];
  }

  [[nodiscard]] inline DNodeID edge_target(const DEdgeID e) const {
    ASSERT(e < m());
    return _edges[e];
  }

  [[nodiscard]] inline DNodeID degree(const DNodeID u) const {
    ASSERT(is_owned_node(u));
    return _nodes[u + 1] - _nodes[u];
  }

  [[nodiscard]] const auto &node_distribution() const { return _node_distribution; }
  [[nodiscard]] const auto &edge_distribution() const { return _edge_distribution; }

  void print_info() const { DLOG << "global_n=" << global_n() << " local_n=" << n() << " ghost_n=" << ghost_n(); }

  PEID find_owner(const DNodeID u) const {
    ASSERT(u < total_n());
    const DNodeID global_u = global_node(u);
    auto it = std::upper_bound(_node_distribution.begin() + 1, _node_distribution.end(), global_u);
    ASSERT(it != _node_distribution.end());
    return static_cast<PEID>(std::distance(_node_distribution.begin(), it) - 1);
  }

  // Parallel iteration
  template<typename Lambda>
  inline void pfor_nodes(const DNodeID from, const DNodeID to, Lambda &&l) const {
    tbb::parallel_for(from, to, std::forward<Lambda &&>(l));
  }

  template<typename Lambda>
  inline void pfor_nodes_range(const DNodeID from, const DNodeID to, Lambda &&l) const {
    tbb::parallel_for(tbb::blocked_range<DNodeID>(from, to), std::forward<Lambda &&>(l));
  }

  template<typename Lambda>
  inline void pfor_nodes(Lambda &&l) const {
    pfor_nodes(0, n(), std::forward<Lambda &&>(l));
  }

  template<typename Lambda>
  inline void pfor_nodes_range(Lambda &&l) const {
    pfor_nodes_range(0, n(), std::forward<Lambda &&>(l));
  }

  template<typename Lambda>
  inline void pfor_ghost_nodes(Lambda &&l) const {
    tbb::parallel_for(static_cast<DNodeID>(0), ghost_n(), std::forward<Lambda &&>(l));
  }

  template<typename Lambda>
  inline void pfor_all_nodes(Lambda &&l) const {
    tbb::parallel_for(static_cast<DNodeID>(0), total_n(), std::forward<Lambda &&>(l));
  }

  template<typename Lambda>
  inline void pfor_edges(Lambda &&l) const {
    tbb::parallel_for(static_cast<DEdgeID>(0), m(), std::forward<Lambda &&>(l));
  }

  // Iterators for nodes / edges
  [[nodiscard]] inline auto nodes() const { return std::views::iota(static_cast<DNodeID>(0), n()); }
  [[nodiscard]] inline auto ghost_nodes() const { return std::views::iota(n(), total_n()); }
  [[nodiscard]] inline auto all_nodes() const { return std::views::iota(static_cast<DNodeID>(0), total_n()); }
  [[nodiscard]] inline auto edges() const { return std::views::iota(static_cast<DEdgeID>(0), m()); }
  [[nodiscard]] inline auto incident_edges(const DNodeID u) const { return std::views::iota(_nodes[u], _nodes[u + 1]); }
  [[nodiscard]] inline auto adjacent_nodes(const DNodeID u) const {
    return std::views::iota(_nodes[u], _nodes[u + 1]) |
           std::views::transform([this](const DEdgeID e) { return this->edge_target(e); });
  }
  [[nodiscard]] inline auto adjacent_nodes_global(const DNodeID u) const {
    return std::views::iota(_nodes[u], _nodes[u + 1]) |
           std::views::transform([this](const DEdgeID e) { return this->global_node(this->edge_target(e)); });
  }
  [[nodiscard]] inline auto neighbors(const DNodeID u) const {
    return std::views::iota(_nodes[u], _nodes[u + 1]) |
           std::views::transform([this](const DEdgeID e) { return std::make_pair(e, this->edge_target(e)); });
  }
  [[nodiscard]] inline auto neighbors_global(const DNodeID u) const {
    return std::views::iota(_nodes[u], _nodes[u + 1]) | std::views::transform([this](const DEdgeID e) {
             return std::make_pair(e, this->global_node(this->edge_target(e)));
           });
  }

  // Degree buckets -- right now only for compatibility to shared memory graph data structure
  [[nodiscard]] inline std::size_t bucket_size(const std::size_t) const { return n(); }
  [[nodiscard]] inline DNodeID first_node_in_bucket(const std::size_t) const { return 0; }
  [[nodiscard]] inline DNodeID first_invalid_node_in_bucket(const std::size_t) const { return n(); }
  [[nodiscard]] inline std::size_t number_of_buckets() const { return 1; }
  [[nodiscard]] inline bool sorted() const { return false; }

private:
  void init_total_node_weight() {
    // TODO ignore ghost nodes
    //    _total_node_weight = shm::parallel::accumulate(_node_weights);
    //    _max_node_weight = shm::parallel::max_element(_node_weights);
    for (const DNodeID u : nodes()) {
      _total_node_weight += node_weight(u);
      _max_node_weight = std::max(_max_node_weight, node_weight(u));
    }
  }

  DNodeID _global_n{0};
  DEdgeID _global_m{0};
  DNodeID _ghost_n{0};
  DNodeID _offset_n{0};
  DEdgeID _offset_m{0};

  DNodeWeight _total_node_weight{};
  DNodeWeight _max_node_weight{};

  scalable_vector<DNodeID> _node_distribution;
  scalable_vector<DEdgeID> _edge_distribution;

  scalable_vector<DEdgeID> _nodes;
  scalable_vector<DNodeID> _edges;
  scalable_vector<DNodeWeight> _node_weights;
  scalable_vector<DEdgeWeight> _edge_weights;

  scalable_vector<PEID> _ghost_owner;
  scalable_vector<DNodeID> _ghost_to_global;
  mutable std::unordered_map<DNodeID, DNodeID> _global_to_ghost;
};

class DistributedPartitionedGraph {
  using block_weights_vector = scalable_vector<shm::parallel::IntegralAtomicWrapper<DBlockWeight>>;

public:
  DistributedPartitionedGraph(const DistributedGraph *graph, const DBlockID k, scalable_vector<DBlockID> partition,
                              block_weights_vector block_weights)
      : _graph{graph},
        _k{k},
        _partition{std::move(partition)},
        _block_weights{std::move(block_weights)} {
    ASSERT(_partition.size() == _graph->total_n());
    ASSERT([&] {
      for (const DBlockID b : _partition) { ASSERT(b < _k); }
    });

    DLOG << V(_block_weights);
  }

  DistributedPartitionedGraph(DistributedGraph *graph, const DBlockID k)
      : DistributedPartitionedGraph(graph, k, scalable_vector<DBlockID>(graph->total_n()),
                                    block_weights_vector(graph->total_n())) {}

  DistributedPartitionedGraph() : _graph{nullptr}, _k{0}, _partition{} {}

  DistributedPartitionedGraph(const DistributedPartitionedGraph &) = delete;
  DistributedPartitionedGraph &operator=(const DistributedPartitionedGraph &) = delete;
  DistributedPartitionedGraph(DistributedPartitionedGraph &&) noexcept = default;
  DistributedPartitionedGraph &operator=(DistributedPartitionedGraph &&) noexcept = default;

  [[nodiscard]] inline DNodeID global_n() const { return _graph->global_n(); }
  [[nodiscard]] inline DNodeID ghost_n() const { return _graph->ghost_n(); }
  [[nodiscard]] inline DNodeID total_n() const { return _graph->total_n(); }
  [[nodiscard]] inline DNodeID n() const { return _graph->n(); }
  [[nodiscard]] inline DEdgeID global_m() const { return _graph->global_m(); }
  [[nodiscard]] inline DEdgeID m() const { return _graph->m(); }
  [[nodiscard]] inline DNodeID offset_n() const { return _graph->offset_n(); }
  [[nodiscard]] inline DEdgeID offset_m() const { return _graph->offset_m(); }
  [[nodiscard]] inline bool is_ghost_node(const DNodeID u) const { return _graph->is_ghost_node(u); }
  [[nodiscard]] inline bool is_owned_node(const DNodeID u) const { return _graph->is_owned_node(u); }
  [[nodiscard]] inline PEID ghost_owner(const DNodeID u) const { return _graph->ghost_owner(u); }
  [[nodiscard]] inline DNodeID global_node(const DNodeID local_u) const { return _graph->global_node(local_u); }
  [[nodiscard]] inline DNodeID local_node(const DNodeID global_u) const { return _graph->local_node(global_u); }
  [[nodiscard]] inline PEID find_owner(const DNodeID u) const { return _graph->find_owner(u); }

  [[nodiscard]] auto edge_weight(const DEdgeID e) const { return _graph->edge_weight(e); }
  [[nodiscard]] auto node_weight(const DNodeID u) const { return _graph->node_weight(u); }

  [[nodiscard]] inline DNodeWeight total_node_weight() const { return _graph->total_node_weight(); }
  [[nodiscard]] inline DNodeWeight max_node_weight() const { return _graph->max_node_weight(); }

  [[nodiscard]] inline DNodeID degree(const DNodeID u) const { return _graph->degree(u); }

  [[nodiscard]] DBlockID k() const { return _k; }

  template<typename Lambda>
  inline void pfor_nodes(const DNodeID from, const DNodeID to, Lambda &&l) const {
    _graph->pfor_nodes(from, to, std::forward<Lambda &&>(l));
  }

  template<typename Lambda>
  inline void pfor_nodes(Lambda &&l) const {
    _graph->pfor_nodes(std::forward<Lambda &&>(l));
  }

  template<typename Lambda>
  inline void pfor_nodes_range(const DNodeID from, const DNodeID to, Lambda &&l) const {
    _graph->pfor_nodes_range(from, to, std::forward<Lambda &&>(l));
  }

  template<typename Lambda>
  inline void pfor_nodes_range(Lambda &&l) const {
    _graph->pfor_nodes_range(std::forward<Lambda &&>(l));
  }

  template<typename Lambda>
  inline void pfor_all_nodes(Lambda &&l) const {
    _graph->pfor_all_nodes(std::forward<Lambda &&>(l));
  }

  template<typename Lambda>
  inline void pfor_ghost_nodes(Lambda &&l) const {
    _graph->pfor_ghost_nodes(std::forward<Lambda &&>(l));
  }

  template<typename Lambda>
  inline void pfor_blocks(Lambda &&l) const {
    tbb::parallel_for(static_cast<DBlockID>(0), k(), std::forward<Lambda &&>(l));
  }

  [[nodiscard]] inline auto blocks() const { return std::views::iota(static_cast<DBlockID>(0), k()); }

  [[nodiscard]] DBlockID block(const DNodeID u) const {
    ASSERT(u < _partition.size());
    return _partition[u];
  }

  void set_block(const DNodeID u, const DBlockID b) {
    ASSERT(u < _graph->total_n());
    const DNodeWeight u_weight = _graph->node_weight(u);

    _block_weights[_partition[u]] -= u_weight;
    _block_weights[b] += u_weight;
    _partition[u] = b;
  }

  [[nodiscard]] inline DBlockWeight block_weight(const DBlockID b) const {
    ASSERT(b < k());
    ASSERT(b < _block_weights.size());
    return _block_weights[b];
  }

  void set_block_weight(const DBlockID b, const DBlockWeight weight) {
    ASSERT(b < k());
    ASSERT(b < _block_weights.size());
    _block_weights[b] = weight;
  }

  [[nodiscard]] scalable_vector<DBlockWeight> block_weights_copy() const {
    scalable_vector<DBlockWeight> copy(k());
    pfor_blocks([&](const DBlockID b) { copy[b] = block_weight(b); });
    return copy;
  }

  [[nodiscard]] auto &&take_block_weights() {
    return std::move(_block_weights);
  }

  [[nodiscard]] scalable_vector<DBlockID> partition_copy() const {
    scalable_vector<DBlockID> copy(n());
    pfor_nodes([&](const DNodeID u) { copy[u] = block(u); });
    return copy;
  }

  [[nodiscard]] inline auto nodes() const { return _graph->nodes(); }
  [[nodiscard]] inline auto ghost_nodes() const { return _graph->ghost_nodes(); }
  [[nodiscard]] inline auto all_nodes() const { return _graph->all_nodes(); }
  [[nodiscard]] inline auto edges() const { return _graph->edges(); }
  [[nodiscard]] inline auto incident_edges(const DNodeID u) const { return _graph->incident_edges(u); }
  [[nodiscard]] inline auto adjacent_nodes(const DNodeID u) const { return _graph->adjacent_nodes(u); }
  [[nodiscard]] inline auto adjacent_nodes_global(const DNodeID u) const { return _graph->adjacent_nodes_global(u); }
  [[nodiscard]] inline auto neighbors(const DNodeID u) const { return _graph->neighbors(u); }
  [[nodiscard]] inline auto neighbors_global(const DNodeID u) const { return _graph->neighbors_global(u); }

private:
  const DistributedGraph *_graph;
  DBlockID _k;
  scalable_vector<DBlockID> _partition;

  scalable_vector<kaminpar::parallel::IntegralAtomicWrapper<DBlockWeight>> _block_weights;
};

namespace debug {
bool validate_partition_state(const DistributedPartitionedGraph &p_graph, MPI_Comm comm = MPI_COMM_WORLD);
}
} // namespace dkaminpar