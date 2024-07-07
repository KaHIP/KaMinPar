/*******************************************************************************
 * Delegate for the graph class.
 *
 * @file:   graph_delegate.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <cstddef>
#include <utility>

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/ranges.h"

namespace kaminpar::shm {

template <class Graph> class GraphDelegate {
public:
  GraphDelegate(const Graph *graph) : _graph(graph) {}

  [[nodiscard]] inline bool initialized() const {
    return _graph != nullptr;
  }

  [[nodiscard]] inline const Graph &graph() const {
    return *_graph;
  }

  template <typename Lambda> decltype(auto) reified(Lambda &&l) const {
    return _graph->reified(std::forward<Lambda>(l));
  }

  //
  // Size of the graph
  //

  [[nodiscard]] inline NodeID n() const {
    return _graph->n();
  }

  [[nodiscard]] inline EdgeID m() const {
    return _graph->m();
  }

  //
  // Node and edge weights
  //

  [[nodiscard]] inline bool is_node_weighted() const {
    return _graph->is_node_weighted();
  }

  [[nodiscard]] inline NodeWeight node_weight(const NodeID u) const {
    return _graph->node_weight(u);
  }

  [[nodiscard]] inline NodeWeight total_node_weight() const {
    return _graph->total_node_weight();
  }

  [[nodiscard]] inline NodeWeight max_node_weight() const {
    return _graph->max_node_weight();
  }

  [[nodiscard]] inline bool is_edge_weighted() const {
    return _graph->is_edge_weighted();
  }

  [[nodiscard]] inline EdgeWeight total_edge_weight() const {
    return _graph->total_edge_weight();
  }

  //
  // Iterators for nodes / edges
  //

  [[nodiscard]] inline IotaRange<NodeID> nodes() const {
    return _graph->nodes();
  }

  [[nodiscard]] inline IotaRange<EdgeID> edges() const {
    return _graph->edges();
  }

  [[nodiscard]] inline IotaRange<EdgeID> incident_edges(const NodeID u) const {
    return _graph->incident_edges(u);
  }

  //
  // Node degree
  //

  [[nodiscard]] inline NodeID max_degree() const {
    return _graph->max_degree();
  }

  [[nodiscard]] inline NodeID degree(const NodeID u) const {
    return _graph->degree(u);
  }

  //
  // Graph operations
  //

  template <typename Lambda> inline void adjacent_nodes(const NodeID u, Lambda &&l) const {
    _graph->adjacent_nodes(u, std::forward<Lambda>(l));
  }

  template <typename Lambda> inline void neighbors(const NodeID u, Lambda &&l) const {
    _graph->neighbors(u, std::forward<Lambda>(l));
  }

  template <typename Lambda>
  inline void neighbors(const NodeID u, const NodeID max_num_neighbors, Lambda &&l) const {
    _graph->neighbors(u, max_num_neighbors, std::forward<Lambda>(l));
  }

  //
  // Parallel iteration
  //

  template <typename Lambda> inline void pfor_nodes(Lambda &&l) const {
    return _graph->pfor_nodes(std::forward<Lambda>(l));
  }

  template <typename Lambda> inline void pfor_edges(Lambda &&l) const {
    return _graph->pfor_edges(std::forward<Lambda>(l));
  }

  //
  // Graph permutation
  //

  [[nodiscard]] inline bool permuted() const {
    return _graph->permuted();
  }

  [[nodiscard]] inline NodeID map_original_node(const NodeID u) const {
    return _graph->map_original_node(u);
  }

  //
  // Degree buckets
  //

  [[nodiscard]] inline bool sorted() const {
    return _graph->sorted();
  }

  [[nodiscard]] inline std::size_t number_of_buckets() const {
    return _graph->number_of_buckets();
  }

  [[nodiscard]] inline std::size_t bucket_size(const std::size_t bucket) const {
    return _graph->bucket_size(bucket);
  }

  [[nodiscard]] inline NodeID first_node_in_bucket(const std::size_t bucket) const {
    return _graph->first_node_in_bucket(bucket);
  }

  [[nodiscard]] inline NodeID first_invalid_node_in_bucket(const std::size_t bucket) const {
    return _graph->first_invalid_node_in_bucket(bucket);
  }

protected:
  const Graph *_graph;
};

} // namespace kaminpar::shm
