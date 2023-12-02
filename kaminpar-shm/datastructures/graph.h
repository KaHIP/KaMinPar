/*******************************************************************************
 * Wrapper class that delegates all function calls to a concrete graph object.
 *
 * Most function calls are resolved via dynamic binding. Thus, they should not
 * be used when performance is critical. Instead, use an downcast and templatize
 * tight loops.
 *
 * @file:   graph.h
 * @author: Daniel Seemaier
 * @date:   17.11.2023
 ******************************************************************************/
#pragma once

#include <numeric>
#include <utility>
#include <vector>

#include <kassert/kassert.hpp>

#include "kaminpar-shm/datastructures/abstract_graph.h"
#include "kaminpar-shm/datastructures/compressed_graph.h"
#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/definitions.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/degree_buckets.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/ranges.h"

namespace kaminpar::shm {

class Graph : public AbstractGraph {
public:
  // Data types used by this graph
  using AbstractGraph::EdgeID;
  using AbstractGraph::EdgeWeight;
  using AbstractGraph::NodeID;
  using AbstractGraph::NodeWeight;

  Graph() = default;

  Graph(std::unique_ptr<AbstractGraph> graph);

  Graph(const Graph &) = delete;
  Graph &operator=(const Graph &) = delete;

  Graph(Graph &&) noexcept = default;
  Graph &operator=(Graph &&) noexcept = default;

  ~Graph() override = default;

  [[nodiscard]] inline StaticArray<EdgeID> &raw_nodes() final {
    return _underlying_graph->raw_nodes();
  }

  [[nodiscard]] inline const StaticArray<EdgeID> &raw_nodes() const final {
    return _underlying_graph->raw_nodes();
  }

  [[nodiscard]] inline StaticArray<NodeID> &raw_edges() final {
    return _underlying_graph->raw_edges();
  }

  [[nodiscard]] inline const StaticArray<EdgeID> &raw_edges() const final {
    return _underlying_graph->raw_edges();
  }

  [[nodiscard]] inline StaticArray<NodeWeight> &raw_node_weights() final {
    return _underlying_graph->raw_node_weights();
  }

  [[nodiscard]] inline const StaticArray<NodeWeight> &raw_node_weights() const final {
    return _underlying_graph->raw_node_weights();
  }

  [[nodiscard]] inline StaticArray<EdgeWeight> &raw_edge_weights() final {
    return _underlying_graph->raw_edge_weights();
  }

  [[nodiscard]] inline const StaticArray<EdgeWeight> &raw_edge_weights() const final {
    return _underlying_graph->raw_edge_weights();
  }

  [[nodiscard]] inline StaticArray<EdgeID> &&take_raw_nodes() final {
    return _underlying_graph->take_raw_nodes();
  }

  [[nodiscard]] inline StaticArray<NodeID> &&take_raw_edges() final {
    return _underlying_graph->take_raw_edges();
  }

  [[nodiscard]] inline StaticArray<NodeWeight> &&take_raw_node_weights() final {
    return _underlying_graph->take_raw_node_weights();
  }

  [[nodiscard]] inline StaticArray<EdgeWeight> &&take_raw_edge_weights() final {
    return _underlying_graph->take_raw_edge_weights();
  }

  // Size of the graph
  [[nodiscard]] inline NodeID n() const final {
    return _underlying_graph->n();
  }

  [[nodiscard]] inline EdgeID m() const final {
    return _underlying_graph->m();
  }

  [[nodiscard]] inline EdgeID compute_max_degree() const {
    if (const auto *graph = dynamic_cast<const CSRGraph *>(_underlying_graph.get());
        graph != nullptr) {
      return parallel::max_difference(graph->raw_nodes().begin(), graph->raw_nodes().end());
    }

    if (const auto *graph = dynamic_cast<const CompressedGraph *>(_underlying_graph.get());
        graph != nullptr) {
      EdgeID max_degree = 0;

      for (const NodeID node : graph->nodes()) {
        max_degree = std::max(max_degree, graph->degree(node));
      }

      return max_degree;
    }

    __builtin_unreachable();
  }

  // Node and edge weights
  [[nodiscard]] inline bool is_node_weighted() const final {
    return _underlying_graph->is_node_weighted();
  }

  [[nodiscard]] inline NodeWeight node_weight(const NodeID u) const final {
    return _underlying_graph->node_weight(u);
  }

  [[nodiscard]] inline NodeWeight max_node_weight() const final {
    return _underlying_graph->max_node_weight();
  }

  [[nodiscard]] inline NodeWeight total_node_weight() const final {
    return _underlying_graph->total_node_weight();
  }

  [[nodiscard]] inline bool is_edge_weighted() const final {
    return _underlying_graph->is_edge_weighted();
  }

  [[nodiscard]] inline EdgeWeight edge_weight(const EdgeID e) const final {
    return _underlying_graph->edge_weight(e);
  }

  [[nodiscard]] inline EdgeWeight total_edge_weight() const final {
    return _underlying_graph->total_edge_weight();
  }

  // Low-level access to the graph structure
  [[nodiscard]] inline NodeID edge_target(const EdgeID e) const final {
    return _underlying_graph->edge_target(e);
  }

  [[nodiscard]] inline EdgeID first_edge(const NodeID u) const final {
    return _underlying_graph->first_edge(u);
  }

  [[nodiscard]] inline EdgeID first_invalid_edge(const NodeID u) const final {
    return _underlying_graph->first_invalid_edge(u);
  }

  [[nodiscard]] inline NodeID degree(const NodeID u) const final {
    return _underlying_graph->degree(u);
  }

  // Parallel iteration
  template <typename Lambda> inline void pfor_nodes(Lambda &&l) const {
    if (const auto *graph = dynamic_cast<const CSRGraph *>(_underlying_graph.get());
        graph != nullptr) {
      graph->pfor_nodes(std::forward<Lambda>(l));
      return;
    }

    if (const auto *graph = dynamic_cast<const CompressedGraph *>(_underlying_graph.get());
        graph != nullptr) {
      graph->pfor_nodes(std::forward<Lambda>(l));
      return;
    }

    __builtin_unreachable();
  }

  template <typename Lambda> inline void pfor_edges(Lambda &&l) const {
    if (const auto *graph = dynamic_cast<const CSRGraph *>(_underlying_graph.get());
        graph != nullptr) {
      graph->pfor_edges(std::forward<Lambda>(l));
      return;
    }

    if (const auto *graph = dynamic_cast<const CompressedGraph *>(_underlying_graph.get());
        graph != nullptr) {
      graph->pfor_edges(std::forward<Lambda>(l));
      return;
    }

    __builtin_unreachable();
  }

  // Iterators for nodes / edges
  [[nodiscard]] inline IotaRange<NodeID> nodes() const final {
    return _underlying_graph->nodes();
  }

  [[nodiscard]] inline IotaRange<EdgeID> edges() const final {
    return _underlying_graph->edges();
  }

  [[nodiscard]] inline decltype(auto) incident_edges(const NodeID u) const {
    if (const auto *graph = dynamic_cast<const CSRGraph *>(_underlying_graph.get());
        graph != nullptr) {
      return graph->incident_edges(u);
    }

    if (const auto *graph = dynamic_cast<const CompressedGraph *>(_underlying_graph.get());
        graph != nullptr) {
      return graph->incident_edges(u);
    }

    __builtin_unreachable();
  }

  [[nodiscard]] inline decltype(auto) adjacent_nodes(const NodeID u) const {
    if (const auto *graph = dynamic_cast<const CSRGraph *>(_underlying_graph.get());
        graph != nullptr) {
      return graph->adjacent_nodes(u);
    }

    if (const auto *graph = dynamic_cast<const CompressedGraph *>(_underlying_graph.get());
        graph != nullptr) {
      return graph->adjacent_nodes(u);
    }

    __builtin_unreachable();
  }

  [[nodiscard]] inline decltype(auto) neighbors(const NodeID u) const {
    if (const auto *graph = dynamic_cast<const CSRGraph *>(_underlying_graph.get());
        graph != nullptr) {
      return graph->neighbors(u);
    }

    if (const auto *graph = dynamic_cast<const CompressedGraph *>(_underlying_graph.get());
        graph != nullptr) {
      return graph->neighbors(u);
    }

    __builtin_unreachable();
  }

  [[nodiscard]] inline decltype(auto)
  neighbors(const NodeID u, const NodeID max_neighbor_count) const {
    if (const auto *graph = dynamic_cast<const CSRGraph *>(_underlying_graph.get());
        graph != nullptr) {
      return graph->neighbors(u, max_neighbor_count);
    }

    if (const auto *graph = dynamic_cast<const CompressedGraph *>(_underlying_graph.get());
        graph != nullptr) {
      return graph->neighbors(u, max_neighbor_count);
    }

    __builtin_unreachable();
  }

  template <typename Lambda>
  inline void pfor_neighbors(const NodeID u, const NodeID max_neighbor_count, Lambda &&l) const {
    if (const auto *graph = dynamic_cast<const CSRGraph *>(_underlying_graph.get());
        graph != nullptr) {
      graph->pfor_neighbors(u, max_neighbor_count, std::forward<Lambda>(l));
      return;
    }

    if (const auto *graph = dynamic_cast<const CompressedGraph *>(_underlying_graph.get());
        graph != nullptr) {
      graph->pfor_neighbors(u, max_neighbor_count, std::forward<Lambda>(l));
      return;
    }

    __builtin_unreachable();
  }

  // Graph permutation
  inline void set_permutation(StaticArray<NodeID> permutation) final {
    _underlying_graph->set_permutation(std::move(permutation));
  }

  [[nodiscard]] inline bool permuted() const final {
    return _underlying_graph->permuted();
  }

  [[nodiscard]] inline NodeID map_original_node(const NodeID u) const final {
    return _underlying_graph->map_original_node(u);
  }

  // Degree buckets
  [[nodiscard]] inline std::size_t bucket_size(const std::size_t bucket) const final {
    return _underlying_graph->bucket_size(bucket);
  }

  [[nodiscard]] inline NodeID first_node_in_bucket(const std::size_t bucket) const final {
    return _underlying_graph->first_node_in_bucket(bucket);
  }

  [[nodiscard]] inline NodeID first_invalid_node_in_bucket(const std::size_t bucket) const final {
    return _underlying_graph->first_invalid_node_in_bucket(bucket);
  }

  [[nodiscard]] inline std::size_t number_of_buckets() const final {
    return _underlying_graph->number_of_buckets();
  }

  [[nodiscard]] inline bool sorted() const final {
    return _underlying_graph->sorted();
  }

  inline void update_total_node_weight() final {
    _underlying_graph->update_total_node_weight();
  }

  [[nodiscard]] AbstractGraph *underlying_graph() const {
    return _underlying_graph.get();
  }

private:
  std::unique_ptr<AbstractGraph> _underlying_graph;
};

bool validate_graph(const Graph &graph, bool check_undirected = true, NodeID num_pseudo_nodes = 0);

void print_graph(const Graph &graph);

class GraphDelegate {
public:
  GraphDelegate(const Graph *graph) : _graph(graph) {}

  [[nodiscard]] inline bool initialized() const {
    return _graph != nullptr;
  }

  [[nodiscard]] inline const Graph &graph() const {
    return *_graph;
  }

  [[nodiscard]] inline NodeID n() const {
    return _graph->n();
  }

  [[nodiscard]] inline NodeID last_node() const {
    return n() - 1;
  }

  [[nodiscard]] inline EdgeID m() const {
    return _graph->m();
  }

  [[nodiscard]] inline NodeWeight node_weight(const NodeID u) const {
    return _graph->node_weight(u);
  }

  [[nodiscard]] inline EdgeWeight edge_weight(const EdgeID e) const {
    return _graph->edge_weight(e);
  }

  [[nodiscard]] inline NodeWeight total_node_weight() const {
    return _graph->total_node_weight();
  }

  [[nodiscard]] inline NodeWeight max_node_weight() const {
    return _graph->max_node_weight();
  }

  [[nodiscard]] inline EdgeWeight total_edge_weight() const {
    return _graph->total_edge_weight();
  }

  [[nodiscard]] inline NodeID edge_target(const EdgeID e) const {
    return _graph->edge_target(e);
  }

  [[nodiscard]] inline NodeID degree(const NodeID u) const {
    return _graph->degree(u);
  }

  [[nodiscard]] inline auto nodes() const {
    return _graph->nodes();
  }

  [[nodiscard]] inline auto edges() const {
    return _graph->edges();
  }

  [[nodiscard]] inline decltype(auto) incident_edges(const NodeID u) const {
    return _graph->incident_edges(u);
  }

  [[nodiscard]] inline decltype(auto) adjacent_nodes(const NodeID u) const {
    return _graph->adjacent_nodes(u);
  }

  [[nodiscard]] inline auto neighbors(const NodeID u) const {
    return _graph->neighbors(u);
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

  [[nodiscard]] inline std::size_t number_of_buckets() const {
    return _graph->number_of_buckets();
  }

  [[nodiscard]] inline bool sorted() const {
    return _graph->sorted();
  }

  template <typename Lambda> inline void pfor_nodes(Lambda &&l) const {
    _graph->pfor_nodes(std::forward<Lambda>(l));
  }

  template <typename Lambda> inline void pfor_edges(Lambda &&l) const {
    _graph->pfor_edges(std::forward<Lambda>(l));
  }

protected:
  const Graph *_graph;
};
} // namespace kaminpar::shm
