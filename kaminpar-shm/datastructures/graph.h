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

  // Size of the graph
  [[nodiscard]] inline NodeID n() const final {
    return _underlying_graph->n();
  }

  [[nodiscard]] inline EdgeID m() const final {
    return _underlying_graph->m();
  }

  // Node and edge weights
  [[nodiscard]] inline bool node_weighted() const final {
    return _underlying_graph->node_weighted();
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

  [[nodiscard]] inline bool edge_weighted() const final {
    return _underlying_graph->edge_weighted();
  }

  [[nodiscard]] inline EdgeWeight edge_weight(const EdgeID e) const final {
    return _underlying_graph->edge_weight(e);
  }

  [[nodiscard]] inline EdgeWeight total_edge_weight() const final {
    return _underlying_graph->total_edge_weight();
  }

  // Low-level access to the graph structure
  [[nodiscard]] inline NodeID max_degree() const final {
    return _underlying_graph->max_degree();
  }

  [[nodiscard]] inline NodeID degree(const NodeID u) const final {
    return _underlying_graph->degree(u);
  }

  // Iterators for nodes / edges
  [[nodiscard]] inline IotaRange<NodeID> nodes() const final {
    return _underlying_graph->nodes();
  }

  [[nodiscard]] inline IotaRange<EdgeID> edges() const final {
    return _underlying_graph->edges();
  }

  // Parallel iteration
  template <typename Lambda> inline void pfor_nodes(Lambda &&l) const {
    if (const auto *graph = dynamic_cast<const CSRGraph *>(_underlying_graph.get());
        graph != nullptr) {
      graph->pfor_nodes(std::forward<Lambda>(l));
      return;
    }

    if (const auto *graph = dynamic_cast<const CompactCSRGraph *>(_underlying_graph.get());
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

    if (const auto *graph = dynamic_cast<const CompactCSRGraph *>(_underlying_graph.get());
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

  // Graph operations
  [[nodiscard]] inline decltype(auto) incident_edges(const NodeID u) const {
    if (const auto *graph = dynamic_cast<const CSRGraph *>(_underlying_graph.get());
        graph != nullptr) {
      return graph->incident_edges(u);
    }

    if (const auto *graph = dynamic_cast<const CompactCSRGraph *>(_underlying_graph.get());
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

    throw std::runtime_error("This operation is only available for csr graphs.");
  }

  template <typename Lambda> inline void adjacent_nodes(const NodeID u, Lambda &&l) const {
    if (const auto *graph = dynamic_cast<const CSRGraph *>(_underlying_graph.get());
        graph != nullptr) {
      graph->adjacent_nodes(u, std::forward<Lambda>(l));
      return;
    }

    if (const auto *graph = dynamic_cast<const CompactCSRGraph *>(_underlying_graph.get());
        graph != nullptr) {
      graph->adjacent_nodes(u, std::forward<Lambda>(l));
      return;
    }

    if (const auto *graph = dynamic_cast<const CompressedGraph *>(_underlying_graph.get());
        graph != nullptr) {
      graph->adjacent_nodes(u, std::forward<Lambda>(l));
      return;
    }

    __builtin_unreachable();
  }

  [[nodiscard]] inline decltype(auto) neighbors(const NodeID u) const {
    if (const auto *graph = dynamic_cast<const CSRGraph *>(_underlying_graph.get());
        graph != nullptr) {
      return graph->neighbors(u);
    }

    throw std::runtime_error("This operation is only available for csr graphs.");
  }

  template <typename Lambda> inline void neighbors(const NodeID u, Lambda &&l) const {
    if (const auto *graph = dynamic_cast<const CSRGraph *>(_underlying_graph.get());
        graph != nullptr) {
      graph->neighbors(u, std::forward<Lambda>(l));
      return;
    }

    if (const auto *graph = dynamic_cast<const CompactCSRGraph *>(_underlying_graph.get());
        graph != nullptr) {
      graph->neighbors(u, std::forward<Lambda>(l));
      return;
    }

    if (const auto *graph = dynamic_cast<const CompressedGraph *>(_underlying_graph.get());
        graph != nullptr) {
      graph->neighbors(u, std::forward<Lambda>(l));
      return;
    }

    __builtin_unreachable();
  }

  template <typename Lambda>
  inline void neighbors(const NodeID u, const NodeID max_neighbor_count, Lambda &&l) const {
    if (const auto *graph = dynamic_cast<const CSRGraph *>(_underlying_graph.get());
        graph != nullptr) {
      graph->neighbors(u, max_neighbor_count, std::forward<Lambda>(l));
      return;
    }

    if (const auto *graph = dynamic_cast<const CompactCSRGraph *>(_underlying_graph.get());
        graph != nullptr) {
      graph->neighbors(u, max_neighbor_count, std::forward<Lambda>(l));
      return;
    }

    if (const auto *graph = dynamic_cast<const CompressedGraph *>(_underlying_graph.get());
        graph != nullptr) {
      graph->neighbors(u, max_neighbor_count, std::forward<Lambda>(l));
      return;
    }

    __builtin_unreachable();
  }

  template <typename Lambda>
  inline void pfor_neighbors(
      const NodeID u, const NodeID max_neighbor_count, const NodeID grainsize, Lambda &&l
  ) const {
    if (const auto *graph = dynamic_cast<const CSRGraph *>(_underlying_graph.get());
        graph != nullptr) {
      graph->pfor_neighbors(u, max_neighbor_count, grainsize, std::forward<Lambda>(l));
      return;
    }

    if (const auto *graph = dynamic_cast<const CompactCSRGraph *>(_underlying_graph.get());
        graph != nullptr) {
      graph->pfor_neighbors(u, max_neighbor_count, grainsize, std::forward<Lambda>(l));
      return;
    }

    if (const auto *graph = dynamic_cast<const CompressedGraph *>(_underlying_graph.get());
        graph != nullptr) {
      graph->pfor_neighbors(u, max_neighbor_count, grainsize, std::forward<Lambda>(l));
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

  [[nodiscard]] inline StaticArray<NodeID> &&take_raw_permutation() final {
    return _underlying_graph->take_raw_permutation();
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

namespace debug {
void print_graph(const Graph &graph);
} // namespace debug

} // namespace kaminpar::shm
