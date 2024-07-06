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

#include <utility>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include "kaminpar-shm/datastructures/abstract_graph.h"
#include "kaminpar-shm/datastructures/compressed_graph.h"
#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/ranges.h"

namespace kaminpar::shm {
namespace graph {
template <typename Lambda> decltype(auto) reified(const AbstractGraph *abstract_graph, Lambda &&l) {
  if (const auto *graph = dynamic_cast<const CSRGraph *>(abstract_graph); graph != nullptr) {
    return l(*graph);
  } else if (auto *graph = dynamic_cast<const CompactCSRGraph *>(abstract_graph);
             graph != nullptr) {
    return l(*graph);
  } else if (auto *graph = dynamic_cast<const CompressedGraph *>(abstract_graph);
             graph != nullptr) {
    return l(*graph);
  }

  __builtin_unreachable();
}
} // namespace graph

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

  // Access to the wrapped graph
  [[nodiscard]] const AbstractGraph *underlying_graph() const {
    return _underlying_graph.get();
  }

  [[nodiscard]] AbstractGraph *underlying_graph() {
    return _underlying_graph.get();
  }

  [[nodiscard]] CSRGraph &csr_graph() {
    AbstractGraph *abstract_graph = _underlying_graph.get();
    return *dynamic_cast<CSRGraph *>(abstract_graph);
  }

  template <typename Lambda> decltype(auto) reified(Lambda &&l) const {
    return graph::reified(underlying_graph(), std::forward<Lambda>(l));
  }

  // Size of the graph
  [[nodiscard]] inline NodeID n() const final {
    return _underlying_graph->n();
  }

  [[nodiscard]] inline EdgeID m() const final {
    return _underlying_graph->m();
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
    reified([&](auto &graph) { graph.pfor_nodes(std::forward<Lambda>(l)); });
  }

  template <typename Lambda> inline void pfor_edges(Lambda &&l) const {
    reified([&](auto &graph) { graph.pfor_edges(std::forward<Lambda>(l)); });
  }

  // Graph operations
  [[nodiscard]] inline decltype(auto) incident_edges(const NodeID u) const {
    return reified([&](auto &graph) { return graph.incident_edges(u); });
  }

  [[nodiscard]] inline decltype(auto) adjacent_nodes(const NodeID u) const {
    if (const auto *graph = dynamic_cast<const CSRGraph *>(_underlying_graph.get());
        graph != nullptr) {
      return graph->adjacent_nodes(u);
    }

    throw std::runtime_error("This operation is only available for csr graphs.");
  }

  template <typename Lambda> inline void adjacent_nodes(const NodeID u, Lambda &&l) const {
    reified([&](auto &graph) { graph.adjacent_nodes(u, std::forward<Lambda>(l)); });
  }

  [[nodiscard]] inline decltype(auto) neighbors(const NodeID u) const {
    if (const auto *graph = dynamic_cast<const CSRGraph *>(_underlying_graph.get());
        graph != nullptr) {
      return graph->neighbors(u);
    }

    throw std::runtime_error("This operation is only available for csr graphs.");
  }

  template <typename Lambda> inline void neighbors(const NodeID u, Lambda &&l) const {
    reified([&](auto &graph) { graph.neighbors(u, std::forward<Lambda>(l)); });
  }

  template <typename Lambda>
  inline void neighbors(const NodeID u, const NodeID max_neighbor_count, Lambda &&l) const {
    reified([&](auto &graph) { graph.neighbors(u, max_neighbor_count, std::forward<Lambda>(l)); });
  }

  template <typename Lambda>
  inline void pfor_neighbors(
      const NodeID u, const NodeID max_neighbor_count, const NodeID grainsize, Lambda &&l
  ) const {
    reified([&](auto &graph) {
      graph.pfor_neighbors(u, max_neighbor_count, grainsize, std::forward<Lambda>(l));
    });
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

private:
  std::unique_ptr<AbstractGraph> _underlying_graph;
};

namespace debug {
void print_graph(const Graph &graph);
} // namespace debug

} // namespace kaminpar::shm
