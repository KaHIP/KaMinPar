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

#include <memory>
#include <utility>

#include "kaminpar-shm/datastructures/abstract_graph.h"
#include "kaminpar-shm/datastructures/compressed_graph.h"
#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"
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

  //
  // Size of the graph
  //

  [[nodiscard]] inline NodeID n() const final {
    return _underlying_graph->n();
  }

  [[nodiscard]] inline EdgeID m() const final {
    return _underlying_graph->m();
  }

  //
  // Node and edge weights
  //

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

  inline void update_total_node_weight() final {
    _underlying_graph->update_total_node_weight();
  }

  [[nodiscard]] inline bool is_edge_weighted() const final {
    return _underlying_graph->is_edge_weighted();
  }

  [[nodiscard]] inline EdgeWeight total_edge_weight() const final {
    return _underlying_graph->total_edge_weight();
  }

  //
  // Iterators for nodes / edges
  //

  [[nodiscard]] inline IotaRange<NodeID> nodes() const final {
    return _underlying_graph->nodes();
  }

  [[nodiscard]] inline IotaRange<EdgeID> edges() const final {
    return _underlying_graph->edges();
  }

  //
  // Node degree
  //

  [[nodiscard]] inline NodeID max_degree() const final {
    return _underlying_graph->max_degree();
  }

  [[nodiscard]] inline NodeID degree(const NodeID u) const final {
    return _underlying_graph->degree(u);
  }

  //
  // Graph operations
  //

  template <typename Lambda> inline void adjacent_nodes(const NodeID u, Lambda &&l) const {
    reified([&](auto &graph) { graph.adjacent_nodes(u, std::forward<Lambda>(l)); });
  }

  template <typename Lambda>
  inline void adjacent_nodes(const NodeID u, const NodeID max_num_neighbors, Lambda &&l) const {
    reified([&](auto &graph) {
      graph.adjacent_nodes(u, max_num_neighbors, std::forward<Lambda>(l));
    });
  }

  //
  // Parallel iteration
  //

  template <typename Lambda> inline void pfor_nodes_range(Lambda &&l) const {
    reified([&](auto &graph) { graph.pfor_nodes_range(std::forward<Lambda>(l)); });
  }

  template <typename Lambda> inline void pfor_nodes(Lambda &&l) const {
    reified([&](auto &graph) { graph.pfor_nodes(std::forward<Lambda>(l)); });
  }

  template <typename Lambda> inline void pfor_edges(Lambda &&l) const {
    reified([&](auto &graph) { graph.pfor_edges(std::forward<Lambda>(l)); });
  }

  template <typename Lambda>
  inline void pfor_adjacent_nodes(
      const NodeID u, const NodeID max_num_neighbors, const NodeID grainsize, Lambda &&l
  ) const {
    reified([&](const auto &graph) {
      graph.pfor_adjacent_nodes(u, max_num_neighbors, grainsize, std::forward<Lambda>(l));
    });
  }

  //
  // Graph permutation
  //

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

  //
  // Degree buckets
  //

  [[nodiscard]] inline bool sorted() const final {
    return _underlying_graph->sorted();
  }

  [[nodiscard]] inline std::size_t number_of_buckets() const final {
    return _underlying_graph->number_of_buckets();
  }

  [[nodiscard]] inline std::size_t bucket_size(const std::size_t bucket) const final {
    return _underlying_graph->bucket_size(bucket);
  }

  [[nodiscard]] inline NodeID first_node_in_bucket(const std::size_t bucket) const final {
    return _underlying_graph->first_node_in_bucket(bucket);
  }

  [[nodiscard]] inline NodeID first_invalid_node_in_bucket(const std::size_t bucket) const final {
    return _underlying_graph->first_invalid_node_in_bucket(bucket);
  }

  void remove_isolated_nodes(const NodeID num_isolated_nodes) final {
    _underlying_graph->remove_isolated_nodes(num_isolated_nodes);
  }

  NodeID integrate_isolated_nodes() final {
    return _underlying_graph->integrate_isolated_nodes();
  }

  //
  // Access to the underlying graph
  //

  [[nodiscard]] AbstractGraph *underlying_graph() {
    return _underlying_graph.get();
  }

  [[nodiscard]] const AbstractGraph *underlying_graph() const {
    return _underlying_graph.get();
  }

  template <typename ConcretizedGraph> [[nodiscard]] ConcretizedGraph &concretize() {
    KASSERT(
        dynamic_cast<ConcretizedGraph *>(underlying_graph()) != nullptr,
        "underlying graph is not a " << typeid(ConcretizedGraph).name()
    );

    return *static_cast<ConcretizedGraph *>(underlying_graph());
  }

  template <typename ConcretizedGraph> [[nodiscard]] const ConcretizedGraph &concretize() const {
    KASSERT(
        dynamic_cast<const ConcretizedGraph *>(underlying_graph()) != nullptr,
        "underlying graph is not a " << typeid(ConcretizedGraph).name()
    );

    return *static_cast<const ConcretizedGraph *>(underlying_graph());
  }

  [[nodiscard]] CSRGraph &csr_graph() {
    AbstractGraph *abstract_graph = _underlying_graph.get();
    return *dynamic_cast<CSRGraph *>(abstract_graph);
  }

  [[nodiscard]] const CSRGraph &csr_graph() const {
    const AbstractGraph *abstract_graph = _underlying_graph.get();
    return *dynamic_cast<const CSRGraph *>(abstract_graph);
  }

  [[nodiscard]] CompressedGraph &compressed_graph() {
    AbstractGraph *abstract_graph = _underlying_graph.get();
    return *dynamic_cast<CompressedGraph *>(abstract_graph);
  }

  [[nodiscard]] const CompressedGraph &compressed_graph() const {
    const AbstractGraph *abstract_graph = _underlying_graph.get();
    return *dynamic_cast<const CompressedGraph *>(abstract_graph);
  }

  template <typename Lambda1, typename Lambda2> decltype(auto) reified(Lambda1 &&l1, Lambda2 &&l2) {
    AbstractGraph *abstract_graph = _underlying_graph.get();

    if (auto *csr_graph = dynamic_cast<CSRGraph *>(abstract_graph); csr_graph != nullptr) {
      return l1(*csr_graph);
    }

    if (auto *compressed_graph = dynamic_cast<CompressedGraph *>(abstract_graph);
        compressed_graph != nullptr) {
      return l2(*compressed_graph);
    }

    __builtin_unreachable();
  }

  template <typename Lambda1, typename Lambda2>
  decltype(auto) reified(Lambda1 &&l1, Lambda2 &&l2) const {
    AbstractGraph *abstract_graph = _underlying_graph.get();

    if (const auto *csr_graph = dynamic_cast<const CSRGraph *>(abstract_graph);
        csr_graph != nullptr) {
      return l1(*csr_graph);
    }

    if (const auto *compressed_graph = dynamic_cast<const CompressedGraph *>(abstract_graph);
        compressed_graph != nullptr) {
      return l2(*compressed_graph);
    }

    __builtin_unreachable();
  }

  template <typename Lambda> decltype(auto) reified(Lambda &&l) {
    return reified(std::forward<Lambda>(l), std::forward<Lambda>(l));
  }

  template <typename Lambda> decltype(auto) reified(Lambda &&l) const {
    return reified(std::forward<Lambda>(l), std::forward<Lambda>(l));
  }

private:
  std::unique_ptr<AbstractGraph> _underlying_graph;
};

namespace debug {

void print_graph(const Graph &graph);

} // namespace debug

} // namespace kaminpar::shm
