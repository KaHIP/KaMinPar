/*******************************************************************************
 * Wrapper class that delegates all function calls to a concrete graph object.
 *
 * Most function calls are resolved via dynamic binding. Thus, they should not
 * be used when performance is critical. Instead, use an downcast and templatize
 * tight loops.
 *
 * @file:   distributed_graph.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 ******************************************************************************/
#pragma once

#include <memory>

#include "kaminpar-dist/datastructures/abstract_distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_compressed_graph.h"
#include "kaminpar-dist/datastructures/distributed_csr_graph.h"
#include "kaminpar-dist/dkaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/ranges.h"

namespace kaminpar::dist {

class DistributedGraph : public AbstractDistributedGraph {
public:
  // Data types used for this graph
  using AbstractDistributedGraph::EdgeID;
  using AbstractDistributedGraph::EdgeWeight;
  using AbstractDistributedGraph::GlobalEdgeID;
  using AbstractDistributedGraph::GlobalEdgeWeight;
  using AbstractDistributedGraph::GlobalNodeID;
  using AbstractDistributedGraph::GlobalNodeWeight;
  using AbstractDistributedGraph::NodeID;
  using AbstractDistributedGraph::NodeWeight;

  DistributedGraph() = default;

  DistributedGraph(std::unique_ptr<AbstractDistributedGraph> graph)
      : _underlying_graph(std::move(graph)) {}

  DistributedGraph(const DistributedGraph &) = delete;
  DistributedGraph &operator=(const DistributedGraph &) = delete;

  DistributedGraph(DistributedGraph &&) noexcept = default;
  DistributedGraph &operator=(DistributedGraph &&) noexcept = default;

  ~DistributedGraph() override = default;

  //
  // Size of the graph
  //

  [[nodiscard]] inline GlobalNodeID global_n() const final {
    return _underlying_graph->global_n();
  }

  [[nodiscard]] inline GlobalEdgeID global_m() const final {
    return _underlying_graph->global_m();
  }

  [[nodiscard]] inline NodeID n() const final {
    return _underlying_graph->n();
  }

  [[nodiscard]] inline NodeID n(const PEID pe) const final {
    return _underlying_graph->n(pe);
  }

  [[nodiscard]] inline NodeID ghost_n() const final {
    return _underlying_graph->ghost_n();
  }

  [[nodiscard]] inline NodeID total_n() const final {
    return _underlying_graph->total_n();
  }

  [[nodiscard]] inline EdgeID m() const final {
    return _underlying_graph->m();
  }

  [[nodiscard]] inline EdgeID m(const PEID pe) const final {
    return _underlying_graph->m(pe);
  }

  [[nodiscard]] inline GlobalNodeID offset_n() const final {
    return _underlying_graph->offset_n();
  }

  [[nodiscard]] inline GlobalNodeID offset_n(const PEID pe) const final {
    return _underlying_graph->offset_n(pe);
  }

  [[nodiscard]] inline GlobalEdgeID offset_m() const final {
    return _underlying_graph->offset_m();
  }

  [[nodiscard]] inline GlobalEdgeID offset_m(const PEID pe) const final {
    return _underlying_graph->offset_m(pe);
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

  [[nodiscard]] inline NodeWeight global_max_node_weight() const final {
    return _underlying_graph->global_max_node_weight();
  }

  [[nodiscard]] inline NodeWeight total_node_weight() const final {
    return _underlying_graph->total_node_weight();
  }

  [[nodiscard]] inline GlobalNodeWeight global_total_node_weight() const final {
    return _underlying_graph->global_total_node_weight();
  }

  [[nodiscard]] inline bool is_edge_weighted() const final {
    return _underlying_graph->is_edge_weighted();
  }

  [[nodiscard]] inline EdgeWeight total_edge_weight() const final {
    return _underlying_graph->total_edge_weight();
  }

  [[nodiscard]] inline GlobalEdgeWeight global_total_edge_weight() const final {
    return _underlying_graph->global_total_edge_weight();
  }

  //
  // Node ownership
  //

  [[nodiscard]] inline bool is_owned_global_node(const GlobalNodeID global_u) const final {
    return _underlying_graph->is_owned_global_node(global_u);
  }

  [[nodiscard]] inline bool contains_global_node(const GlobalNodeID global_u) const final {
    return _underlying_graph->contains_global_node(global_u);
  }

  [[nodiscard]] inline bool contains_local_node(const NodeID local_u) const final {
    return _underlying_graph->contains_local_node(local_u);
  }

  //
  // Node type
  //

  [[nodiscard]] inline bool is_ghost_node(const NodeID u) const final {
    return _underlying_graph->is_ghost_node(u);
  }

  [[nodiscard]] inline bool is_owned_node(const NodeID u) const final {
    return _underlying_graph->is_owned_node(u);
  }

  [[nodiscard]] inline PEID ghost_owner(const NodeID u) const final {
    return _underlying_graph->ghost_owner(u);
  }

  [[nodiscard]] inline NodeID
  map_remote_node(const NodeID their_lnode, const PEID owner) const final {
    return _underlying_graph->map_remote_node(their_lnode, owner);
  }

  [[nodiscard]] inline GlobalNodeID local_to_global_node(const NodeID local_u) const final {
    return _underlying_graph->local_to_global_node(local_u);
  }

  [[nodiscard]] inline NodeID global_to_local_node(const GlobalNodeID global_u) const final {
    return _underlying_graph->global_to_local_node(global_u);
  }

  //
  // Iterators for nodes / edges
  //

  [[nodiscard]] inline IotaRange<NodeID> nodes(const NodeID from, const NodeID to) const final {
    return _underlying_graph->nodes(from, to);
  }

  [[nodiscard]] inline IotaRange<NodeID> nodes() const final {
    return _underlying_graph->nodes();
  }

  [[nodiscard]] inline IotaRange<NodeID> ghost_nodes() const final {
    return _underlying_graph->ghost_nodes();
  }

  [[nodiscard]] inline IotaRange<NodeID> all_nodes() const final {
    return _underlying_graph->all_nodes();
  }

  [[nodiscard]] inline IotaRange<EdgeID> edges() const final {
    return _underlying_graph->edges();
  }

  //
  // Access methods
  //

  [[nodiscard]] inline NodeID degree(const NodeID u) const final {
    return _underlying_graph->degree(u);
  }

  [[nodiscard]] inline const StaticArray<NodeWeight> &node_weights() const final {
    return _underlying_graph->node_weights();
  }

  inline void set_ghost_node_weight(const NodeID ghost_node, const NodeWeight weight) final {
    _underlying_graph->set_ghost_node_weight(ghost_node, weight);
  }

  [[nodiscard]] inline const StaticArray<GlobalNodeID> &node_distribution() const final {
    return _underlying_graph->node_distribution();
  }

  [[nodiscard]] inline GlobalNodeID node_distribution(const PEID pe) const final {
    return _underlying_graph->node_distribution(pe);
  }

  [[nodiscard]] inline PEID find_owner_of_global_node(const GlobalNodeID u) const final {
    return _underlying_graph->find_owner_of_global_node(u);
  }

  [[nodiscard]] inline const StaticArray<GlobalEdgeID> &edge_distribution() const final {
    return _underlying_graph->edge_distribution();
  }

  [[nodiscard]] inline GlobalEdgeID edge_distribution(const PEID pe) const final {
    return _underlying_graph->edge_distribution(pe);
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

  template <typename Lambda>
  inline void pfor_nodes(const NodeID from, const NodeID to, Lambda &&l) const {
    reified([&](auto &graph) { graph.pfor_nodes(from, to, std::forward<Lambda>(l)); });
  }

  template <typename Lambda>
  inline void pfor_nodes_range(const NodeID from, const NodeID to, Lambda &&l) const {
    reified([&](auto &graph) { graph.pfor_nodes_range(from, to, std::forward<Lambda>(l)); });
  }

  template <typename Lambda> inline void pfor_ghost_nodes(Lambda &&l) const {
    reified([&](auto &graph) { graph.pfor_ghost_nodes(std::forward<Lambda>(l)); });
  }

  template <typename Lambda> inline void pfor_nodes(Lambda &&l) const {
    reified([&](auto &graph) { graph.pfor_nodes(std::forward<Lambda>(l)); });
  }

  template <typename Lambda> inline void pfor_all_nodes(Lambda &&l) const {
    reified([&](auto &graph) { graph.pfor_all_nodes(std::forward<Lambda>(l)); });
  }

  template <typename Lambda> inline void pfor_nodes_range(Lambda &&l) const {
    reified([&](auto &graph) { graph.pfor_nodes_range(std::forward<Lambda>(l)); });
  }

  template <typename Lambda> inline void pfor_all_nodes_range(Lambda &&l) const {
    reified([&](auto &graph) { graph.pfor_all_nodes_range(std::forward<Lambda>(l)); });
  }

  template <typename Lambda> inline void pfor_edges(Lambda &&l) const {
    reified([&](auto &graph) { graph.pfor_edges(std::forward<Lambda>(l)); });
  }

  //
  // Cached inter-PE metrics
  //

  [[nodiscard]] inline EdgeID edge_cut_to_pe(const PEID pe) const final {
    return _underlying_graph->edge_cut_to_pe(pe);
  }

  [[nodiscard]] inline EdgeID comm_vol_to_pe(const PEID pe) const final {
    return _underlying_graph->comm_vol_to_pe(pe);
  }

  [[nodiscard]] inline MPI_Comm communicator() const final {
    return _underlying_graph->communicator();
  }

  //
  // High degree classification
  //

  inline void init_high_degree_info(const EdgeID high_degree_threshold) const final {
    _underlying_graph->init_high_degree_info(high_degree_threshold);
  }

  [[nodiscard]] inline bool is_high_degree_node(const NodeID node) const final {
    return _underlying_graph->is_high_degree_node(node);
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

  //
  // Graph permutation by coloring
  //

  inline void set_color_sorted(StaticArray<NodeID> color_sizes) final {
    _underlying_graph->set_color_sorted(std::move(color_sizes));
  }

  [[nodiscard]] inline bool color_sorted() const final {
    return _underlying_graph->color_sorted();
  }

  [[nodiscard]] inline std::size_t number_of_colors() const final {
    return _underlying_graph->number_of_colors();
  }

  [[nodiscard]] inline NodeID color_size(const std::size_t c) const final {
    return _underlying_graph->color_size(c);
  }

  [[nodiscard]] inline const StaticArray<NodeID> &get_color_sizes() const final {
    return _underlying_graph->get_color_sizes();
  }

  //
  // Access to underlying graph
  //

  [[nodiscard]] inline AbstractDistributedGraph *underlying_graph() {
    return _underlying_graph.get();
  }

  [[nodiscard]] inline const AbstractDistributedGraph *underlying_graph() const {
    return _underlying_graph.get();
  }

  [[nodiscard]] inline AbstractDistributedGraph *take_underlying_graph() {
    return _underlying_graph.release();
  }

  [[nodiscard]] inline DistributedCSRGraph &csr_graph() {
    AbstractDistributedGraph *abstract_graph = _underlying_graph.get();
    return *dynamic_cast<DistributedCSRGraph *>(abstract_graph);
  }

  [[nodiscard]] inline const DistributedCSRGraph &csr_graph() const {
    const AbstractDistributedGraph *abstract_graph = _underlying_graph.get();
    return *dynamic_cast<const DistributedCSRGraph *>(abstract_graph);
  }

  [[nodiscard]] inline DistributedCompressedGraph &compressed_graph() {
    AbstractDistributedGraph *abstract_graph = _underlying_graph.get();
    return *dynamic_cast<DistributedCompressedGraph *>(abstract_graph);
  }

  [[nodiscard]] inline const DistributedCompressedGraph &compressed_graph() const {
    const AbstractDistributedGraph *abstract_graph = _underlying_graph.get();
    return *dynamic_cast<const DistributedCompressedGraph *>(abstract_graph);
  }

  template <typename Lambda1, typename Lambda2>
  inline decltype(auto) reified(Lambda1 &&l1, Lambda2 &&l2) const {
    const AbstractDistributedGraph *abstract_graph = _underlying_graph.get();

    if (const auto *graph = dynamic_cast<const DistributedCSRGraph *>(abstract_graph);
        graph != nullptr) {
      return l1(*graph);
    } else if (const auto *graph = dynamic_cast<const DistributedCompressedGraph *>(abstract_graph);
               graph != nullptr) {
      return l2(*graph);
    }

    __builtin_unreachable();
  }

  template <typename Lambda> inline decltype(auto) reified(Lambda &&l) const {
    return reified(std::forward<Lambda>(l), std::forward<Lambda>(l));
  }

private:
  std::unique_ptr<AbstractDistributedGraph> _underlying_graph;
};

/**
 * Prints verbose statistics on the distribution of the graph across PEs and the
 * number of ghost nodes, but only if verbose statistics are enabled as build
 * option.
 * @param graph Graph for which statistics are printed.
 */
void print_graph_summary(const DistributedGraph &graph);

namespace debug {

void print_graph(const DistributedGraph &graph);

void print_local_graph_stats(const DistributedGraph &graph);

/**
 * Validates the distributed graph datastructure:
 * - validate node and edge distributions
 * - check that undirected edges have the same weight in both directions
 * - check that the graph is actually undirected
 * - check the weight of interface and ghost nodes
 *
 * @param graph the graph to check.
 * @param root PE to use for sequential validation.
 * @return whether the graph data structure is consistent.
 */
bool validate_graph(const DistributedGraph &graph);

} // namespace debug

} // namespace kaminpar::dist
