/*******************************************************************************
 * Abstract interface for a graph data structure.
 *
 * @file:   abstract_distributed_graph.h
 * @author: Daniel Salwasser
 * @date:   06.06.2024
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/dkaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/ranges.h"

namespace kaminpar::dist {

class AbstractDistributedGraph {
public:
  // Data types used for this graph
  using NodeID = dist::NodeID;
  using EdgeID = dist::EdgeID;
  using GlobalNodeID = dist::GlobalNodeID;
  using GlobalEdgeID = dist::GlobalEdgeID;
  using NodeWeight = dist::NodeWeight;
  using EdgeWeight = dist::EdgeWeight;
  using GlobalNodeWeight = dist::GlobalNodeWeight;
  using GlobalEdgeWeight = dist::GlobalEdgeWeight;

  AbstractDistributedGraph() = default;

  AbstractDistributedGraph(const AbstractDistributedGraph &) = delete;
  AbstractDistributedGraph &operator=(const AbstractDistributedGraph &) = delete;

  AbstractDistributedGraph(AbstractDistributedGraph &&) noexcept = default;
  AbstractDistributedGraph &operator=(AbstractDistributedGraph &&) noexcept = default;

  virtual ~AbstractDistributedGraph() = default;

  // Size of the graph
  [[nodiscard]] virtual GlobalNodeID global_n() const = 0;
  [[nodiscard]] virtual GlobalEdgeID global_m() const = 0;

  [[nodiscard]] virtual NodeID n() const = 0;
  [[nodiscard]] virtual NodeID n(const PEID pe) const = 0;
  [[nodiscard]] virtual NodeID ghost_n() const = 0;
  [[nodiscard]] virtual NodeID total_n() const = 0;

  [[nodiscard]] virtual EdgeID m() const = 0;
  [[nodiscard]] virtual EdgeID m(const PEID pe) const = 0;

  [[nodiscard]] virtual GlobalNodeID offset_n() const = 0;
  [[nodiscard]] virtual GlobalNodeID offset_n(const PEID pe) const = 0;

  [[nodiscard]] virtual GlobalEdgeID offset_m() const = 0;
  [[nodiscard]] virtual GlobalEdgeID offset_m(const PEID pe) const = 0;

  // Node and edge weights
  [[nodiscard]] virtual bool is_node_weighted() const = 0;
  [[nodiscard]] virtual NodeWeight node_weight(const NodeID u) const = 0;
  [[nodiscard]] virtual NodeWeight max_node_weight() const = 0;
  [[nodiscard]] virtual NodeWeight global_max_node_weight() const = 0;
  [[nodiscard]] virtual NodeWeight total_node_weight() const = 0;
  [[nodiscard]] virtual GlobalNodeWeight global_total_node_weight() const = 0;

  [[nodiscard]] virtual bool is_edge_weighted() const = 0;
  [[nodiscard]] virtual EdgeWeight edge_weight(const EdgeID e) const = 0;
  [[nodiscard]] virtual EdgeWeight total_edge_weight() const = 0;
  [[nodiscard]] virtual GlobalEdgeWeight global_total_edge_weight() const = 0;

  // Node ownership
  [[nodiscard]] virtual bool is_owned_global_node(const GlobalNodeID global_u) const = 0;
  [[nodiscard]] virtual bool contains_global_node(const GlobalNodeID global_u) const = 0;
  [[nodiscard]] virtual bool contains_local_node(const NodeID local_u) const = 0;

  // Node type
  [[nodiscard]] virtual bool is_ghost_node(const NodeID u) const = 0;
  [[nodiscard]] virtual bool is_owned_node(const NodeID u) const = 0;
  [[nodiscard]] virtual PEID ghost_owner(const NodeID u) const = 0;
  [[nodiscard]] virtual NodeID
  map_remote_node(const NodeID their_lnode, const PEID owner) const = 0;
  [[nodiscard]] virtual GlobalNodeID local_to_global_node(const NodeID local_u) const = 0;
  [[nodiscard]] virtual NodeID global_to_local_node(const GlobalNodeID global_u) const = 0;

  // Iterators for nodes / edges
  [[nodiscard]] virtual IotaRange<NodeID> nodes(const NodeID from, const NodeID to) const = 0;
  [[nodiscard]] virtual IotaRange<NodeID> nodes() const = 0;
  [[nodiscard]] virtual IotaRange<NodeID> ghost_nodes() const = 0;
  [[nodiscard]] virtual IotaRange<NodeID> all_nodes() const = 0;

  [[nodiscard]] virtual IotaRange<EdgeID> edges() const = 0;
  [[nodiscard]] virtual IotaRange<EdgeID> incident_edges(const NodeID u) const = 0;

  // Access methods
  [[nodiscard]] virtual NodeID degree(const NodeID u) const = 0;

  [[nodiscard]] virtual const StaticArray<NodeWeight> &node_weights() const = 0;
  [[nodiscard]] virtual const StaticArray<EdgeWeight> &edge_weights() const = 0;

  virtual void set_ghost_node_weight(const NodeID ghost_node, const NodeWeight weight) = 0;

  [[nodiscard]] virtual const StaticArray<GlobalNodeID> &node_distribution() const = 0;
  [[nodiscard]] virtual GlobalNodeID node_distribution(const PEID pe) const = 0;
  [[nodiscard]] virtual PEID find_owner_of_global_node(const GlobalNodeID u) const = 0;

  [[nodiscard]] virtual const StaticArray<GlobalEdgeID> &edge_distribution() const = 0;
  [[nodiscard]] virtual GlobalEdgeID edge_distribution(const PEID pe) const = 0;

  // Cached inter-PE metrics
  [[nodiscard]] virtual EdgeID edge_cut_to_pe(const PEID pe) const = 0;
  [[nodiscard]] virtual EdgeID comm_vol_to_pe(const PEID pe) const = 0;
  [[nodiscard]] virtual MPI_Comm communicator() const = 0;

  // High degree classification
  virtual void init_high_degree_info(const EdgeID high_degree_threshold) const = 0;
  [[nodiscard]] virtual bool is_high_degree_node(const NodeID node) const = 0;

  // Graph permutation
  virtual void set_permutation(StaticArray<NodeID> permutation) = 0;
  [[nodiscard]] virtual bool permuted() const = 0;
  [[nodiscard]] virtual NodeID map_original_node(const NodeID u) const = 0;

  // Degree buckets
  [[nodiscard]] virtual bool sorted() const = 0;
  [[nodiscard]] virtual std::size_t number_of_buckets() const = 0;
  [[nodiscard]] virtual std::size_t bucket_size(const std::size_t bucket) const = 0;
  [[nodiscard]] virtual NodeID first_node_in_bucket(const std::size_t bucket) const = 0;
  [[nodiscard]] virtual NodeID first_invalid_node_in_bucket(const std::size_t bucket) const = 0;

  // Graph permutation by coloring
  virtual void set_color_sorted(StaticArray<NodeID> color_sizes) = 0;
  [[nodiscard]] virtual bool color_sorted() const = 0;
  [[nodiscard]] virtual std::size_t number_of_colors() const = 0;
  [[nodiscard]] virtual NodeID color_size(const std::size_t c) const = 0;
  [[nodiscard]] virtual const StaticArray<NodeID> &get_color_sizes() const = 0;
};

} // namespace kaminpar::dist
