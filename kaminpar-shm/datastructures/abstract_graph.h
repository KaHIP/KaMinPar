/*******************************************************************************
 * Abstract interface for a graph data structure.
 *
 * @file:   abstract_graph.h
 * @author: Daniel Seemaier
 * @date:   17.11.2023
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/definitions.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/degree_buckets.h"
#include "kaminpar-common/ranges.h"

namespace kaminpar::shm {
class AbstractGraph {
public:
  // Data types used by this graph
  using NodeID = ::kaminpar::shm::NodeID;
  using NodeWeight = ::kaminpar::shm::NodeWeight;
  using EdgeID = ::kaminpar::shm::EdgeID;
  using EdgeWeight = ::kaminpar::shm::EdgeWeight;

  AbstractGraph() = default;

  AbstractGraph(const AbstractGraph &) = delete;
  AbstractGraph &operator=(const AbstractGraph &) = delete;

  AbstractGraph(AbstractGraph &&) noexcept = default;
  AbstractGraph &operator=(AbstractGraph &&) noexcept = default;

  virtual ~AbstractGraph() = default;

  // Size of the graph
  [[nodiscard]] virtual NodeID n() const = 0;
  [[nodiscard]] virtual EdgeID m() const = 0;

  // Node and edge weights
  [[nodiscard]] virtual bool node_weighted() const = 0;
  [[nodiscard]] virtual NodeWeight node_weight(NodeID u) const = 0;
  [[nodiscard]] virtual NodeWeight max_node_weight() const = 0;
  [[nodiscard]] virtual NodeWeight total_node_weight() const = 0;

  [[nodiscard]] virtual bool edge_weighted() const = 0;
  [[nodiscard]] virtual EdgeWeight edge_weight(EdgeID e) const = 0;
  [[nodiscard]] virtual EdgeWeight total_edge_weight() const = 0;

  // Low-level access to the graph structure
  [[nodiscard]] virtual NodeID max_degree() const = 0;
  [[nodiscard]] virtual NodeID degree(NodeID u) const = 0;

  // Iterators for nodes / edges
  [[nodiscard]] virtual IotaRange<NodeID> nodes() const = 0;
  [[nodiscard]] virtual IotaRange<EdgeID> edges() const = 0;

  // Graph permutation
  virtual void set_permutation(StaticArray<NodeID> permutation) = 0;
  [[nodiscard]] virtual bool permuted() const = 0;
  [[nodiscard]] virtual NodeID map_original_node(NodeID u) const = 0;
  [[nodiscard]] virtual StaticArray<NodeID> &&take_raw_permutation() = 0;

  // Degree buckets
  [[nodiscard]] virtual std::size_t bucket_size(std::size_t bucket) const = 0;
  [[nodiscard]] virtual NodeID first_node_in_bucket(std::size_t bucket) const = 0;
  [[nodiscard]] virtual NodeID first_invalid_node_in_bucket(std::size_t bucket) const = 0;
  [[nodiscard]] virtual std::size_t number_of_buckets() const = 0;
  [[nodiscard]] virtual bool sorted() const = 0;

  virtual void update_total_node_weight() = 0;
  virtual void update_degree_buckets() = 0;
};
} // namespace kaminpar::shm
