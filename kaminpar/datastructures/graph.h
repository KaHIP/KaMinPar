/*******************************************************************************
 * @file:   graph.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Static graph data structure with dynamic partition wrapper.
 ******************************************************************************/
#pragma once

#include <numeric>
#include <utility>
#include <vector>

#include <kassert/kassert.hpp>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include "kaminpar/definitions.h"

#include "common/datastructures/static_array.h"
#include "common/degree_buckets.h"
#include "common/ranges.h"
#include "common/tags.h"

namespace kaminpar::shm {
using NodeArray = StaticArray<NodeID>;
using EdgeArray = StaticArray<EdgeID>;
using NodeWeightArray = StaticArray<NodeWeight>;
using EdgeWeightArray = StaticArray<EdgeWeight>;

class Graph;
class PartitionedGraph;

class Graph {
public:
  // Data types used by this graph
  using NodeID = ::kaminpar::shm::NodeID;
  using NodeWeight = ::kaminpar::shm::NodeWeight;
  using EdgeID = ::kaminpar::shm::EdgeID;
  using EdgeWeight = ::kaminpar::shm::EdgeWeight;

  Graph() = default;

  Graph(
      StaticArray<EdgeID> nodes,
      StaticArray<NodeID> edges,
      StaticArray<NodeWeight> node_weights = {},
      StaticArray<EdgeWeight> edge_weights = {},
      bool sorted = false
  );

  Graph(
      tag::Sequential,
      StaticArray<EdgeID> nodes,
      StaticArray<NodeID> edges,
      StaticArray<NodeWeight> node_weights = {},
      StaticArray<EdgeWeight> edge_weights = {},
      bool sorted = false
  );

  Graph(const Graph &) = delete;
  Graph &operator=(const Graph &) = delete;

  Graph(Graph &&) noexcept = default;
  Graph &operator=(Graph &&) noexcept = default;

  // clang-format off
  [[nodiscard]] inline auto &raw_nodes() { return _nodes; }
  [[nodiscard]] inline const auto &raw_nodes() const { return _nodes; }
  [[nodiscard]] inline const auto &raw_edges() const { return _edges; }
  [[nodiscard]] inline auto &raw_node_weights() { return _node_weights; }
  [[nodiscard]] inline const auto &raw_node_weights() const { return _node_weights; }
  [[nodiscard]] inline const auto &raw_edge_weights() const { return _edge_weights; }
  [[nodiscard]] inline auto &&take_raw_nodes() { return std::move(_nodes); }
  [[nodiscard]] inline auto &&take_raw_edges() { return std::move(_edges); }
  [[nodiscard]] inline auto &&take_raw_node_weights() { return std::move(_node_weights); }
  [[nodiscard]] inline auto &&take_raw_edge_weights() { return std::move(_edge_weights); }

  // Edge and node weights
  [[nodiscard]] inline NodeWeight total_node_weight() const { return _total_node_weight; }
  [[nodiscard]] inline EdgeWeight total_edge_weight() const { return _total_edge_weight; }
  [[nodiscard]] inline const StaticArray<NodeWeight> &node_weights() const { return _node_weights; }
  [[nodiscard]] inline bool is_node_weighted() const { return static_cast<NodeWeight>(n()) != total_node_weight(); }
  [[nodiscard]] inline bool is_edge_weighted() const { return static_cast<EdgeWeight>(m()) != total_edge_weight(); }
  [[nodiscard]] inline NodeID n() const { return static_cast<NodeID>(_nodes.size() - 1); }
  [[nodiscard]] inline NodeID last_node() const { return n() - 1; }
  [[nodiscard]] inline EdgeID m() const { return static_cast<EdgeID>(_edges.size()); }
  [[nodiscard]] inline NodeWeight node_weight(const NodeID u) const { return is_node_weighted() ? _node_weights[u] : 1; }
  [[nodiscard]] inline EdgeWeight edge_weight(const EdgeID e) const { return is_edge_weighted() ? _edge_weights[e] : 1; }
  [[nodiscard]] inline NodeWeight max_node_weight() const { return _max_node_weight; }

  // Graph structure
  [[nodiscard]] inline NodeID edge_target(const EdgeID e) const { KASSERT(e < _edges.size()); return _edges[e]; }
  [[nodiscard]] inline NodeID degree(const NodeID u) const { return static_cast<NodeID>(_nodes[u + 1] - _nodes[u]); }
  [[nodiscard]] inline EdgeID first_edge(const NodeID u) const { return _nodes[u]; }
  [[nodiscard]] inline EdgeID first_invalid_edge(const NodeID u) const { return _nodes[u + 1]; }

  // Parallel iteration
  template<typename Lambda>
  inline void pfor_nodes(Lambda &&l) const {
    tbb::parallel_for(static_cast<NodeID>(0), n(), std::forward<Lambda>(l));
  }

  template<typename Lambda>
  inline void pfor_edges(Lambda &&l) const {
    tbb::parallel_for(static_cast<EdgeID>(0), m(), std::forward<Lambda>(l));
  }

  // Iterators for nodes / edges
  [[nodiscard]] inline IotaRange<NodeID> nodes() const { return IotaRange(static_cast<NodeID>(0), n()); }
  [[nodiscard]] inline IotaRange<EdgeID> edges() const { return IotaRange(static_cast<EdgeID>(0), m()); }
  [[nodiscard]] inline IotaRange<EdgeID> incident_edges(const NodeID u) const { return IotaRange(_nodes[u], _nodes[u + 1]); }
  [[nodiscard]] inline auto adjacent_nodes(const NodeID u) const { return TransformedIotaRange(_nodes[u], _nodes[u + 1], [this](const EdgeID e) { return this->edge_target(e); }); }
  [[nodiscard]] inline auto neighbors(const NodeID u) const { return TransformedIotaRange(_nodes[u], _nodes[u + 1], [this](const EdgeID e) { return std::make_pair(e, this->edge_target(e)); }); }

  // Graph permutation
  inline void set_permutation(StaticArray<NodeID> permutation) { _permutation = std::move(permutation); }
  [[nodiscard]] inline bool permuted() const { return !_permutation.empty(); }
  [[nodiscard]] inline NodeID map_original_node(const NodeID u) const { return _permutation[u]; }

  // Degree buckets
  [[nodiscard]] inline std::size_t bucket_size(const std::size_t bucket) const { return _buckets[bucket + 1] - _buckets[bucket]; }
  [[nodiscard]] inline NodeID first_node_in_bucket(const std::size_t bucket) const { return _buckets[bucket]; }
  [[nodiscard]] inline NodeID first_invalid_node_in_bucket(const std::size_t bucket) const { return first_node_in_bucket(bucket + 1); }
  [[nodiscard]] inline std::size_t number_of_buckets() const { return _number_of_buckets; }
  [[nodiscard]] inline bool sorted() const { return _sorted; }
  // clang-format on

  void update_total_node_weight();

private:
  void init_degree_buckets();

  StaticArray<EdgeID> _nodes;
  StaticArray<NodeID> _edges;
  StaticArray<NodeWeight> _node_weights;
  StaticArray<EdgeWeight> _edge_weights;

  NodeWeight _total_node_weight = kInvalidNodeWeight;
  EdgeWeight _total_edge_weight = kInvalidEdgeWeight;
  NodeWeight _max_node_weight = kInvalidNodeWeight;

  StaticArray<NodeID> _permutation;
  bool _sorted;
  std::vector<NodeID> _buckets = std::vector<NodeID>(kNumberOfDegreeBuckets<NodeID> + 1);
  std::size_t _number_of_buckets = 0;
};

bool validate_graph(const Graph &graph, bool check_undirected = true, NodeID num_pseudo_nodes = 0);

void print_graph(const Graph &graph);

class GraphDelegate {
public:
  GraphDelegate(const Graph *graph) : _graph(graph) {}

  // clang-format off
  [[nodiscard]] inline bool initialized() const { return _graph != nullptr; }
  [[nodiscard]] inline const Graph &graph() const { return *_graph; }
  [[nodiscard]] inline NodeID n() const { return _graph->n(); }
  [[nodiscard]] inline NodeID last_node() const { return n() - 1; }
  [[nodiscard]] inline EdgeID m() const { return _graph->m(); }
  [[nodiscard]] inline NodeWeight node_weight(const NodeID u) const { return _graph->node_weight(u); }
  [[nodiscard]] inline EdgeWeight edge_weight(const EdgeID e) const { return _graph->edge_weight(e); }
  [[nodiscard]] inline NodeWeight total_node_weight() const { return _graph->total_node_weight(); }
  [[nodiscard]] inline NodeWeight max_node_weight() const { return _graph->max_node_weight(); }
  [[nodiscard]] inline EdgeWeight total_edge_weight() const { return _graph->total_edge_weight(); }
  [[nodiscard]] inline NodeID edge_target(const EdgeID e) const { return _graph->edge_target(e); }
  [[nodiscard]] inline NodeID degree(const NodeID u) const { return _graph->degree(u); }
  [[nodiscard]] inline auto nodes() const { return _graph->nodes(); }
  [[nodiscard]] inline auto edges() const { return _graph->edges(); }
  [[nodiscard]] inline auto adjacent_nodes(const NodeID u) const { return _graph->adjacent_nodes(u); }
  [[nodiscard]] inline auto neighbors(const NodeID u) const { return _graph->neighbors(u); }
  [[nodiscard]] inline auto incident_edges(const NodeID u) const { return _graph->incident_edges(u); }
  [[nodiscard]] inline std::size_t bucket_size(const std::size_t bucket) const { return _graph->bucket_size(bucket); }
  [[nodiscard]] inline NodeID first_node_in_bucket(const std::size_t bucket) const { return _graph->first_node_in_bucket(bucket); }
  [[nodiscard]] inline NodeID first_invalid_node_in_bucket(const std::size_t bucket) const { return _graph->first_invalid_node_in_bucket(bucket); }
  [[nodiscard]] inline std::size_t number_of_buckets() const { return _graph->number_of_buckets(); }
  [[nodiscard]] inline bool sorted() const { return _graph->sorted(); }
  template <typename Lambda> inline void pfor_nodes(Lambda &&l) const { _graph->pfor_nodes(std::forward<Lambda>(l)); }
  template <typename Lambda> inline void pfor_edges(Lambda &&l) const { _graph->pfor_edges(std::forward<Lambda>(l)); }
  // clang-format on

protected:
  const Graph *_graph;
};
} // namespace kaminpar::shm
