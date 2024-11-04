/*******************************************************************************
 * Static uncompressed CSR graph data structure.
 *
 * @file:   csr_graph.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <utility>
#include <vector>

#include <kassert/kassert.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "kaminpar-shm/datastructures/abstract_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/constexpr_utils.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/degree_buckets.h"
#include "kaminpar-common/ranges.h"

namespace kaminpar::shm {
struct CSRGraphMemory {
  StaticArray<EdgeID> nodes;
  StaticArray<NodeID> edges;
  StaticArray<NodeWeight> node_weights;
  StaticArray<EdgeWeight> edge_weights;
  std::vector<NodeID> buckets;
};

class CSRGraph : public AbstractGraph {
public:
  // Data types used by this graph
  using AbstractGraph::EdgeID;
  using AbstractGraph::EdgeWeight;
  using AbstractGraph::NodeID;
  using AbstractGraph::NodeWeight;

  // Tag for the sequential ctor.
  struct seq {};

  explicit CSRGraph(const class Graph &graph);

  CSRGraph(
      StaticArray<EdgeID> nodes,
      StaticArray<NodeID> edges,
      StaticArray<NodeWeight> node_weights = {},
      StaticArray<EdgeWeight> edge_weights = {},
      bool sorted = false
  );

  CSRGraph(
      seq,
      StaticArray<EdgeID> nodes,
      StaticArray<NodeID> edges,
      StaticArray<NodeWeight> node_weights = {},
      StaticArray<EdgeWeight> edge_weights = {},
      bool sorted = false,
      std::vector<NodeID> buckets = std::vector<NodeID>(kNumberOfDegreeBuckets<NodeID> + 1)
  );

  CSRGraph(const CSRGraph &) = delete;
  CSRGraph &operator=(const CSRGraph &) = delete;

  CSRGraph(CSRGraph &&) noexcept = default;
  CSRGraph &operator=(CSRGraph &&) noexcept = default;

  ~CSRGraph() override = default;

  //
  // Size of the graph
  //

  [[nodiscard]] inline NodeID n() const final {
    return static_cast<NodeID>(_nodes.size() - 1);
  }

  [[nodiscard]] inline EdgeID m() const final {
    return static_cast<EdgeID>(_edges.size());
  }

  //
  // Node and edge weights
  //

  [[nodiscard]] inline bool is_node_weighted() const final {
    return static_cast<NodeWeight>(n()) != total_node_weight();
  }

  [[nodiscard]] inline NodeWeight node_weight(const NodeID u) const final {
    KASSERT(!is_node_weighted() || u < _node_weights.size());
    return is_node_weighted() ? _node_weights[u] : 1;
  }

  [[nodiscard]] inline NodeWeight max_node_weight() const final {
    return _max_node_weight;
  }

  [[nodiscard]] inline NodeWeight total_node_weight() const final {
    return _total_node_weight;
  }

  void update_total_node_weight() final;

  [[nodiscard]] inline bool is_edge_weighted() const final {
    return static_cast<EdgeWeight>(m()) != total_edge_weight();
  }

  [[nodiscard]] inline EdgeWeight edge_weight(const EdgeID e) const {
    KASSERT(!is_edge_weighted() || e < _edge_weights.size());
    return is_edge_weighted() ? _edge_weights[e] : 1;
  }

  [[nodiscard]] inline EdgeWeight total_edge_weight() const final {
    return _total_edge_weight;
  }

  //
  // Iterators for nodes / edges
  //

  [[nodiscard]] inline IotaRange<NodeID> nodes() const final {
    return {static_cast<NodeID>(0), n()};
  }

  [[nodiscard]] inline IotaRange<EdgeID> edges() const final {
    return {static_cast<EdgeID>(0), m()};
  }

  [[nodiscard]] inline IotaRange<EdgeID> incident_edges(const NodeID u) const final {
    KASSERT(u + 1 < _nodes.size());
    return {_nodes[u], _nodes[u + 1]};
  }

  //
  // Node degree
  //

  [[nodiscard]] inline NodeID max_degree() const final {
    return _max_degree;
  }

  [[nodiscard]] inline NodeID degree(const NodeID u) const final {
    return static_cast<NodeID>(_nodes[u + 1] - _nodes[u]);
  }

  //
  // Graph operations not part of the interface
  //

  [[nodiscard]] EdgeID first_edge(const NodeID u) const {
    return _nodes[u];
  }

  [[nodiscard]] EdgeID first_invalid_edge(const NodeID u) const {
    return _nodes[u + 1];
  }

  [[nodiscard]] NodeID edge_target(const EdgeID e) const {
    return _edges[e];
  }

  //
  // Graph operations
  //

  template <typename Lambda> inline void adjacent_nodes(const NodeID u, Lambda &&l) const {
    KASSERT(u < n());

    constexpr bool kDontDecodeEdgeWeights = std::is_invocable_v<Lambda, NodeID>;
    constexpr bool kDecodeEdgeWeights = std::is_invocable_v<Lambda, NodeID, EdgeWeight>;
    static_assert(kDontDecodeEdgeWeights || kDecodeEdgeWeights);

    using LambdaReturnType = std::conditional_t<
        kDecodeEdgeWeights,
        std::invoke_result<Lambda, NodeID, EdgeWeight>,
        std::invoke_result<Lambda, NodeID>>::type;
    constexpr bool kNonStoppable = std::is_void_v<LambdaReturnType>;

    const auto decode_adjacent_nodes = [&](auto &&decode_edge_weight) {
      const auto invoke_caller = [&](const EdgeID edge) {
        if constexpr (kDecodeEdgeWeights) {
          return l(_edges[edge], decode_edge_weight(edge));
        } else {
          return l(_edges[edge]);
        }
      };

      const EdgeID from = _nodes[u];
      const EdgeID to = _nodes[u + 1];
      for (EdgeID edge = from; edge < to; ++edge) {
        if constexpr (kNonStoppable) {
          invoke_caller(edge);
        } else {
          const bool stop = invoke_caller(edge);
          if (stop) {
            return;
          }
        }
      }
    };

    if (is_edge_weighted()) {
      decode_adjacent_nodes([&](const EdgeID edge) { return _edge_weights[edge]; });
    } else {
      decode_adjacent_nodes([](const EdgeID) { return 1; });
    }
  }

  template <typename Lambda> inline void neighbors(const NodeID u, Lambda &&l) const {
    KASSERT(u < n());

    constexpr bool kDontDecodeEdgeWeights = std::is_invocable_v<Lambda, EdgeID, NodeID>;
    constexpr bool kDecodeEdgeWeights = std::is_invocable_v<Lambda, EdgeID, NodeID, EdgeWeight>;
    static_assert(kDontDecodeEdgeWeights || kDecodeEdgeWeights);

    using LambdaReturnType = std::conditional_t<
        kDecodeEdgeWeights,
        std::invoke_result<Lambda, EdgeID, NodeID, EdgeWeight>,
        std::invoke_result<Lambda, EdgeID, NodeID>>::type;
    constexpr bool kNonStoppable = std::is_void_v<LambdaReturnType>;

    const auto decode_neighbors = [&](auto &&decode_edge_weight) {
      const auto invoke_caller = [&](const EdgeID edge) {
        if constexpr (kDecodeEdgeWeights) {
          return l(edge, _edges[edge], decode_edge_weight(edge));
        } else {
          return l(edge, _edges[edge]);
        }
      };

      const EdgeID from = _nodes[u];
      const EdgeID to = _nodes[u + 1];
      for (EdgeID edge = from; edge < to; ++edge) {
        if constexpr (kNonStoppable) {
          invoke_caller(edge);
        } else {
          const bool stop = invoke_caller(edge);
          if (stop) {
            return;
          }
        }
      }
    };

    if (is_edge_weighted()) {
      decode_neighbors([&](const EdgeID edge) { return _edge_weights[edge]; });
    } else {
      decode_neighbors([](const EdgeID) { return 1; });
    }
  }

  template <typename Lambda>
  inline void neighbors(const NodeID u, const NodeID max_neighbor_count, Lambda &&l) const {
    KASSERT(u < n());

    constexpr bool kDontDecodeEdgeWeights = std::is_invocable_v<Lambda, EdgeID, NodeID>;
    constexpr bool kDecodeEdgeWeights = std::is_invocable_v<Lambda, EdgeID, NodeID, EdgeWeight>;
    static_assert(kDontDecodeEdgeWeights || kDecodeEdgeWeights);

    using LambdaReturnType = std::conditional_t<
        kDecodeEdgeWeights,
        std::invoke_result<Lambda, EdgeID, NodeID, EdgeWeight>,
        std::invoke_result<Lambda, EdgeID, NodeID>>::type;
    constexpr bool kNonStoppable = std::is_void_v<LambdaReturnType>;

    const auto decode_neighbors = [&](auto &&decode_edge_weight) {
      const auto invoke_caller = [&](const EdgeID edge) {
        if constexpr (kDecodeEdgeWeights) {
          return l(edge, _edges[edge], decode_edge_weight(edge));
        } else {
          return l(edge, _edges[edge]);
        }
      };

      const EdgeID from = _nodes[u];
      const NodeID degree = static_cast<NodeID>(_nodes[u + 1] - from);
      const EdgeID to = from + std::min(degree, max_neighbor_count);
      for (EdgeID edge = from; edge < to; ++edge) {
        if constexpr (kNonStoppable) {
          invoke_caller(edge);
        } else {
          const bool stop = invoke_caller(edge);
          if (stop) {
            return;
          }
        }
      }
    };

    if (is_edge_weighted()) {
      decode_neighbors([&](const EdgeID edge) { return _edge_weights[edge]; });
    } else {
      decode_neighbors([](const EdgeID) { return 1; });
    }
  }

  //
  // Parallel iteration
  //

  template <typename Lambda> inline void pfor_nodes(Lambda &&l) const {
    tbb::parallel_for(static_cast<NodeID>(0), n(), std::forward<Lambda>(l));
  }

  template <typename Lambda> inline void pfor_edges(Lambda &&l) const {
    tbb::parallel_for(static_cast<EdgeID>(0), m(), std::forward<Lambda>(l));
  }

  template <typename Lambda>
  inline void pfor_neighbors(
      const NodeID u, const NodeID max_num_neighbors, const NodeID grainsize, Lambda &&l
  ) const {
    KASSERT(u < n());
    constexpr bool kInvokeDirectly = std::is_invocable_v<Lambda, EdgeID, NodeID, EdgeWeight>;

    const EdgeID from = _nodes[u];
    const NodeID degree = static_cast<NodeID>(_nodes[u + 1] - from);
    const EdgeID to = from + std::min(degree, max_num_neighbors);

    const auto visit_neighbors = [&](auto &&decode_edge_weight) {
      tbb::parallel_for(tbb::blocked_range<EdgeID>(from, to, grainsize), [&](const auto &range) {
        const auto end = range.end();

        invoke_indirect<kInvokeDirectly>(std::forward<Lambda>(l), [&](auto &&l2) {
          for (EdgeID e = range.begin(); e < end; ++e) {
            l2(e, _edges[e], decode_edge_weight(e));
          }
        });
      });
    };

    if (is_edge_weighted()) {
      visit_neighbors([&](const EdgeID e) { return _edge_weights[e]; });
    } else {
      visit_neighbors([](const EdgeID) { return 1; });
    }
  }

  //
  // Graph permutation
  //

  inline void set_permutation(StaticArray<NodeID> permutation) final {
    _permutation = std::move(permutation);
  }

  [[nodiscard]] inline bool permuted() const final {
    return !_permutation.empty();
  }

  [[nodiscard]] inline NodeID map_original_node(const NodeID u) const final {
    KASSERT(u < _permutation.size());
    return _permutation[u];
  }

  [[nodiscard]] inline StaticArray<NodeID> &&take_raw_permutation() final {
    return std::move(_permutation);
  }

  //
  // Degree buckets
  //

  [[nodiscard]] inline bool sorted() const final {
    return _sorted;
  }

  [[nodiscard]] inline std::size_t number_of_buckets() const final {
    return _number_of_buckets;
  }

  [[nodiscard]] inline std::size_t bucket_size(const std::size_t bucket) const final {
    return _buckets[bucket + 1] - _buckets[bucket];
  }

  [[nodiscard]] inline NodeID first_node_in_bucket(const std::size_t bucket) const final {
    return _buckets[bucket];
  }

  [[nodiscard]] inline NodeID first_invalid_node_in_bucket(const std::size_t bucket) const final {
    return first_node_in_bucket(bucket + 1);
  }

  //
  // Isolated nodes
  //

  void remove_isolated_nodes(const NodeID num_isolated_nodes);

  void integrate_isolated_nodes();

  //
  // Direct member access -- used for some "low level" operations
  //

  template <typename Lambda> decltype(auto) reified(Lambda &&l) const {
    return l(*this);
  }

  [[nodiscard]] inline StaticArray<EdgeID> &raw_nodes() {
    return _nodes;
  }

  [[nodiscard]] inline const StaticArray<EdgeID> &raw_nodes() const {
    return _nodes;
  }

  [[nodiscard]] inline StaticArray<NodeID> &raw_edges() {
    return _edges;
  }

  [[nodiscard]] inline const StaticArray<NodeID> &raw_edges() const {
    return _edges;
  }

  [[nodiscard]] inline StaticArray<NodeWeight> &raw_node_weights() {
    return _node_weights;
  }

  [[nodiscard]] inline const StaticArray<NodeWeight> &raw_node_weights() const {
    return _node_weights;
  }

  [[nodiscard]] inline StaticArray<EdgeWeight> &raw_edge_weights() {
    return _edge_weights;
  }

  [[nodiscard]] inline const StaticArray<EdgeWeight> &raw_edge_weights() const {
    return _edge_weights;
  }

  [[nodiscard]] inline StaticArray<EdgeID> &&take_raw_nodes() {
    return std::move(_nodes);
  }

  [[nodiscard]] inline StaticArray<NodeID> &&take_raw_edges() {
    return std::move(_edges);
  }

  [[nodiscard]] inline StaticArray<NodeWeight> &&take_raw_node_weights() {
    return std::move(_node_weights);
  }

  [[nodiscard]] inline StaticArray<EdgeWeight> &&take_raw_edge_weights() {
    return std::move(_edge_weights);
  }

  [[nodiscard]] inline std::vector<NodeID> &&take_raw_buckets() {
    return std::move(_buckets);
  }

private:
  void init_degree_buckets();

  StaticArray<EdgeID> _nodes;
  StaticArray<NodeID> _edges;
  StaticArray<NodeWeight> _node_weights;
  StaticArray<EdgeWeight> _edge_weights;

  NodeWeight _max_node_weight = kInvalidNodeWeight;
  NodeWeight _total_node_weight = kInvalidNodeWeight;
  EdgeWeight _total_edge_weight = kInvalidEdgeWeight;

  NodeID _max_degree;

  StaticArray<NodeID> _permutation;
  bool _sorted;
  std::vector<NodeID> _buckets;
  std::size_t _number_of_buckets = 0;
};

namespace debug {
bool validate_graph(const CSRGraph &graph, bool undirected = true, NodeID num_pseudo_nodes = 0);

bool validate_graph(
    const NodeID n,
    const StaticArray<EdgeID> &xadj,
    const StaticArray<NodeID> &adjncy,
    const StaticArray<NodeWeight> &vwgt,
    const StaticArray<EdgeWeight> &adjwgt,
    bool undirected = true,
    NodeID num_pseudo_nodes = 0
);

CSRGraph sort_neighbors(CSRGraph graph);
} // namespace debug
} // namespace kaminpar::shm
