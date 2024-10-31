/*******************************************************************************
 * Static distributed compressed graph data structure.
 *
 * @file:   distributed_compressed_graph.h
 * @author: Daniel Salwasser
 * @date:   07.06.2024
 ******************************************************************************/
#pragma once

#include <vector>

#include "kaminpar-mpi/utils.h"

#include "kaminpar-dist/datastructures/abstract_distributed_graph.h"
#include "kaminpar-dist/datastructures/ghost_node_mapper.h"
#include "kaminpar-dist/datastructures/growt.h"
#include "kaminpar-dist/dkaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/degree_buckets.h"
#include "kaminpar-common/graph_compression/compressed_neighborhoods.h"

namespace kaminpar::dist {

class DistributedCompressedGraph : public AbstractDistributedGraph {

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

  using CompressedNeighborhoods = kaminpar::CompressedNeighborhoods<NodeID, EdgeID, EdgeWeight>;

  DistributedCompressedGraph(
      StaticArray<GlobalNodeID> node_distribution,
      StaticArray<GlobalEdgeID> edge_distribution,
      CompressedNeighborhoods compressed_neighborhoods,
      CompactGhostNodeMapping ghost_node_mapping,
      const bool sorted,
      MPI_Comm comm
  )
      : DistributedCompressedGraph(
            std::move(node_distribution),
            std::move(edge_distribution),
            std::move(compressed_neighborhoods),
            {},
            std::move(ghost_node_mapping),
            sorted,
            comm
        ) {}

  DistributedCompressedGraph(
      StaticArray<GlobalNodeID> node_distribution,
      StaticArray<GlobalEdgeID> edge_distribution,
      CompressedNeighborhoods compressed_neighborhoods,
      StaticArray<NodeWeight> node_weights,
      CompactGhostNodeMapping ghost_node_mapping,
      const bool sorted,
      MPI_Comm comm
  )
      : _node_distribution(std::move(node_distribution)),
        _edge_distribution(std::move(edge_distribution)),
        _compressed_neighborhoods(std::move(compressed_neighborhoods)),
        _node_weights(std::move(node_weights)),
        _ghost_node_mapping(std::move(ghost_node_mapping)),
        _sorted(sorted),
        _communicator(comm) {
    const PEID rank = mpi::get_comm_rank(communicator());

    _n = _compressed_neighborhoods.num_nodes();
    _m = compressed_neighborhoods.num_edges();
    _ghost_n = ghost_node_mapping.num_ghost_nodes();
    _offset_n = _node_distribution[rank];
    _offset_m = _edge_distribution[rank];
    _global_n = _node_distribution.back();
    _global_m = _edge_distribution.back();

    init_total_weights();
    init_communication_metrics();
    init_degree_buckets();
  }

  DistributedCompressedGraph(const DistributedCompressedGraph &) = delete;
  DistributedCompressedGraph &operator=(const DistributedCompressedGraph &) = delete;

  DistributedCompressedGraph(DistributedCompressedGraph &&) noexcept = default;
  DistributedCompressedGraph &operator=(DistributedCompressedGraph &&) noexcept = default;

  ~DistributedCompressedGraph() override = default;

  //
  // Size of the graph
  //

  [[nodiscard]] inline GlobalNodeID global_n() const final {
    return _global_n;
  }

  [[nodiscard]] inline GlobalEdgeID global_m() const final {
    return _global_m;
  }

  [[nodiscard]] inline NodeID n() const final {
    return _n;
  }

  [[nodiscard]] inline NodeID n(const PEID pe) const final {
    KASSERT(pe < static_cast<PEID>(_node_distribution.size()));
    return _node_distribution[pe + 1] - _node_distribution[pe];
  }

  [[nodiscard]] inline NodeID ghost_n() const final {
    return _ghost_n;
  }

  [[nodiscard]] inline NodeID total_n() const final {
    return ghost_n() + n();
  }

  [[nodiscard]] inline EdgeID m() const final {
    return _m;
  }

  [[nodiscard]] inline EdgeID m(const PEID pe) const final {
    KASSERT(pe < static_cast<PEID>(_edge_distribution.size()));
    return _edge_distribution[pe + 1] - _edge_distribution[pe];
  }

  [[nodiscard]] inline GlobalNodeID offset_n() const final {
    return _offset_n;
  }

  [[nodiscard]] inline GlobalNodeID offset_n(const PEID pe) const final {
    return _node_distribution[pe];
  }

  [[nodiscard]] inline GlobalEdgeID offset_m() const final {
    return _offset_m;
  }

  [[nodiscard]] inline GlobalEdgeID offset_m(const PEID pe) const final {
    return _edge_distribution[pe];
  }

  //
  // Node and edge weights
  //

  [[nodiscard]] inline bool is_node_weighted() const final {
    return !_node_weights.empty();
  }

  [[nodiscard]] inline NodeWeight node_weight(const NodeID u) const final {
    return is_node_weighted() ? _node_weights[u] : 1;
  }

  [[nodiscard]] inline NodeWeight max_node_weight() const final {
    return _max_node_weight;
  }

  [[nodiscard]] inline NodeWeight global_max_node_weight() const final {
    return _global_max_node_weight;
  }

  [[nodiscard]] inline NodeWeight total_node_weight() const final {
    return _total_node_weight;
  }

  [[nodiscard]] inline GlobalNodeWeight global_total_node_weight() const final {
    return _global_total_node_weight;
  }

  [[nodiscard]] inline bool is_edge_weighted() const final {
    return _compressed_neighborhoods.has_edge_weights();
  }

  [[nodiscard]] inline EdgeWeight total_edge_weight() const final {
    return _compressed_neighborhoods.total_edge_weight();
  }

  [[nodiscard]] inline GlobalEdgeWeight global_total_edge_weight() const final {
    return _global_total_edge_weight;
  }

  //
  // Node ownership
  //

  [[nodiscard]] inline bool is_owned_global_node(const GlobalNodeID global_u) const final {
    return (offset_n() <= global_u && global_u < offset_n() + n());
  }

  [[nodiscard]] inline bool contains_global_node(const GlobalNodeID global_u) const final {
    return is_owned_global_node(global_u) || _ghost_node_mapping.contains_global_as_ghost(global_u);
  }

  [[nodiscard]] inline bool contains_local_node(const NodeID local_u) const final {
    return local_u < total_n();
  }

  //
  // Node type
  //

  [[nodiscard]] inline bool is_ghost_node(const NodeID u) const final {
    KASSERT(u < total_n());
    return u >= n();
  }

  [[nodiscard]] inline bool is_owned_node(const NodeID u) const final {
    KASSERT(u < total_n());
    return u < n();
  }

  [[nodiscard]] inline PEID ghost_owner(const NodeID u) const final {
    KASSERT(is_ghost_node(u));
    KASSERT(u - n() < _ghost_node_mapping.num_ghost_nodes());
    return _ghost_node_mapping.ghost_owner(u - n());
  }

  [[nodiscard]] inline NodeID
  map_remote_node(const NodeID their_lnode, const PEID owner) const final {
    const auto gnode = static_cast<GlobalNodeID>(their_lnode + offset_n(owner));
    return global_to_local_node(gnode);
  }

  [[nodiscard]] inline GlobalNodeID local_to_global_node(const NodeID local_u) const final {
    KASSERT(contains_local_node(local_u));
    return is_owned_node(local_u) ? _offset_n + local_u
                                  : _ghost_node_mapping.ghost_to_global(local_u - n());
  }

  [[nodiscard]] inline NodeID global_to_local_node(const GlobalNodeID global_u) const final {
    KASSERT(contains_global_node(global_u));

    if (offset_n() <= global_u && global_u < offset_n() + n()) {
      return global_u - offset_n();
    } else {
      return _ghost_node_mapping.global_to_ghost(global_u);
    }
  }

  //
  // Iterators for nodes / edges
  //

  [[nodiscard]] inline IotaRange<NodeID> nodes(const NodeID from, const NodeID to) const final {
    return {from, to};
  }

  [[nodiscard]] inline IotaRange<NodeID> nodes() const final {
    return nodes(0, n());
  }

  [[nodiscard]] inline IotaRange<NodeID> ghost_nodes() const final {
    return {n(), total_n()};
  }

  [[nodiscard]] inline IotaRange<NodeID> all_nodes() const final {
    return {static_cast<NodeID>(0), total_n()};
  }

  [[nodiscard]] inline IotaRange<EdgeID> edges() const final {
    return {static_cast<EdgeID>(0), m()};
  }

  [[nodiscard]] inline IotaRange<EdgeID> incident_edges(const NodeID u) const final {
    return _compressed_neighborhoods.incident_edges(u);
  }

  //
  // Graph operations
  //

  template <typename Lambda> inline void adjacent_nodes(const NodeID u, Lambda &&l) const {
    constexpr bool kDontDecodeEdgeWeights = std::is_invocable_v<Lambda, NodeID>;
    constexpr bool kDecodeEdgeWeights = std::is_invocable_v<Lambda, NodeID, EdgeWeight>;
    static_assert(kDontDecodeEdgeWeights || kDecodeEdgeWeights);

    _compressed_neighborhoods.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      if constexpr (kDecodeEdgeWeights) {
        return l(v, w);
      } else {
        return l(v);
      }
    });
  }

  template <typename Lambda> inline void neighbors(const NodeID u, Lambda &&l) const {
    constexpr bool kDontDecodeEdgeWeights = std::is_invocable_v<Lambda, EdgeID, NodeID>;
    constexpr bool kDecodeEdgeWeights = std::is_invocable_v<Lambda, EdgeID, NodeID, EdgeWeight>;
    static_assert(kDontDecodeEdgeWeights || kDecodeEdgeWeights);

    _compressed_neighborhoods.neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
      if constexpr (kDecodeEdgeWeights) {
        return l(e, v, w);
      } else {
        return l(e, v);
      }
    });
  }

  template <typename Lambda>
  inline void neighbors(const NodeID u, const NodeID max_num_neighbors, Lambda &&l) const {
    constexpr bool kDontDecodeEdgeWeights = std::is_invocable_v<Lambda, EdgeID, NodeID>;
    constexpr bool kDecodeEdgeWeights = std::is_invocable_v<Lambda, EdgeID, NodeID, EdgeWeight>;
    static_assert(kDontDecodeEdgeWeights || kDecodeEdgeWeights);

    _compressed_neighborhoods
        .neighbors(u, max_num_neighbors, [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
          if constexpr (kDecodeEdgeWeights) {
            return l(e, v, w);
          } else {
            return l(e, v);
          }
        });
  }

  //
  // Parallel iteration
  //

  template <typename Lambda>
  inline void pfor_nodes(const NodeID from, const NodeID to, Lambda &&l) const {
    tbb::parallel_for(from, to, std::forward<Lambda>(l));
  }

  template <typename Lambda>
  inline void pfor_nodes_range(const NodeID from, const NodeID to, Lambda &&l) const {
    tbb::parallel_for(tbb::blocked_range<NodeID>(from, to), std::forward<Lambda>(l));
  }

  template <typename Lambda> inline void pfor_ghost_nodes(Lambda &&l) const {
    pfor_nodes(n(), total_n(), std::forward<Lambda>(l));
  }

  template <typename Lambda> inline void pfor_nodes(Lambda &&l) const {
    pfor_nodes(0, n(), std::forward<Lambda>(l));
  }

  template <typename Lambda> inline void pfor_all_nodes(Lambda &&l) const {
    pfor_nodes(0, total_n(), std::forward<Lambda>(l));
  }

  template <typename Lambda> inline void pfor_nodes_range(Lambda &&l) const {
    pfor_nodes_range(0, n(), std::forward<Lambda>(l));
  }

  template <typename Lambda> inline void pfor_all_nodes_range(Lambda &&l) const {
    pfor_nodes_range(0, total_n(), std::forward<Lambda>(l));
  }

  template <typename Lambda> inline void pfor_edges(Lambda &&l) const {
    pfor_nodes([&](const NodeID u) { neighbors(u, std::forward<Lambda>(l)); });
  }

  //
  // Access methods
  //

  [[nodiscard]] inline NodeID degree(const NodeID u) const final {
    KASSERT(is_owned_node(u));
    return _compressed_neighborhoods.degree(u);
  }

  [[nodiscard]] inline const StaticArray<NodeWeight> &node_weights() const final {
    return _node_weights;
  }

  inline void set_ghost_node_weight(const NodeID ghost_node, const NodeWeight weight) final {
    KASSERT(is_ghost_node(ghost_node));
    KASSERT(is_node_weighted());
    _node_weights[ghost_node] = weight;
  }

  [[nodiscard]] inline const StaticArray<GlobalNodeID> &node_distribution() const final {
    return _node_distribution;
  }

  [[nodiscard]] inline GlobalNodeID node_distribution(const PEID pe) const final {
    KASSERT(static_cast<std::size_t>(pe) < _node_distribution.size());
    return _node_distribution[pe];
  }

  [[nodiscard]] inline PEID find_owner_of_global_node(const GlobalNodeID u) const final {
    KASSERT(u < global_n());
    auto it = std::upper_bound(_node_distribution.begin() + 1, _node_distribution.end(), u);
    KASSERT(it != _node_distribution.end());
    return static_cast<PEID>(std::distance(_node_distribution.begin(), it) - 1);
  }

  [[nodiscard]] inline const StaticArray<GlobalEdgeID> &edge_distribution() const final {
    return _edge_distribution;
  }

  [[nodiscard]] inline GlobalEdgeID edge_distribution(const PEID pe) const final {
    KASSERT(static_cast<std::size_t>(pe) < _edge_distribution.size());
    return _edge_distribution[pe];
  }

  //
  // Cached inter-PE metrics
  //

  [[nodiscard]] inline EdgeID edge_cut_to_pe(const PEID pe) const final {
    KASSERT(static_cast<std::size_t>(pe) < _edge_cut_to_pe.size());
    return _edge_cut_to_pe[pe];
  }

  [[nodiscard]] inline EdgeID comm_vol_to_pe(const PEID pe) const final {
    KASSERT(static_cast<std::size_t>(pe) < _comm_vol_to_pe.size());
    return _comm_vol_to_pe[pe];
  }

  [[nodiscard]] inline MPI_Comm communicator() const final {
    return _communicator;
  }

  //
  // High degree classification
  //

  void init_high_degree_info(const EdgeID high_degree_threshold) const final;

  [[nodiscard]] bool is_high_degree_node(const NodeID node) const final {
    KASSERT(_high_degree_ghost_node.size() == ghost_n());
    KASSERT(!is_ghost_node(node) || node - n() < _high_degree_ghost_node.size());
    return is_ghost_node(node) ? _high_degree_ghost_node[node - n()]
                               : degree(node) > _high_degree_threshold;
  }

  //
  // Graph permutation
  //

  void set_permutation(StaticArray<NodeID> permutation) final {
    _permutation = std::move(permutation);
  }

  [[nodiscard]] inline bool permuted() const final {
    return !_permutation.empty();
  }

  [[nodiscard]] inline NodeID map_original_node(const NodeID u) const final {
    KASSERT(permuted());
    KASSERT(u < _permutation.size());
    return _permutation[u];
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
  // Graph permutation by coloring
  //

  inline void set_color_sorted(StaticArray<NodeID> color_sizes) final {
    KASSERT(color_sizes.front() == 0u);
    KASSERT(color_sizes.back() == n());
    _color_sizes = std::move(color_sizes);
  }

  [[nodiscard]] inline bool color_sorted() const final {
    return !_color_sizes.empty();
  }

  [[nodiscard]] inline std::size_t number_of_colors() const final {
    return _color_sizes.size() - 1;
  }

  [[nodiscard]] inline NodeID color_size(const std::size_t c) const final {
    KASSERT(c < number_of_colors());
    return _color_sizes[c + 1] - _color_sizes[c];
  }

  [[nodiscard]] inline const StaticArray<NodeID> &get_color_sizes() const final {
    return _color_sizes;
  }

  //
  // Statistics about graph compression
  //

  [[nodiscard]] double compression_ratio() const {
    std::size_t uncompressed_size = (n() + 1) * sizeof(EdgeID) + m() * sizeof(NodeID);
    std::size_t compressed_size = _compressed_neighborhoods.memory_space();

    if (is_node_weighted()) {
      uncompressed_size += n() * sizeof(NodeWeight);
      compressed_size += n() * sizeof(NodeWeight);
    }

    if (is_edge_weighted()) {
      uncompressed_size += m() * sizeof(EdgeWeight);
    }

    return uncompressed_size / static_cast<double>(compressed_size);
  }

  [[nodiscard]] std::size_t memory_space() const {
    std::size_t memory_space = _compressed_neighborhoods.memory_space();

    if (is_node_weighted()) {
      memory_space += n() * sizeof(NodeWeight);
    }

    return memory_space;
  }

  [[nodiscard]] std::size_t uncompressed_memory_space() const {
    std::size_t memory_space = (n() + 1) * sizeof(EdgeID) + m() * sizeof(NodeID);

    if (is_node_weighted()) {
      memory_space += n() * sizeof(NodeWeight);
    }

    if (is_edge_weighted()) {
      memory_space += m() * sizeof(EdgeWeight);
    }

    return memory_space;
  }

  //
  // Functions to access raw members of this graph
  //

  [[nodiscard]] const auto &raw_node_weights() const {
    return _node_weights;
  }

private:
  void init_degree_buckets();
  void init_total_weights();
  void init_communication_metrics();

  NodeID _n;
  EdgeID _m;
  NodeID _ghost_n;
  GlobalNodeID _offset_n;
  GlobalEdgeID _offset_m;
  GlobalNodeID _global_n;
  GlobalEdgeID _global_m;

  NodeWeight _total_node_weight{};
  GlobalNodeWeight _global_total_node_weight{};
  NodeWeight _max_node_weight{};
  NodeWeight _global_max_node_weight{};

  GlobalEdgeWeight _global_total_edge_weight{};

  StaticArray<GlobalNodeID> _node_distribution{};
  StaticArray<GlobalEdgeID> _edge_distribution{};

  CompressedNeighborhoods _compressed_neighborhoods;
  StaticArray<NodeWeight> _node_weights{};

  CompactGhostNodeMapping _ghost_node_mapping;

  // mutable for lazy initialization
  mutable StaticArray<std::uint8_t> _high_degree_ghost_node{};
  mutable EdgeID _high_degree_threshold = 0;

  std::vector<EdgeID> _edge_cut_to_pe{};
  std::vector<EdgeID> _comm_vol_to_pe{};

  StaticArray<NodeID> _permutation;
  bool _sorted = false;
  std::vector<NodeID> _buckets = std::vector<NodeID>(kNumberOfDegreeBuckets<NodeID> + 1);
  std::size_t _number_of_buckets = 0;

  StaticArray<NodeID> _color_sizes{};

  MPI_Comm _communicator;
};

} // namespace kaminpar::dist
