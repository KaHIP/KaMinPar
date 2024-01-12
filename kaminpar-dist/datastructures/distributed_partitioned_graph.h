/*******************************************************************************
 * Wrapper with a dynamic partition for a static distributed graph.
 *
 * @file:   distributed_partitioned_graph.h
 * @author: Daniel Seemaier
 * @date:   21.06.2023
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/datastructures/distributed_graph.h"

namespace kaminpar::dist {
class DistributedPartitionedGraph {
public:
  using NodeID = DistributedGraph::NodeID;
  using GlobalNodeID = DistributedGraph::GlobalNodeID;
  using NodeWeight = DistributedGraph::NodeWeight;
  using GlobalNodeWeight = DistributedGraph::GlobalNodeWeight;
  using EdgeID = DistributedGraph::EdgeID;
  using GlobalEdgeID = DistributedGraph::GlobalEdgeID;
  using EdgeWeight = DistributedGraph::EdgeWeight;
  using GlobalEdgeWeight = DistributedGraph::GlobalEdgeWeight;
  using BlockID = ::kaminpar::dist::BlockID;
  using BlockWeight = ::kaminpar::dist::BlockWeight;

  DistributedPartitionedGraph(
      const DistributedGraph *graph, const BlockID k, StaticArray<BlockID> partition
  )
      : DistributedPartitionedGraph(graph, k, std::move(partition), StaticArray<BlockWeight>(k)) {
    init_block_weights();
  }

  DistributedPartitionedGraph(
      const DistributedGraph *graph,
      const BlockID k,
      StaticArray<BlockID> partition,
      StaticArray<BlockWeight> block_weights
  )
      : _graph(graph),
        _k(k),
        _partition(std::move(partition)),
        _block_weights(std::move(block_weights)) {
    KASSERT(
        _partition.size() == _graph->total_n(),
        "partition size does not match the number of nodes in the graph"
    );
    KASSERT(
        [&] {
          for (const BlockID b : _partition) {
            if (b >= _k) {
              return false;
            }
          }
          return true;
        }(),
        "partition assignes out-of-bound labels to some nodes"
    );
  }

  DistributedPartitionedGraph(const DistributedGraph *graph, const BlockID k)
      : DistributedPartitionedGraph(
            graph, k, StaticArray<BlockID>(graph->total_n()), StaticArray<BlockWeight>(k)
        ) {}

  DistributedPartitionedGraph() : _graph(nullptr), _k(0), _partition() {}

  DistributedPartitionedGraph(const DistributedPartitionedGraph &) = delete;
  DistributedPartitionedGraph &operator=(const DistributedPartitionedGraph &) = delete;

  DistributedPartitionedGraph(DistributedPartitionedGraph &&) noexcept = default;
  DistributedPartitionedGraph &operator=(DistributedPartitionedGraph &&) noexcept = default;

  [[nodiscard]] const DistributedGraph &graph() const {
    return *_graph;
  }

  void UNSAFE_set_graph(const DistributedGraph *graph) {
    _graph = graph;
  }

  // Delegates to _graph
  // clang-format off
  [[nodiscard]] inline GlobalNodeID global_n() const { return _graph->global_n(); }
  [[nodiscard]] inline GlobalEdgeID global_m() const { return _graph->global_m(); }
  [[nodiscard]] inline NodeID n() const { return _graph->n(); }
  [[nodiscard]] inline NodeID ghost_n() const { return _graph->ghost_n(); }
  [[nodiscard]] inline NodeID total_n() const { return _graph->total_n(); }
  [[nodiscard]] inline EdgeID m() const { return _graph->m(); }
  [[nodiscard]] inline GlobalNodeID offset_n(const PEID pe) const { return _graph->offset_n(pe); }
  [[nodiscard]] inline GlobalNodeID offset_n() const { return _graph->offset_n(); }
  [[nodiscard]] inline GlobalEdgeID offset_m() const { return _graph->offset_m(); }
  [[nodiscard]] inline NodeWeight total_node_weight() const { return _graph->total_node_weight(); }
  [[nodiscard]] inline NodeWeight max_node_weight() const { return _graph->max_node_weight(); }
  [[nodiscard]] inline bool contains_global_node(const GlobalNodeID global_u) const { return _graph->contains_global_node(global_u); }
  [[nodiscard]] inline bool contains_local_node(const NodeID local_u) const { return _graph->contains_local_node(local_u); }
  [[nodiscard]] inline bool is_ghost_node(const NodeID u) const { return _graph->is_ghost_node(u); }
  [[nodiscard]] inline bool is_owned_node(const NodeID u) const { return _graph->is_owned_node(u); }
  [[nodiscard]] inline bool is_owned_global_node(const GlobalNodeID u) const { return _graph->is_owned_global_node(u); }
  [[nodiscard]] inline PEID find_owner_of_global_node(const GlobalNodeID u) const { return _graph->find_owner_of_global_node(u); }
  [[nodiscard]] inline PEID ghost_owner(const NodeID u) const { return _graph->ghost_owner(u); }
  [[nodiscard]] inline GlobalNodeID local_to_global_node(const NodeID local_u) const { return _graph->local_to_global_node(local_u); }
  [[nodiscard]] inline NodeID map_foreign_node(const NodeID lnode, const PEID owner) const { return _graph->map_foreign_node(lnode, owner); }
  [[nodiscard]] inline NodeID global_to_local_node(const GlobalNodeID global_u) const { return _graph->global_to_local_node(global_u); }
  [[nodiscard]] inline NodeWeight node_weight(const NodeID u) const { return _graph->node_weight(u); }
  [[nodiscard]] inline EdgeWeight edge_weight(const EdgeID e) const { return _graph->edge_weight(e); }
  [[nodiscard]] inline EdgeID first_edge(const NodeID u) const { return _graph->first_edge(u); }
  [[nodiscard]] inline EdgeID first_invalid_edge(const NodeID u) const { return _graph->first_invalid_edge(u); }
  [[nodiscard]] inline NodeID edge_target(const EdgeID e) const { return _graph->edge_target(e); }
  [[nodiscard]] inline NodeID degree(const NodeID u) const { return _graph->degree(u); }
  [[nodiscard]] inline const auto &node_distribution() const { return _graph->node_distribution(); }
  [[nodiscard]] inline GlobalNodeID node_distribution(const PEID pe) const { return _graph->node_distribution(pe); }
  [[nodiscard]] inline const auto &edge_distribution() const { return _graph->edge_distribution(); }
  [[nodiscard]] inline GlobalEdgeID edge_distribution(const PEID pe) const { return _graph->edge_distribution(pe); }
  [[nodiscard]] const auto &raw_nodes() const { return _graph->raw_nodes(); }
  [[nodiscard]] const auto &raw_node_weights() const { return _graph->raw_node_weights(); }
  [[nodiscard]] const auto &raw_edges() const { return _graph->raw_edges(); }
  [[nodiscard]] const auto &raw_edge_weights() const { return _graph->raw_edge_weights(); }
  template<typename Lambda> inline void pfor_nodes(const NodeID from, const NodeID to, Lambda &&l) const { _graph->pfor_nodes(from, to, std::forward<Lambda>(l)); }
  template<typename Lambda> inline void pfor_nodes_range(const NodeID from, const NodeID to, Lambda &&l) const { _graph->pfor_nodes_range(from, to, std::forward<Lambda>(l)); }
  template<typename Lambda> inline void pfor_all_nodes(Lambda &&l) const { _graph->pfor_all_nodes(std::forward<Lambda>(l)); }
  template<typename Lambda> inline void pfor_nodes(Lambda &&l) const { _graph->pfor_nodes(std::forward<Lambda>(l)); }
  template<typename Lambda> inline void pfor_nodes_range(Lambda &&l) const { _graph->pfor_nodes_range(std::forward<Lambda>(l)); }
  template<typename Lambda> inline void pfor_ghost_nodes(Lambda &&l) const { _graph->pfor_ghost_nodes(std::forward<Lambda>(l)); }
  template<typename Lambda> inline void pfor_edges(Lambda &&l) const { _graph->pfor_edges(std::forward<Lambda>(l)); }
  [[nodiscard]] inline auto nodes() const { return _graph->nodes(); }
  [[nodiscard]] inline auto nodes(const NodeID from, const NodeID to) const { return _graph->nodes(from, to); }
  [[nodiscard]] inline auto ghost_nodes() const { return _graph->ghost_nodes(); }
  [[nodiscard]] inline auto all_nodes() const { return _graph->all_nodes(); }
  [[nodiscard]] inline auto edges() const { return _graph->edges(); }
  [[nodiscard]] inline auto incident_edges(const NodeID u) const { return _graph->incident_edges(u); }
  [[nodiscard]] inline auto adjacent_nodes(const NodeID u) const { return _graph->adjacent_nodes(u); }
  [[nodiscard]] inline auto neighbors(const NodeID u) const { return _graph->neighbors(u); }
  [[nodiscard]] inline std::size_t bucket_size(const std::size_t bucket) const { return _graph->bucket_size(bucket); }
  [[nodiscard]] inline NodeID first_node_in_bucket(const std::size_t bucket) const { return _graph->first_node_in_bucket(bucket); }
  [[nodiscard]] inline NodeID first_invalid_node_in_bucket(const std::size_t bucket) const { return _graph->first_invalid_node_in_bucket(bucket); }
  [[nodiscard]] inline std::size_t number_of_buckets() const { return _graph->number_of_buckets(); }
  [[nodiscard]] inline bool sorted() const { return _graph->sorted(); }
  [[nodiscard]] inline EdgeID edge_cut_to_pe(const PEID pe) const { return _graph->edge_cut_to_pe(pe); }
  [[nodiscard]] inline EdgeID comm_vol_to_pe(const PEID pe) const { return _graph->comm_vol_to_pe(pe); }
  [[nodiscard]] inline MPI_Comm communicator() const { return _graph->communicator(); }
  [[nodiscard]] inline bool permuted() const { return _graph->permuted(); }
  [[nodiscard]] inline NodeID map_original_node(const NodeID u) const { return _graph->map_original_node(u); }
  // clang-format on

  [[nodiscard]] BlockID k() const {
    return _k;
  }

  template <typename Lambda> inline void pfor_blocks(Lambda &&l) const {
    tbb::parallel_for(static_cast<BlockID>(0), k(), std::forward<Lambda>(l));
  }

  [[nodiscard]] inline auto blocks() const {
    return IotaRange<BlockID>(0, k());
  }

  [[nodiscard]] BlockID block(const NodeID u) const {
    KASSERT(u < _partition.size());
    return __atomic_load_n(&_partition[u], __ATOMIC_RELAXED);
  }

  template <bool update_block_weights = true> void set_block(const NodeID u, const BlockID b) {
    KASSERT(u < _graph->total_n());

    if constexpr (update_block_weights) {
      const NodeWeight u_weight = _graph->node_weight(u);
      __atomic_fetch_sub(&_block_weights[block(u)], u_weight, __ATOMIC_RELAXED);
      __atomic_fetch_add(&_block_weights[b], u_weight, __ATOMIC_RELAXED);
    }
    __atomic_store_n(&_partition[u], b, __ATOMIC_RELAXED);
  }

  [[nodiscard]] inline BlockWeight block_weight(const BlockID b) const {
    KASSERT(b < k());
    KASSERT(b < _block_weights.size());
    return __atomic_load_n(&_block_weights[b], __ATOMIC_RELAXED);
  }

  void set_block_weight(const BlockID b, const BlockWeight weight) {
    KASSERT(b < k());
    KASSERT(b < _block_weights.size());
    __atomic_store_n(&_block_weights[b], weight, __ATOMIC_RELAXED);
  }

  [[nodiscard]] const auto &block_weights() const {
    return _block_weights;
  }

  [[nodiscard]] auto &&take_block_weights() {
    return std::move(_block_weights);
  }

  [[nodiscard]] const auto &partition() const {
    return _partition;
  }
  [[nodiscard]] auto &&take_partition() {
    return std::move(_partition);
  }

  void reinit_block_weights() {
    init_block_weights();
  }

  [[nodiscard]] inline bool check_border_node(const NodeID u) const {
    const BlockID u_block = block(u);
    return std::any_of(adjacent_nodes(u).begin(), adjacent_nodes(u).end(), [&](const NodeID v) {
      return u_block != block(v);
    });
  }

private:
  void init_block_weights();

  const DistributedGraph *_graph;

  BlockID _k;
  StaticArray<BlockID> _partition;
  StaticArray<BlockWeight> _block_weights;
};

namespace debug {
/**
 * Validates the distributed graph partition:
 * - check the block assignment of interface and ghost nodes
 * - check the block weights
 *
 * @param p_graph the graph partition to check.
 * @return whether the graph partition is consistent.
 */
bool validate_partition(const DistributedPartitionedGraph &p_graph);
} // namespace debug
} // namespace kaminpar::dist
