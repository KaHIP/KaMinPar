/*******************************************************************************
 * Data structure for clusters and their connections.
 *
 * @file:   clusters.h
 * @author: Daniel Seemaier
 * @date:   19.07.2023
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"

#include "kaminpar-common/datastructures/fast_reset_array.h"
#include "kaminpar-common/ranges.h"

namespace kaminpar::dist {
struct ClustersMemoryContext {
  //! Maps a node ID to its move set ID.
  NoinitVector<NodeID> node_to_cluster;

  NoinitVector<NodeID> clusters;
  NoinitVector<NodeID> cluster_indices;

  //! Weight of move sets to adjacent blocks.
  NoinitVector<EdgeWeight> cluster_conns;

  void clear() {
    node_to_cluster.clear();
    clusters.clear();
    cluster_indices.clear();
    cluster_conns.clear();
  }

  void resize(const DistributedPartitionedGraph &p_graph) {
    resize(p_graph.n(), p_graph.k());
  }

private:
  void resize(const NodeID n, const BlockID k) {
    if (node_to_cluster.size() < n) {
      node_to_cluster.resize(n);
    }
    if (clusters.size() < n) {
      clusters.resize(n);
    }
    if (cluster_indices.size() < n + 1) {
      cluster_indices.resize(n + 1);
    }
    if (cluster_conns.size() < n * k) {
      cluster_conns.resize(n * k);
    }
  }
};

class Clusters {
public:
  Clusters() = default;

  Clusters(
      const DistributedPartitionedGraph &p_graph,
      const PartitionContext &p_ctx,
      ClustersMemoryContext m_ctx
  );

  Clusters(
      const DistributedPartitionedGraph &p_graph,
      const PartitionContext &p_ctx,
      NoinitVector<NodeID> node_to_cluster,
      NoinitVector<NodeID> clusters,
      NoinitVector<NodeID> cluster_indices,
      NoinitVector<EdgeWeight> cluster_conns
  );

  Clusters(const Clusters &) = delete;
  Clusters &operator=(const Clusters &) = delete;

  Clusters(Clusters &&) noexcept = default;
  Clusters &operator=(Clusters &&) noexcept = default;

  operator ClustersMemoryContext() &&;

  [[nodiscard]] inline NodeID size(const NodeID cluster) const {
    KASSERT(cluster + 1 < _cluster_indices.size());
    return _cluster_indices[cluster + 1] - _cluster_indices[cluster];
  }

  [[nodiscard]] inline auto clusters() const {
    return IotaRange<NodeID>(0, num_clusters());
  }

  [[nodiscard]] inline auto nodes(const NodeID cluster) const {
    KASSERT(cluster < num_clusters());
    return TransformedIotaRange(
        _cluster_indices[cluster],
        _cluster_indices[cluster + 1],
        [this](const NodeID i) { return _clusters[i]; }
    );
  }

  [[nodiscard]] inline EdgeWeight gain(const NodeID cluster, const BlockID to_block) const {
    return conn(cluster, to_block) - conn(cluster, block(cluster));
  }

  [[nodiscard]] inline double relative_gain(const NodeID cluster, const BlockID to_block) const {
    return compute_relative_gain(gain(cluster, to_block), weight(cluster));
  }

  [[nodiscard]] inline EdgeWeight conn(const NodeID cluster, const BlockID to_block) const {
    return _cluster_conns[cluster * _p_graph->k() + to_block];
  }

  [[nodiscard]] inline BlockID block(const NodeID cluster) const {
    return _p_graph->block(_clusters[_cluster_indices[cluster]]);
  }

  [[nodiscard]] inline NodeWeight weight(const NodeID cluster) const {
    NodeWeight weight = 0;
    for (const NodeID u : nodes(cluster)) {
      weight += _p_graph->node_weight(u);
    }
    return weight;
  }

  [[nodiscard]] inline NodeID num_clusters() const {
    return _cluster_indices.size() - 1;
  }

  template <typename AdjacentClusterHandler>
  inline void move_ghost_node(
      const NodeID ghost, const BlockID from, const BlockID to, AdjacentClusterHandler &&handler
  ) {
    KASSERT(from != to);
    KASSERT(_p_graph->is_ghost_node(ghost));
    const NodeID nth_ghost = ghost - _p_graph->n();

    KASSERT(nth_ghost + 1 < _ghost_node_indices.size());
    for (EdgeID edge = _ghost_node_indices[nth_ghost]; edge < _ghost_node_indices[nth_ghost + 1];
         ++edge) {
      KASSERT(edge < _ghost_node_edges.size());
      const auto [cluster, weight] = _ghost_node_edges[edge];

      KASSERT((cluster + 1) * _p_graph->k() <= _cluster_conns.size());
      _cluster_conns[cluster * _p_graph->k() + from] -= weight;
      _cluster_conns[cluster * _p_graph->k() + to] += weight;

      handler(cluster);
    }
  }

  inline NodeID cluster_of(const NodeID node) const {
    KASSERT(node < _node_to_cluster.size());
    return _node_to_cluster[node];
  }

  inline bool contains(const NodeID node) const {
    return cluster_of(node) != kInvalidNodeID;
  }

  inline void move_cluster(const NodeID set, const BlockID from, const BlockID to) {
    for (const NodeID u : nodes(set)) {
      KASSERT(_p_graph->is_owned_node(u));

      for (const auto [e, v] : _p_graph->neighbors(u)) {
        if (!_p_graph->is_owned_node(v)) {
          continue;
        }

        const NodeID set_v = _node_to_cluster[v];
        if (set_v == kInvalidNodeID || set_v == set) {
          continue;
        }

        const EdgeWeight delta = _p_graph->edge_weight(e);
        _cluster_conns[set_v * _p_graph->k() + from] -= delta;
        _cluster_conns[set_v * _p_graph->k() + to] += delta;
      }
    }
  }

  inline std::pair<EdgeWeight, BlockID> find_max_conn(const NodeID cluster) const {
    KASSERT(size(cluster) > 0);

    EdgeWeight max_conn = std::numeric_limits<EdgeWeight>::min();
    BlockID max_gainer = kInvalidBlockID;

    const BlockID set_b = block(cluster);
    for (const BlockID b : _p_graph->blocks()) {
      if (b != set_b && conn(cluster, b) > max_conn &&
          _p_graph->block_weight(b) + weight(cluster) <= _p_ctx->graph->max_block_weight(b)) {
        max_conn = conn(cluster, b);
        max_gainer = b;
      }
    }

    KASSERT(max_conn >= 0);
    KASSERT(max_gainer != kInvalidBlockID);

    return {max_conn, max_gainer};
  }

  inline std::pair<EdgeWeight, BlockID> find_max_gain(const NodeID cluster) const {
    const auto [max_conn, max_gainer] = find_max_conn(cluster);
    return {max_conn - conn(cluster, block(cluster)), max_gainer};
  }

  inline std::pair<double, BlockID> find_max_relative_gain(const NodeID set) const {
    const auto [absolute_gain, max_gainer] = find_max_gain(set);
    return {compute_relative_gain(absolute_gain, weight(set)), max_gainer};
  }

  bool dbg_check_all_nodes_covered() const;
  bool dbg_check_clusters_contained_in_blocks() const;
  bool dbg_check_conns() const;
  bool dbg_check_conns(const NodeID cluster) const;

private:
  double
  compute_relative_gain(const EdgeWeight absolute_gain, const NodeWeight cluster_weight) const {
    if (absolute_gain >= 0) {
      return absolute_gain * cluster_weight;
    } else {
      return 1.0 * absolute_gain / cluster_weight;
    }
  }

  void init_ghost_node_adjacency();

  const DistributedPartitionedGraph *_p_graph;
  const PartitionContext *_p_ctx;

  NoinitVector<NodeID> _node_to_cluster;
  NoinitVector<NodeID> _clusters;
  NoinitVector<NodeID> _cluster_indices;
  NoinitVector<EdgeWeight> _cluster_conns;

  NoinitVector<NodeID> _ghost_node_indices;
  NoinitVector<std::pair<NodeID, EdgeWeight>> _ghost_node_edges;
};

Clusters build_clusters(
    ClusterStrategy strategy,
    const DistributedPartitionedGraph &p_graph,
    const Context &ctx,
    const PartitionContext &p_ctx,
    NodeWeight max_move_cluster_size,
    ClustersMemoryContext m_ctx
);
} // namespace kaminpar::dist
