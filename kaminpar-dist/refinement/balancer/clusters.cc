/*******************************************************************************
 * Data structure for clusters and their connections.
 *
 * @file:   clusters.cc
 * @author: Daniel Seemaier
 * @date:   19.07.2023
 ******************************************************************************/
#include "kaminpar-dist/refinement/balancer/clusters.h"

#include <csignal>

#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "kaminpar-dist/coarsening/clustering/clusterer.h"
#include "kaminpar-dist/context.h"
#include "kaminpar-dist/factories.h"

#include "kaminpar-shm/refinement/fm/stopping_policies.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/binary_heap.h"
#include "kaminpar-common/datastructures/fast_reset_array.h"
#include "kaminpar-common/datastructures/marker.h"
#include "kaminpar-common/datastructures/noinit_vector.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/timer.h"

#define HEAVY assert::heavy

namespace kaminpar::dist {
SET_DEBUG(false);

Clusters::Clusters(
    const DistributedPartitionedGraph &p_graph,
    const PartitionContext &p_ctx,
    ClustersMemoryContext m_ctx
)
    : Clusters(
          p_graph,
          p_ctx,
          std::move(m_ctx.node_to_cluster),
          std::move(m_ctx.clusters),
          std::move(m_ctx.cluster_indices),
          std::move(m_ctx.cluster_conns)
      ) {}

Clusters::Clusters(
    const DistributedPartitionedGraph &p_graph,
    const PartitionContext &p_ctx,
    NoinitVector<NodeID> node_to_cluster,
    NoinitVector<NodeID> clusters,
    NoinitVector<NodeID> cluster_indices,
    NoinitVector<EdgeWeight> cluster_conns

)
    : _p_graph(&p_graph),
      _p_ctx(&p_ctx),
      _node_to_cluster(std::move(node_to_cluster)),
      _clusters(std::move(clusters)),
      _cluster_indices(std::move(cluster_indices)),
      _cluster_conns(std::move(cluster_conns)) {
  KASSERT(_cluster_indices.front() == 0u);
  init_ghost_node_adjacency();

  KASSERT(
      dbg_check_all_nodes_covered(),
      "not all nodes in overloaded blocks are covered by clusters",
      HEAVY
  );
  KASSERT(
      dbg_check_clusters_contained_in_blocks(),
      "clusters span multiple blocks, which is not allowed",
      HEAVY
  );
  KASSERT(dbg_check_conns(), "invalid cluster connections", HEAVY);
}

Clusters::operator ClustersMemoryContext() && {
  return {
      std::move(_node_to_cluster),
      std::move(_clusters),
      std::move(_cluster_indices),
      std::move(_cluster_conns),
  };
}

void Clusters::init_ghost_node_adjacency() {
  std::vector<std::tuple<NodeID, EdgeWeight, NodeID>> ghost_to_cluster;
  FastResetArray<EdgeWeight> weight_to_ghost(_p_graph->ghost_n());

  for (const NodeID cluster : clusters()) {
    for (const NodeID u : nodes(cluster)) {
      for (const auto [e, v] : _p_graph->neighbors(u)) {
        if (!_p_graph->is_ghost_node(v)) {
          continue;
        }

        weight_to_ghost[v - _p_graph->n()] += _p_graph->edge_weight(e);
      }
    }

    for (const auto &[ghost, weight] : weight_to_ghost.entries()) {
      ghost_to_cluster.emplace_back(ghost, weight, cluster);
    }
    weight_to_ghost.clear();
  }

  std::sort(ghost_to_cluster.begin(), ghost_to_cluster.end(), [](const auto &a, const auto &b) {
    return std::get<0>(a) < std::get<0>(b);
  });

  _ghost_node_indices.resize(_p_graph->ghost_n() + 1);
  _ghost_node_indices.front() = 0;
  _ghost_node_edges.resize(ghost_to_cluster.size());

  NodeID prev_ghost = 0;
  for (std::size_t i = 0; i < ghost_to_cluster.size(); ++i) {
    const auto [ghost, weight, cluster] = ghost_to_cluster[i];
    for (; prev_ghost < ghost; ++prev_ghost) {
      _ghost_node_indices[prev_ghost + 1] = _ghost_node_edges.size();
    }
    _ghost_node_edges.emplace_back(cluster, weight);
  }
  for (; prev_ghost < _p_graph->ghost_n(); ++prev_ghost) {
    _ghost_node_indices[prev_ghost + 1] = _ghost_node_edges.size();
  }

  KASSERT(
      [&] {
        if (_p_graph->ghost_n() + 1 < _ghost_node_indices.size()) {
          LOG_WARNING << "ghost node indices is too small";
          return false;
        }
        for (std::size_t i = 0; i + 1 < _ghost_node_indices.size(); ++i) {
          if (_ghost_node_indices[i] > _ghost_node_indices[i + 1]) {
            LOG_WARNING << "ghost node indices is not a prefix sum";
            return false;
          }
        }
        if (_ghost_node_indices[_p_graph->ghost_n()] > _ghost_node_edges.size()) {
          LOG_WARNING << "end position of last ghost node is too large";
          return false;
        }
        return true;
      }(),
      "ghost node adjacency array is inconsistent",
      assert::heavy
  );
}

bool Clusters::dbg_check_all_nodes_covered() const {
  std::vector<bool> covered(_p_graph->n());
  BlockWeight min_block_weight_covered = std::numeric_limits<BlockWeight>::max();

  for (const NodeID cluster : clusters()) {
    for (const NodeID node : nodes(cluster)) {
      if (block(cluster) != _p_graph->block(node)) {
        LOG_ERROR << "block of node " << node << " = " << _p_graph->block(node)
                  << " is inconsistent with the cluster block " << block(cluster);
        return false;
      }

      covered[node] = true;
      min_block_weight_covered =
          std::min(min_block_weight_covered, _p_graph->block_weight(_p_graph->block(node)));
    }
  }

  for (const NodeID node : _p_graph->nodes()) {
    if (_p_graph->block_weight(_p_graph->block(node)) < min_block_weight_covered) {
      // Since we don't have the partition context here, we cannot conclude whether this node should
      // be covered or not -- thus skip it
      continue;
    }

    if (!covered[node]) {
      LOG_ERROR << "node " << node << " should be covered by a cluster, but is not";
      return false;
    }

    if (cluster_of(node) == kInvalidNodeID) {
      LOG_ERROR << "node " << node
                << " is covered by some cluster, but cluster_of() returns an invalid value";
      return false;
    }
  }

  return true;
}

bool Clusters::dbg_check_clusters_contained_in_blocks() const {
  for (const NodeID cluster : clusters()) {
    for (const NodeID node : nodes(cluster)) {
      if (_p_graph->block(node) != block(cluster)) {
        LOG_WARNING << "node " << node << " in cluster " << cluster << " is assigned to block "
                    << _p_graph->block(node) << ", but the cluster is assigned to block "
                    << block(cluster);
        return false;
      }
    }
  }
  return true;
}

bool Clusters::dbg_check_conns() const {
  for (const NodeID cluster : clusters()) {
    if (!dbg_check_conns(cluster)) {
      return false;
    }
  }

  return true;
}

bool Clusters::dbg_check_conns(const NodeID cluster) const {
  std::vector<EdgeWeight> actual(_p_graph->k());

  for (const NodeID u : nodes(cluster)) {
    for (const auto &[e, v] : _p_graph->neighbors(u)) {
      if (!_p_graph->is_owned_node(v) || cluster_of(v) != cluster_of(u)) {
        actual[_p_graph->block(v)] += _p_graph->edge_weight(e);
      }
    }
  }

  for (const BlockID b : _p_graph->blocks()) {
    if (actual[b] != conn(cluster, b)) {
      LOG_WARNING << "cluster " << cluster << " in block " << block(cluster)
                  << " has conn to block " << b << " = " << conn(cluster, b)
                  << ", but the actual conn is " << actual[b];
      return false;
    }
  }

  return true;
}

namespace {
class BatchedClusterBuilder {
public:
  BatchedClusterBuilder(
      const DistributedPartitionedGraph &p_graph,
      const PartitionContext &p_ctx,
      ClustersMemoryContext m_ctx
  )
      : _p_graph(p_graph),
        _p_ctx(p_ctx),
        _node_to_cluster(std::move(m_ctx.node_to_cluster)),
        _clusters(std::move(m_ctx.clusters)),
        _cluster_indices(std::move(m_ctx.cluster_indices)),
        _conns(std::move(m_ctx.cluster_conns)),
        _frontier(0),
        _cur_conns(0),
        _stopping_policy(1.0) {
    _stopping_policy.init(_p_graph.n());
    allocate();
  }

  void allocate() {
    if (_node_to_cluster.size() < _p_graph.n()) {
      _node_to_cluster.resize(_p_graph.n());
    }
    if (_clusters.size() < _p_graph.n()) {
      _clusters.resize(_p_graph.n());
    }
    if (_cluster_indices.size() < _p_graph.n() + 1) {
      _cluster_indices.resize(_p_graph.n() + 1);
    }
    if (_conns.size() < _p_graph.n() * _p_graph.k()) {
      _conns.resize(_p_graph.n() * _p_graph.k());
    }
    if (_frontier.size() < _p_graph.n()) {
      _frontier.resize(_p_graph.n());
    }
    if (_cur_conns.size() < _p_graph.k()) {
      _cur_conns.resize(_p_graph.k());
    }

    _p_graph.pfor_nodes([&](const NodeID u) {
      _node_to_cluster[u] = kInvalidNodeID;
      _clusters[u] = kInvalidNodeID;
    });
    _cluster_indices.front() = 0;
  }

  void build(const NodeWeight max_cluster_weight) {
    reset_cur_conns();

    for (const NodeID u : _p_graph.nodes()) {
      const BlockID bu = _p_graph.block(u);
      if (_p_graph.block_weight(bu) > _p_ctx.graph->max_block_weight(bu) &&
          _node_to_cluster[u] == kInvalidNodeID) {
        grow_cluster(u, max_cluster_weight);
      }
    }
  }

  void grow_cluster(const NodeID seed, const NodeWeight max_weight) {
    KASSERT(_node_to_cluster[seed] == kInvalidNodeID);

    _frontier.push(seed, 0);
    while (!_frontier.empty() && _cur_weight < max_weight && !_stopping_policy.should_stop()) {
      const NodeID u = _frontier.peek_id();
      const BlockID bu = _p_graph.block(u);
      _frontier.pop();

      add_to_cluster(u);

      for (const auto [e, v] : _p_graph.neighbors(u)) {
        if (_p_graph.is_owned_node(v) && _node_to_cluster[v] == kInvalidBlockID &&
            _p_graph.block(v) == bu) {
          if (_frontier.contains(v)) {
            _frontier.decrease_priority(v, _frontier.key(v) + _p_graph.edge_weight(e));
          } else {
            _frontier.push(v, _p_graph.edge_weight(e));
          }
        }
      }
    }

    finish_cluster();

    KASSERT(_node_to_cluster[seed] != kInvalidBlockID, "unassigned seed node " << seed);
  }

  void add_to_cluster(const NodeID u) {
    KASSERT(_cur_block == kInvalidBlockID || _cur_block == _p_graph.block(u));

    if (_cur_block == kInvalidBlockID) {
      _cur_block = _p_graph.block(u);
    }

    _cur_weight += _p_graph.node_weight(u);
    _node_to_cluster[u] = _cur_cluster;
    _clusters[_cur_pos] = u;
    ++_cur_pos;

    for (const auto [e, v] : _p_graph.neighbors(u)) {
      if (_p_graph.is_owned_node(v) && _node_to_cluster[v] == _cur_cluster) {
        _cur_block_conn -= _p_graph.edge_weight(e);
      } else {
        const BlockID bv = _p_graph.block(v);
        if (bv == _cur_block) {
          _cur_block_conn += _p_graph.edge_weight(e);
        } else if (_p_graph.block_weight(bv) + _cur_weight <= _p_ctx.graph->max_block_weight(bv)) {
          _cur_conns.change_priority(bv, _cur_conns.key(bv) + _p_graph.edge_weight(e));
        } else if (_cur_conns.key(bv) > 0) { // no longer a viable target
          _cur_conns.change_priority(bv, -1);
        }
      }
    }

    _stopping_policy.update(_cur_conns.peek_key() - _cur_block_conn);

    if (_cur_conns.peek_key() >= _best_prefix_conn) {
      _best_prefix_block = _cur_conns.peek_id();
      _best_prefix_conn = _cur_conns.peek_key();
      _best_prefix_pos = _cur_pos;
    }
  }

  void finish_cluster() {
    for (NodeID pos = _best_prefix_pos + 1; pos < _cur_pos; ++pos) {
      _node_to_cluster[_clusters[pos]] = kInvalidNodeID;
    }
    for (const BlockID block : _p_graph.blocks()) {
      _conns[_cur_cluster * _p_graph.k() + block] = 0;
    }
    // @todo should do this when updating _best_*
    for (NodeID pos = _cluster_indices[_cur_cluster]; pos < _best_prefix_pos; ++pos) {
      const NodeID u = _clusters[pos];
      for (const auto &[e, v] : _p_graph.neighbors(u)) {
        if (_p_graph.is_owned_node(v) && _node_to_cluster[v] == _cur_cluster) {
          continue;
        }
        const BlockID bv = _p_graph.block(v);
        _conns[_cur_cluster * _p_graph.k() + bv] += _p_graph.edge_weight(e);
      }
    }

    _cluster_indices[++_cur_cluster] = _best_prefix_pos;

    reset_cur_conns();
    _cur_block = kInvalidBlockID;
    _cur_block_conn = 0;
    _cur_pos = _best_prefix_pos;
    _cur_weight = 0;

    _best_prefix_block = kInvalidBlockID;
    _best_prefix_conn = 0;

    _frontier.clear();
    _stopping_policy.reset();
  }

  Clusters finalize() {
    _cluster_indices.resize(_cur_cluster + 1);
    KASSERT(_cluster_indices.front() == 0);

    KASSERT([&] {
      for (NodeID cluster = 1; cluster < _cluster_indices.size(); ++cluster) {
        if (_cluster_indices[cluster] < _cluster_indices[cluster - 1]) {
          LOG_WARNING << "bad cluster " << cluster - 1 << ": spans from "
                      << _cluster_indices[cluster - 1] << " to " << _cluster_indices[cluster];
          return false;
        }
      }
      return true;
    }());

    return {
        _p_graph,
        _p_ctx,
        std::move(_node_to_cluster),
        std::move(_clusters),
        std::move(_cluster_indices),
        std::move(_conns),
    };
  }

  NodeWeight current_weight() {
    return _cur_weight;
  }

private:
  void reset_cur_conns() {
    _cur_conns.clear();
    for (const BlockID b : _p_graph.blocks()) {
      _cur_conns.push(b, 0);
    }
  }

  const DistributedPartitionedGraph &_p_graph;
  const PartitionContext &_p_ctx;

  NoinitVector<NodeID> _node_to_cluster;
  NoinitVector<NodeID> _clusters;
  NoinitVector<NodeID> _cluster_indices;

  NoinitVector<EdgeWeight> _conns;

  BinaryMaxHeap<EdgeWeight> _frontier;

  NodeID _cur_pos = 0;
  NodeID _cur_cluster = 0;
  EdgeWeight _cur_block_conn = 0;
  BinaryMaxHeap<EdgeWeight> _cur_conns;
  BlockID _cur_block = kInvalidBlockID;
  NodeWeight _cur_weight = 0;

  NodeID _best_prefix_pos = 0;
  BlockID _best_prefix_block = kInvalidBlockID;
  EdgeWeight _best_prefix_conn = 0;

  shm::AdaptiveStoppingPolicy _stopping_policy;
};

Clusters build_singleton_clusters(
    const DistributedPartitionedGraph &p_graph,
    const PartitionContext &p_ctx,
    const NodeWeight max_weight,
    ClustersMemoryContext m_ctx
) {
  m_ctx.clear();

  NodeID cur_move_set = 0;
  for (const NodeID u : p_graph.nodes()) {
    const BlockID bu = p_graph.block(u);

    if (p_graph.block_weight(bu) > p_ctx.graph->max_block_weight(bu)) {
      m_ctx.node_to_cluster.push_back(cur_move_set);
      m_ctx.cluster_indices.push_back(cur_move_set);
      m_ctx.clusters.push_back(u);

      for (const BlockID k : p_graph.blocks()) {
        m_ctx.cluster_conns.push_back(0);
      }
      for (const auto [e, v] : p_graph.neighbors(u)) {
        const BlockID bv = p_graph.block(v);
        const std::size_t idx = cur_move_set * p_graph.k() + bv;
        KASSERT(idx < m_ctx.cluster_conns.size());
        m_ctx.cluster_conns[idx] += p_graph.edge_weight(e);
      }

      ++cur_move_set;
    } else {
      m_ctx.node_to_cluster.push_back(kInvalidNodeID);
    }
  }
  m_ctx.cluster_indices.push_back(cur_move_set);

  KASSERT(
      [&] {
        for (const NodeID u : p_graph.nodes()) {
          const BlockID bu = p_graph.block(u);
          if (p_graph.block_weight(bu) <= p_ctx.graph->max_block_weight(bu) &&
              m_ctx.node_to_cluster[u] != kInvalidNodeID) {
            LOG_ERROR << "node " << u << " is in block " << bu
                      << ", which is not overloaded, yet assigned to a move set";
            return false;
          }
        }
        return true;
      }(),
      "some sets contain nodes in non-overloaded blocks",
      assert::heavy
  );

  return {p_graph, p_ctx, std::move(m_ctx)};
}

Clusters build_local_clusters(
    const DistributedPartitionedGraph &p_graph,
    const PartitionContext &p_ctx,
    const NodeWeight max_weight,
    std::unique_ptr<LocalClusterer> clusterer,
    ClustersMemoryContext m_ctx
) {
  clusterer->initialize(p_graph.graph());
  auto &clustering = clusterer->cluster(p_graph, max_weight);

  std::vector<NodeID> cluster_to_move_set(p_graph.n());
  std::vector<NodeID> cluster_sizes(p_graph.n());
  for (const NodeID u : p_graph.nodes()) {
    const BlockID bu = p_graph.block(u);
    if (p_graph.block_weight(bu) > p_ctx.graph->max_block_weight(bu)) {
      KASSERT(clustering[u] < p_graph.n());
      cluster_to_move_set[clustering[u]] = 1;
      cluster_sizes[clustering[u]]++;
    }
  }
  parallel::prefix_sum(
      cluster_to_move_set.begin(), cluster_to_move_set.end(), cluster_to_move_set.begin()
  );
  std::exclusive_scan(cluster_sizes.begin(), cluster_sizes.end(), cluster_sizes.begin(), 0);

  m_ctx.clear();
  m_ctx.resize(p_graph);
  m_ctx.cluster_indices.front() = 0;
  std::fill(m_ctx.cluster_conns.begin(), m_ctx.cluster_conns.end(), 0);

  for (const NodeID u : p_graph.nodes()) {
    const BlockID bu = p_graph.block(u);
    if (p_graph.block_weight(bu) > p_ctx.graph->max_block_weight(bu)) {
      const NodeID ms = cluster_to_move_set[clustering[u]] - 1;

      m_ctx.node_to_cluster[u] = ms;
      m_ctx.clusters[cluster_sizes[clustering[u]]++] = u;
      m_ctx.cluster_indices[ms + 1] = cluster_sizes[clustering[u]];

      for (const auto [e, v] : p_graph.neighbors(u)) {
        // We may not access clustering[.] for ghost vertices
        if (!p_graph.is_owned_node(v) || clustering[v] != clustering[u]) {
          const BlockID bv = p_graph.block(v);
          m_ctx.cluster_conns[ms * p_graph.k() + bv] += p_graph.edge_weight(e);
        }
      }
    } else {
      m_ctx.node_to_cluster[u] = kInvalidNodeID;
    }
  }
  m_ctx.cluster_indices.resize(cluster_to_move_set.back() + 1);

  KASSERT(
      [&] {
        for (const NodeID u : p_graph.nodes()) {
          const BlockID bu = p_graph.block(u);
          if (p_graph.block_weight(bu) <= p_ctx.graph->max_block_weight(bu) &&
              m_ctx.node_to_cluster[u] != kInvalidNodeID) {
            LOG_ERROR << "node " << u << " is in block " << bu
                      << ", which is not overloaded, yet assigned to a move set";
            return false;
          }
        }
        return true;
      }(),
      "some sets contain nodes in non-overloaded blocks",
      assert::heavy
  );

  return {p_graph, p_ctx, std::move(m_ctx)};
}
} // namespace

Clusters build_clusters(
    const ClusterStrategy strategy,
    const DistributedPartitionedGraph &p_graph,
    const Context &ctx,
    const PartitionContext &p_ctx,
    const NodeWeight max_move_set_weight,
    ClustersMemoryContext m_ctx

) {
  SCOPED_TIMER("Build move sets");

  switch (strategy) {
  case ClusterStrategy::SINGLETONS:
    return build_singleton_clusters(p_graph, p_ctx, max_move_set_weight, std::move(m_ctx));

  case ClusterStrategy::LP:
    return build_local_clusters(
        p_graph,
        p_ctx,
        max_move_set_weight,
        factory::create_local_clusterer(ctx, LocalClusteringAlgorithm::LP),
        std::move(m_ctx)
    );

  case ClusterStrategy::GREEDY_BATCH_PREFIX:
    BatchedClusterBuilder builder(p_graph, p_ctx, std::move(m_ctx));
    builder.build(max_move_set_weight);
    return builder.finalize();
  }

  __builtin_unreachable();
}
} // namespace kaminpar::dist
