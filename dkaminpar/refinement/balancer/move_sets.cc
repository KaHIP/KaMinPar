#include "dkaminpar/refinement/balancer/move_sets.h"

#include <csignal>

#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "dkaminpar/coarsening/clustering/clusterer.h"
#include "dkaminpar/context.h"
#include "dkaminpar/factories.h"

#include "kaminpar/refinement/stopping_policies.h"

#include "common/assertion_levels.h"
#include "common/datastructures/binary_heap.h"
#include "common/datastructures/fast_reset_array.h"
#include "common/datastructures/marker.h"
#include "common/datastructures/noinit_vector.h"
#include "common/datastructures/rating_map.h"
#include "common/timer.h"

namespace kaminpar::dist {
SET_DEBUG(true);

MoveSets::MoveSets(const DistributedPartitionedGraph &p_graph, MoveSetsMemoryContext m_ctx)
    : MoveSets(
          p_graph,
          std::move(m_ctx.node_to_move_set),
          std::move(m_ctx.move_sets),
          std::move(m_ctx.move_set_indices),
          std::move(m_ctx.move_set_conns)
      ) {
  KASSERT(
      dbg_check_all_nodes_covered(),
      "not all nodes in overloaded blocks are covered by move sets",
      assert::heavy
  );
}

MoveSets::MoveSets(
    const DistributedPartitionedGraph &p_graph,
    NoinitVector<NodeID> node_to_move_set,
    NoinitVector<NodeID> move_sets,
    NoinitVector<NodeID> move_set_indices,
    NoinitVector<EdgeWeight> move_set_conns

)
    : _p_graph(&p_graph),
      _node_to_move_set(std::move(node_to_move_set)),
      _move_sets(std::move(move_sets)),
      _move_set_indices(std::move(move_set_indices)),
      _move_set_conns(std::move(move_set_conns)) {
  KASSERT(_move_set_indices.front() == 0u);
  init_ghost_node_adjacency();

  KASSERT(
      dbg_check_all_nodes_covered(),
      "not all nodes in overloaded blocks are covered by move sets",
      assert::heavy
  );
}

MoveSets::operator MoveSetsMemoryContext() && {
  return {
      std::move(_node_to_move_set),
      std::move(_move_sets),
      std::move(_move_set_indices),
      std::move(_move_set_conns),
  };
}

void MoveSets::init_ghost_node_adjacency() {
  std::vector<std::tuple<NodeID, EdgeWeight, NodeID>> ghost_to_set;
  FastResetArray<EdgeWeight> weight_to_ghost(_p_graph->ghost_n());

  for (const NodeID set : sets()) {
    for (const NodeID u : elements(set)) {
      for (const auto [e, v] : _p_graph->neighbors(u)) {
        if (!_p_graph->is_ghost_node(v)) {
          continue;
        }

        weight_to_ghost[v - _p_graph->n()] += _p_graph->edge_weight(e);
      }
    }

    for (const auto &[ghost, weight] : weight_to_ghost.entries()) {
      ghost_to_set.emplace_back(ghost, weight, set);
    }
    weight_to_ghost.clear();
  }

  std::sort(ghost_to_set.begin(), ghost_to_set.end(), [](const auto &a, const auto &b) {
    return std::get<0>(a) < std::get<0>(b);
  });

  _ghost_node_indices.resize(_p_graph->ghost_n() + 1);
  _ghost_node_edges.resize(ghost_to_set.size());

  NodeID prev_ghost = 0;
  for (std::size_t i = 0; i < ghost_to_set.size(); ++i) {
    const auto [ghost, weight, set] = ghost_to_set[i];
    for (; prev_ghost < ghost; ++prev_ghost) {
      _ghost_node_indices[prev_ghost] = _ghost_node_indices[prev_ghost];
    }
    _ghost_node_edges.emplace_back(weight, set);
  }
  for (; prev_ghost < _p_graph->ghost_n() + 1; ++prev_ghost) {
    _ghost_node_indices[prev_ghost] = _ghost_node_indices[prev_ghost];
  }
}

bool MoveSets::dbg_check_all_nodes_covered() const {
  std::vector<bool> covered(_p_graph->n());
  BlockWeight min_block_weight_covered = std::numeric_limits<BlockWeight>::max();

  for (const NodeID set : sets()) {
    for (const NodeID node : elements(set)) {
      if (block(set) != _p_graph->block(node)) {
        LOG_ERROR << "block of node " << node << " = " << _p_graph->block(node)
                  << " is inconsistent with the move set block " << block(set);
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
      LOG_ERROR << "node " << node << " should be covered by a move set, but is not";
      return false;
    }

    if (set_of(node) == kInvalidNodeID) {
      LOG_ERROR << "node " << node
                << " is covered by some move set, but set_of() returns an invalid value";
      return false;
    }
  }

  LOG_SUCCESS << "All nodes in blocks with weight up to " << min_block_weight_covered
              << " are covered by move sets";

  return true;
}

namespace {
class MoveSetBuilder {
public:
  MoveSetBuilder(
      const DistributedPartitionedGraph &p_graph,
      const PartitionContext &p_ctx,
      MoveSetsMemoryContext m_ctx
  )
      : _p_graph(p_graph),
        _p_ctx(p_ctx),
        _node_to_move_set(std::move(m_ctx.node_to_move_set)),
        _move_sets(std::move(m_ctx.move_sets)),
        _move_set_indices(std::move(m_ctx.move_set_indices)),
        _conns(std::move(m_ctx.move_set_conns)),
        _frontier(0),
        _cur_conns(0),
        _stopping_policy(1.0) {
    _stopping_policy.init(_p_graph.n());
    allocate();
  }

  void allocate() {
    if (_node_to_move_set.size() < _p_graph.n()) {
      _node_to_move_set.resize(_p_graph.n());
    }
    if (_move_sets.size() < _p_graph.n()) {
      _move_sets.resize(_p_graph.n());
    }
    if (_move_set_indices.size() < _p_graph.n() + 1) {
      _move_set_indices.resize(_p_graph.n() + 1);
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
      _node_to_move_set[u] = kInvalidNodeID;
      _move_sets[u] = kInvalidNodeID;
    });
    _move_set_indices.front() = 0;
  }

  void build(const NodeWeight max_move_set_weight) {
    reset_cur_conns();

    for (const NodeID u : _p_graph.nodes()) {
      const BlockID bu = _p_graph.block(u);
      if (_p_graph.block_weight(bu) > _p_ctx.graph->max_block_weight(bu) &&
          _node_to_move_set[u] == kInvalidNodeID) {
        grow_move_set(u, max_move_set_weight);
      }
    }
  }

  void grow_move_set(const NodeID seed, const NodeWeight max_weight) {
    KASSERT(_node_to_move_set[seed] == kInvalidNodeID);

    _frontier.push(seed, 0);
    while (!_frontier.empty() && _cur_weight < max_weight && !_stopping_policy.should_stop()) {
      const NodeID u = _frontier.peek_id();
      const BlockID bu = _p_graph.block(u);
      _frontier.pop();

      add_to_move_set(u);

      for (const auto [e, v] : _p_graph.neighbors(u)) {
        if (_p_graph.contains_local_node(v) && _node_to_move_set[v] == kInvalidBlockID &&
            _p_graph.block(v) == bu) {
          if (_frontier.contains(v)) {
            _frontier.decrease_priority(v, _frontier.key(v) + _p_graph.edge_weight(e));
          } else {
            _frontier.push(v, _p_graph.edge_weight(e));
          }
        }
      }
    }

    finish_move_set();

    KASSERT(_node_to_move_set[seed] != kInvalidBlockID, "unassigned seed node " << seed);
  }

  void add_to_move_set(const NodeID u) {
    KASSERT(_cur_block == kInvalidBlockID || _cur_block == _p_graph.block(u));

    if (_cur_block == kInvalidBlockID) {
      _cur_block = _p_graph.block(u);
    }

    _cur_weight += _p_graph.node_weight(u);
    _node_to_move_set[u] = _cur_move_set;
    _move_sets[_cur_pos] = u;
    ++_cur_pos;

    for (const auto [e, v] : _p_graph.neighbors(u)) {
      if (_p_graph.is_owned_node(v) && _node_to_move_set[v] == _cur_move_set) {
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

  void finish_move_set() {
    for (NodeID pos = _best_prefix_pos + 1; pos < _cur_pos; ++pos) {
      _node_to_move_set[_move_sets[pos]] = kInvalidNodeID;
    }

    _move_set_indices[++_cur_move_set] = _best_prefix_pos;
    KASSERT(_move_set_indices[_cur_move_set] - _move_set_indices[_cur_move_set - 1] <= 64);

    reset_cur_conns();
    _cur_block = kInvalidBlockID;
    _cur_block_conn = 0;
    _cur_pos = _best_prefix_pos;
    _cur_weight = 0;

    _best_prefix_block = kInvalidBlockID;
    _best_prefix_conn = 0;
    // _best_prefix_pos = _cur_pos;

    _frontier.clear();
    _stopping_policy.reset();
  }

  MoveSets finalize() {
    _move_set_indices.resize(_cur_move_set + 1);
    KASSERT(_move_set_indices.front() == 0);

    KASSERT([&] {
      for (NodeID set = 1; set < _move_set_indices.size(); ++set) {
        if (_move_set_indices[set] < _move_set_indices[set - 1]) {
          LOG_WARNING << "bad set " << set - 1 << ": spans from " << _move_set_indices[set - 1]
                      << " to " << _move_set_indices[set];
          return false;
        }
      }
      return true;
    }());

    return {
        _p_graph,
        std::move(_node_to_move_set),
        std::move(_move_sets),
        std::move(_move_set_indices),
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

  NoinitVector<NodeID> _node_to_move_set;
  NoinitVector<NodeID> _move_sets;
  NoinitVector<NodeID> _move_set_indices;

  NoinitVector<EdgeWeight> _conns;

  BinaryMaxHeap<EdgeWeight> _frontier;

  NodeID _cur_pos = 0;
  NodeID _cur_move_set = 0;
  EdgeWeight _cur_block_conn = 0;
  BinaryMaxHeap<EdgeWeight> _cur_conns;
  BlockID _cur_block = kInvalidBlockID;
  NodeWeight _cur_weight = 0;

  NodeID _best_prefix_pos = 0;
  BlockID _best_prefix_block = kInvalidBlockID;
  EdgeWeight _best_prefix_conn = 0;

  shm::AdaptiveStoppingPolicy _stopping_policy;
};

MoveSets build_singleton_move_sets(
    const DistributedPartitionedGraph &p_graph,
    const PartitionContext &p_ctx,
    const NodeWeight max_weight,
    MoveSetsMemoryContext m_ctx
) {
  m_ctx.clear();

  NodeID cur_move_set = 0;
  for (const NodeID u : p_graph.nodes()) {
    const BlockID bu = p_graph.block(u);

    if (p_graph.block_weight(bu) > p_ctx.graph->max_block_weight(bu)) {
      m_ctx.node_to_move_set.push_back(cur_move_set);
      m_ctx.move_set_indices.push_back(cur_move_set);
      m_ctx.move_set_conns.resize((cur_move_set + 1) * p_graph.k());
      m_ctx.move_sets.push_back(u);

      for (const BlockID k : p_graph.blocks()) {
        m_ctx.move_set_conns[cur_move_set * p_graph.k() + k] = 0;
      }
      for (const auto [e, v] : p_graph.neighbors(u)) {
        const BlockID bv = p_graph.block(v);
        m_ctx.move_set_conns[cur_move_set * p_graph.k() + bv] += p_graph.edge_weight(e);
      }

      ++cur_move_set;
    }
  }
  m_ctx.move_set_conns.push_back(cur_move_set);

  return {p_graph, std::move(m_ctx)};
}

MoveSets build_clustered_move_sets(
    const DistributedPartitionedGraph &p_graph,
    const PartitionContext &p_ctx,
    const NodeWeight max_weight,
    std::unique_ptr<LocalClusterer> clusterer,
    MoveSetsMemoryContext m_ctx
) {
  clusterer->initialize(p_graph.graph());
  auto &clustering = clusterer->cluster(p_graph, max_weight);

  std::vector<NodeID> cluster_to_move_set(p_graph.n());
  std::vector<NodeID> cluster_sizes(p_graph.n());
  for (const NodeID u : p_graph.nodes()) {
    const BlockID bu = p_graph.block(u);
    if (p_graph.block_weight(bu) > p_ctx.graph->max_block_weight(bu)) {
      cluster_to_move_set[clustering[u]] = 1;
      cluster_sizes[clustering[u]]++;
    }
  }
  parallel::prefix_sum(cluster_to_move_set.begin(), cluster_to_move_set.end(), cluster_to_move_set.begin());
  parallel::prefix_sum(cluster_sizes.begin(), cluster_sizes.end(), cluster_sizes.begin());

  m_ctx.resize(p_graph);
  m_ctx.move_set_indices[0] = 0;
  std::fill(m_ctx.move_set_conns.begin(), m_ctx.move_set_conns.end(), 0);

  for (const NodeID u : p_graph.nodes()) {
    const BlockID bu = p_graph.block(u);
    if (p_graph.block_weight(bu) > p_ctx.graph->max_block_weight(bu)) {
      const NodeID ms = cluster_to_move_set[clustering[u]] - 1;

      m_ctx.node_to_move_set[u] = ms;
      m_ctx.move_sets[cluster_sizes[clustering[u]]++] = u;
      m_ctx.move_set_indices[ms + 1] = cluster_sizes[clustering[u]];

      for (const auto [e, v] : p_graph.neighbors(u)) {
        const BlockID bv = p_graph.block(v);
        m_ctx.move_set_conns[ms * p_graph.k() + bv] += p_graph.edge_weight(e);
      }
    }
  }

  return {p_graph, std::move(m_ctx)};
}
} // namespace

MoveSets build_move_sets(
    const MoveSetStrategy strategy,
    const DistributedPartitionedGraph &p_graph,
    const Context &ctx,
    const PartitionContext &p_ctx,
    const NodeWeight max_move_set_weight,
    MoveSetsMemoryContext m_ctx

) {
  SCOPED_TIMER("Build move sets");

  switch (strategy) {
  case MoveSetStrategy::SINGLETONS:
    return build_singleton_move_sets(p_graph, p_ctx, max_move_set_weight, std::move(m_ctx));

  case MoveSetStrategy::LP:
    return build_clustered_move_sets(
        p_graph,
        p_ctx,
        max_move_set_weight,
        factory::create_local_clusterer(ctx, LocalClusteringAlgorithm::LP),
        std::move(m_ctx)
    );

  case MoveSetStrategy::GREEDY_BATCH_PREFIX:
    MoveSetBuilder builder(p_graph, p_ctx, std::move(m_ctx));
    builder.build(max_move_set_weight);
    return builder.finalize();
  }

  __builtin_unreachable();
}
} // namespace kaminpar::dist

