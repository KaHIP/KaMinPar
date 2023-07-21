/*******************************************************************************
 * Distributed FM refiner.
 *
 * @file:   local_fm_refiner.cc
 * @author: Daniel Seemaier
 * @date:   02.08.2022
 ******************************************************************************/
#include "dkaminpar/refinement/fm/local_fm_refiner.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <stack>

#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/datastructures/growt.h"
#include "dkaminpar/graphutils/synchronization.h"
#include "dkaminpar/metrics.h"
#include "dkaminpar/mpi/alltoall.h"
#include "dkaminpar/mpi/utils.h"

#include "kaminpar/datastructures/graph.h"

#include "common/datastructures/binary_heap.h"
#include "common/datastructures/marker.h"
#include "common/datastructures/rating_map.h"
#include "common/logger.h"
#include "common/math.h"
#include "common/datastructures/noinit_vector.h"
#include "common/parallel/atomic.h"
#include "common/random.h"
#include "common/timer.h"

namespace kaminpar::dist {
SET_DEBUG(true);
LocalFMRefinerFactory::LocalFMRefinerFactory(const Context &ctx) : _ctx(ctx) {}

std::unique_ptr<GlobalRefiner>
LocalFMRefinerFactory::create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  return std::make_unique<LocalFMRefiner>(_ctx, p_graph, p_ctx);
}

void LocalFMRefiner::Statistics::print() const {
  const auto [n_min, n_mean, n_max] = math::find_min_mean_max(graphs_n);
  const auto [b_min, b_mean, b_max] = math::find_min_mean_max(graphs_border_n);
  const auto [m_min, m_mean, m_max] = math::find_min_mean_max(graphs_m);

  LOG_STATS << "Distributed FM refiner:";
  LOG_STATS << "  * Search graphs: #=" << graphs_n.size();
  LOG_STATS << "    | Number of all nodes:    min=" << n_min << ", mean=" << n_mean
            << ", max=" << n_max;
  LOG_STATS << "    | Number of border nodes: min=" << b_min << ", mean=" << b_mean
            << ", max=" << b_max;
  LOG_STATS << "    | Number of edges:        min=" << m_min << ", mean=" << m_mean
            << ", max=" << m_max;
  LOG_STATS << "  * Improvement: from=" << initial_cut << ", to=" << final_cut
            << ", by=" << initial_cut - final_cut;
  LOG_STATS << "    | Number of searches with positive gain: " << num_searches_with_improvement;
  LOG_STATS << "    | Average gain improvement per search:   "
            << 1.0 * (initial_cut - final_cut) / num_searches_with_improvement;
  LOG_STATS << "    | Number of conflicts: @todo";
}

namespace {
struct AdaptiveStoppingPolicy {
  AdaptiveStoppingPolicy(const GlobalNodeID n) : _beta(std::log(n)) {}

  [[nodiscard]] bool should_stop(const double alpha) const {
    const double factor = alpha / 2.0 - 0.25;
    return (_num_steps > _beta) &&
           ((_mk == 0.0) || (_num_steps >= (_variance / (_mk * _mk)) * factor));
  }

  void reset() {
    _num_steps = 0;
    _variance = 0.0;
  }

  void update(const EdgeWeight gain) {
    ++_num_steps;

    // see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    if (_num_steps == 1) {
      _mk_minus_one = 1.0 * gain;
      _mk = _mk_minus_one;
      _sk_minus_one = 0.0;
    } else {
      _mk = _mk_minus_one + (gain - _mk_minus_one) / _num_steps;
      _sk = _sk_minus_one + (gain - _mk_minus_one) * (gain - _mk);
      _variance = _sk / (_num_steps - 1.0);
      _mk_minus_one = _mk;
      _sk_minus_one = _sk;
    }
  }

private:
  double _beta{0.0};
  std::size_t _num_steps{0};
  double _variance{0.0};
  double _mk{0.0};
  double _mk_minus_one{0.0};
  double _sk{0.0};
  double _sk_minus_one{0.0};
};

class LocalLocalFMRefiner {
public:
  struct Move {
    Move(const NodeID node, const BlockID block) : node(node), block(block) {}

    NodeID node;
    BlockID block;
  };

  LocalLocalFMRefiner(
      const DistributedPartitionedGraph &global_graph,
      const FMRefinementContext &fm_ctx,
      const PartitionContext &p_ctx
  )
      : _global_graph(global_graph),
        _fm_ctx(fm_ctx),
        _p_ctx(p_ctx),
        _rating_map(_p_ctx.k),
        _stopping_policy(p_ctx.graph->global_n) {}

  std::vector<Move> refine(shm::PartitionedGraph &p_graph, const std::vector<bool> &fixed_nodes) {
    if (p_graph.n() == 0) {
      return {};
    }

    initialize(p_graph, fixed_nodes);

    // record of FM nodes for rollback
    EdgeWeight best_total_gain = 0;
    EdgeWeight current_total_gain = 0;
    std::size_t rollback_index = 0;
    std::vector<Move> moves;

    while (!_pq.empty() && !_stopping_policy.should_stop(_fm_ctx.alpha)) {
      // retrieve next node from PQ
      const NodeID u = _pq.peek_id();
      const NodeWeight weight = _p_graph->node_weight(u);
      const BlockID from = _p_graph->block(u);
      const auto [gain, to] = find_best_target_block<false>(u);
      _pq.pop();

      // only perform move if target block can take u without becoming
      // overloaded
      const bool feasible =
          to != from && _global_graph.block_weight(to) + _block_weight_deltas[to] + weight <=
                            _p_ctx.graph->max_block_weight(to);

      if (feasible) {
        // move u to its target block
        _p_graph->set_block<false>(u, to);
        _block_weight_deltas[from] -= weight;
        _block_weight_deltas[to] += weight;
        moves.emplace_back(u, from);
        update_pq_after_move(u, from, to, fixed_nodes);
        current_total_gain += gain;
        _stopping_policy.update(gain);

        if (current_total_gain > best_total_gain) {
          best_total_gain = current_total_gain;
          rollback_index = moves.size();
          _stopping_policy.reset();
        }
      } else {
        // target block became too full since PQ initialization
        // --> retry with new target block
        insert_node_into_pq<false>(u);
      }
    }

    // rollback to best cut found
    while (moves.size() > rollback_index) {
      const auto [node, to] = moves.back();
      const BlockID from = _p_graph->block(node);
      const NodeWeight weight = _p_graph->node_weight(node);

      _block_weight_deltas[from] -= weight;
      _block_weight_deltas[to] += weight;
      _p_graph->set_block<false>(node, to);

      moves.pop_back();
    }

    // for the moves that we keep, store the new block rather than the old one
    for (auto &[u, block] : moves) {
      block = _p_graph->block(u);
    }

    // DBG << "Improved local cut by " << best_total_gain;
    return moves;
  }

  void commit_block_weight_deltas(DistributedPartitionedGraph &p_graph) {
    for (const auto &[block, delta] : _block_weight_deltas) {
      p_graph.set_block_weight(block, p_graph.block_weight(block) + delta);
    }
  }

private:
  void initialize(shm::PartitionedGraph &p_graph, const std::vector<bool> fixed_nodes) {
    _p_graph = &p_graph;

    // resize data structures s.t. they are large enough for _p_graph
    if (_pq.capacity() < _p_graph->n()) {
      _pq.clear();
      _pq.resize(_p_graph->n());
      _marker.resize(_p_graph->n());
    }

    // clear data structures from previous run
    _marker.reset();
    _pq.clear();
    _stopping_policy.reset();
    _block_weight_deltas.clear();

    // fill PQ with all border nodes
    for (const NodeID u : _p_graph->nodes()) {
      if (!fixed_nodes[u]) {
        insert_node_into_pq<true>(u);
      }
    }
  }

  void update_pq_after_move(
      const NodeID u, const BlockID from, const BlockID to, const std::vector<bool> &fixed_nodes
  ) {
    // update neighbors
    for (const auto [e, v] : _p_graph->neighbors(u)) {
      if (fixed_nodes[v]) {
        continue;
      }

      const BlockID v_block = _p_graph->block(v);

      if (v_block == from || v_block == to) {
        const auto [best_gain, best_target_block] = find_best_target_block<false>(v);
        if (_pq.contains(v)) {
          _pq.change_priority(v, best_gain);
        } else if (!_marker.get(v)) {
          insert_node_into_pq<false>(v);
          _marker.set(v);
        }
      }
    }
  }

  template <bool initialization> void insert_node_into_pq(const NodeID u) {
    const auto [best_gain, best_target_block] = find_best_target_block<initialization>(u);
    if (_p_graph->block(u) != best_target_block) {
      _pq.push(u, best_gain);
      if (initialization) {
        _marker.set(u);
      }
    }
  }

  template <bool initialization>
  std::pair<EdgeWeight, BlockID> find_best_target_block(const NodeID u) {
    const BlockID u_block = _p_graph->block(u);

    auto action = [&](auto &map) {
      EdgeWeight internal_degree = 0;
      for (const auto [e, v] : _p_graph->neighbors(u)) {
        KASSERT(v < _p_graph->n());

        const BlockID v_block = _p_graph->block(v);
        const EdgeWeight e_weight = _p_graph->edge_weight(e);

        if (v_block != u_block) {
          map[v_block] += e_weight;
        } else {
          internal_degree += e_weight;
        }
      }

      EdgeWeight best_gain = std::numeric_limits<EdgeWeight>::min();
      BlockID best_target_block = u_block;

      const NodeWeight u_weight = _p_graph->node_weight(u);

      for (const auto [current_target_block, current_gain] : map.entries()) {
        // compute weight of block if we were to move the node
        BlockWeight block_weight_prime =
            _global_graph.block_weight(current_target_block) + u_weight;
        if (!initialization) {
          block_weight_prime += _block_weight_deltas[current_target_block];
        }
        const bool feasible =
            block_weight_prime <= _p_ctx.graph->max_block_weight(current_target_block);

        // accept as better block if gain is larger
        // if gain is equal, flip a coin
        if (feasible &&
            (current_gain > best_gain || (current_gain == best_gain && _rand.random_bool()))) {
          best_gain = current_gain;
          best_target_block = current_target_block;
        }
      }

      // subtract internal degree to get the actual gain value of this move
      best_gain -= internal_degree; // overflow OK, value unused if still set to min

      map.clear(); // clear for next node
      return std::make_pair(best_gain, best_target_block);
    };

    _rating_map.update_upper_bound_size(std::min<BlockID>(_p_ctx.k, _p_graph->degree(u)));
    return _rating_map.run_with_map(action, action);
  }

  // initialized by ctor
  const DistributedPartitionedGraph &_global_graph;
  const FMRefinementContext &_fm_ctx;
  const PartitionContext &_p_ctx;
  RatingMap<EdgeWeight, NodeID> _rating_map;
  AdaptiveStoppingPolicy _stopping_policy;

  // initialized by initialize_refiner()
  shm::PartitionedGraph *_p_graph;
  BinaryMaxHeap<EdgeWeight> _pq{0};
  Marker<> _marker{0};

  // initialized right here
  Random &_rand{Random::instance()};
  std::unordered_map<BlockID, BlockWeight> _block_weight_deltas;
};
} // namespace

LocalFMRefiner::LocalFMRefiner(
    const Context &ctx, DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
)
    : _fm_ctx(ctx.refinement.fm),
      _p_graph(p_graph),
      _p_ctx(p_ctx) {}

void LocalFMRefiner::initialize() {
  _locked.resize(_p_graph.total_n());
}

bool LocalFMRefiner::refine() {
  SCOPED_TIMER("FM");

  if (_fm_ctx.contract_border) {
    START_TIMER("Initialize external degrees");
    init_external_degrees();
    STOP_TIMER();
  }

  if (kStatistics) {
    _stats = Statistics{};
    _stats.initial_cut = metrics::edge_cut(_p_graph);
  }

  for (_round = 0; _round < _fm_ctx.num_iterations; ++_round) {
    refinement_round();
  }

  if (kStatistics) {
    _stats.final_cut = metrics::edge_cut(_p_graph);
    _stats.print();
  }

  return false;
}

void LocalFMRefiner::refinement_round() {
  // collect seed nodes
  const auto seed_nodes = find_seed_nodes();

  std::vector<shm::Graph> local_graphs(seed_nodes.size());
  std::vector<shm::PartitionedGraph> p_local_graphs(seed_nodes.size());
  std::vector<std::vector<GlobalNodeID>> local_graph_mappings(seed_nodes.size());
  std::vector<std::vector<bool>> fixed_nodes(seed_nodes.size());
  using GlobalMovesMap = growt::GlobalNodeIDMap<BlockID>;
  GlobalMovesMap global_moves(0);
  tbb::enumerable_thread_specific<typename GlobalMovesMap::handle_type> global_moves_handle_ets{
      [&] {
        return GlobalMovesMap::handle_type{global_moves};
      }};

  START_TIMER("Collect local search graphs");
  // mark seed nodes as taken
  tbb::parallel_for<std::size_t>(0, seed_nodes.size(), [&](const std::size_t i) {
    _locked[seed_nodes[i]] = 1;
  });

  tbb::enumerable_thread_specific<LocalLocalFMRefiner> local_refiner_ets([&] {
    return LocalLocalFMRefiner(_p_graph, _fm_ctx, _p_ctx);
  });

  auto do_refine_local_graph = [&](const std::size_t i) {
    auto &p_local_graph = p_local_graphs[i];
    const auto &local_fixed_nodes = fixed_nodes[i];
    auto &refiner = local_refiner_ets.local();
    const auto moves = refiner.refine(p_local_graph, local_fixed_nodes);
    refiner.commit_block_weight_deltas(_p_graph);
    auto &global_moves_handle = global_moves_handle_ets.local();
    const auto &mapping = local_graph_mappings[i];

    if (kStatistics && !moves.empty()) {
      ++_stats.num_searches_with_improvement;
    }

    for (const auto [u, to] : moves) {
      const GlobalNodeID global_u = mapping[u];

      // move nodes owned by this PE right away
      if (_fm_ctx.premove_locally && _p_graph.is_owned_global_node(global_u)) {
        const NodeID local_u = _p_graph.global_to_local_node(global_u);
        const BlockID from = _p_graph.block(local_u);
        _p_graph.set_block<false>(local_u, to);

        // update external degrees
        if (_fm_ctx.contract_border) {
          for (const auto [e, v] : _p_graph.neighbors(local_u)) {
            const EdgeWeight weight = _p_graph.edge_weight(e);
            external_degree(v, from) -= weight;
            external_degree(v, to) += weight;
          }
        }
      } else { // remember non-local nodes in a global move buffer
        const GlobalNodeID global_u_prime = global_u + 1; // growt does not allow 0 as key
        [[maybe_unused]] const auto [it, success] = global_moves_handle.insert(global_u_prime, to);
        KASSERT(success);
      }
    }
  };

  auto do_build_local_graph = [&](const std::size_t i) {
    build_local_graph(
        seed_nodes[i], local_graphs[i], p_local_graphs[i], local_graph_mappings[i], fixed_nodes[i]
    );
  };

  if (_fm_ctx.sequential) {
    for (std::size_t i = 0; i < local_graphs.size(); ++i) {
      START_TIMER("Build local graphs");
      do_build_local_graph(i);
      STOP_TIMER();
      START_TIMER("Refine local graphs");
      do_refine_local_graph(i);
      STOP_TIMER();
    }
  } else {
    START_TIMER("Build and refine local graphs");
    tbb::parallel_for<std::size_t>(0, local_graphs.size(), [&](const std::size_t i) {
      do_build_local_graph(i);
      do_refine_local_graph(i);
    });
    STOP_TIMER();
  }
  STOP_TIMER();

  START_TIMER("Broadcast moves");
  struct MoveMessage {
    MoveMessage() {}
    MoveMessage(const NodeID node, const BlockID block) : node(node), block(block) {}
    NodeID node = 0;
    BlockID block = 0;
  };

  START_TIMER("Build send buffer");
  std::vector<std::vector<MoveMessage>> sendbuf(mpi::get_comm_size(_p_graph.communicator()));
  for (const auto [to, global_u_prime] : global_moves_handle_ets.local()) { // @todo parallelize
    KASSERT(global_u_prime != 0u);
    const GlobalNodeID global_u = global_u_prime - 1;
    KASSERT(global_u < _p_graph.global_n());

    // If we move nodes between global synchronization steps, the global move
    // buffer should not contain any moves for owned nodes
    KASSERT(!_p_graph.is_owned_global_node(global_u) || !_fm_ctx.premove_locally);

    if (!_fm_ctx.premove_locally && _p_graph.is_owned_global_node(global_u)) {
      const NodeID local_u = _p_graph.global_to_local_node(global_u);
      _p_graph.set_block<false>(local_u, to);
    } else {
      const PEID owner = _p_graph.find_owner_of_global_node(global_u);
      sendbuf[owner].emplace_back(static_cast<NodeID>(global_u - _p_graph.offset_n(owner)), to);
    }
  }
  STOP_TIMER();

  mpi::sparse_alltoall<MoveMessage>(
      sendbuf,
      [&](const auto &recvbuf) {
        tbb::parallel_for<std::size_t>(0, recvbuf.size(), [&](const std::size_t i) {
          const auto [node, block] = recvbuf[i];
          _p_graph.set_block(node, block);
        });
      },
      _p_graph.communicator()
  );
  STOP_TIMER();

  START_TIMER("Synchronize ghost node labels");
  graph::synchronize_ghost_node_block_ids(_p_graph);
  _p_graph.reinit_block_weights();
  STOP_TIMER();

  // free all locked nodes
  std::fill(_locked.begin(), _locked.end(), 0);
}

tbb::concurrent_vector<NodeID> LocalFMRefiner::find_seed_nodes() {
  SCOPED_TIMER("Select seed nodes");

  // compute indepentent set
  NoinitVector<double> score(_p_graph.n());
  std::uniform_real_distribution<> dist(0.0, 1.0 * _p_graph.total_n());
  tbb::enumerable_thread_specific<std::mt19937> generator_ets;

  _p_graph.pfor_nodes([&](const NodeID u) {
    // score for ghost nodes is computed lazy
    if (!_p_graph.is_owned_node(u)) {
      score[u] = -1.0;
      return;
    }

    // check if u is a bordert node
    bool is_border_node = false;
    const BlockID u_block = _p_graph.block(u);

    for (const auto [e, v] : _p_graph.neighbors(u)) {
      if (_p_graph.block(v) != u_block) {
        is_border_node = true;
        break;
      }
    }

    if (is_border_node) {
      auto &generator = generator_ets.local();
      generator.seed(_round + _p_graph.local_to_global_node(u));
      score[u] = dist(generator);
    } else {
      score[u] = std::numeric_limits<double>::max();
    }
  });

  // find seed nodes
  tbb::concurrent_vector<NodeID> seed_nodes;
  _p_graph.pfor_nodes([&](const NodeID u) {
    if (score[u] == std::numeric_limits<double>::max()) {
      return; // not a border node
    }

    bool is_seed_node = true;
    for (const auto [e, v] : _p_graph.neighbors(u)) {
      if (score[v] < 0) { // ghost node, compute score lazy
        auto &generator = generator_ets.local();
        generator.seed(_round + _p_graph.local_to_global_node(v));
        score[v] = dist(generator);
      }

      if (score[v] < score[u]) {
        is_seed_node = false;
        break;
      }
    }

    if (is_seed_node) {
      seed_nodes.push_back(u);
    }
  });

  DBG << "Selected " << seed_nodes.size() << " seed nodes";

  return seed_nodes;
}

void LocalFMRefiner::build_local_graph(
    const NodeID seed_node,
    shm::Graph &out_graph,
    shm::PartitionedGraph &out_p_graph,
    std::vector<GlobalNodeID> &mapping,
    std::vector<bool> &fixed
) {
  struct DiscoveredNode {
    DiscoveredNode() {}
    DiscoveredNode(const NodeID node, const NodeID distance, const bool border)
        : node(node),
          distance(distance),
          border(border) {}

    NodeID node;
    NodeID distance;
    bool border;
  };

  std::vector<DiscoveredNode> discovered_owned_nodes;
  std::vector<DiscoveredNode> discovered_ghost_nodes;

  std::stack<NodeID> search_front;
  search_front.push(seed_node);
  NodeID current_front_size = 1; // no. of elements beloning to current front
  NodeID current_distance = 0;
  Random &rand = Random::instance();

  while (current_distance < _fm_ctx.radius + 1 && !search_front.empty()) {
    const NodeID current = search_front.top();
    search_front.pop();
    --current_front_size;

    // try to take this node
    if (_p_graph.is_owned_node(current)) {
      discovered_owned_nodes.emplace_back(current, current_distance, false);

      const bool sample_neighbors =
          _fm_ctx.bound_degree > 0 && _p_graph.degree(current) > _fm_ctx.bound_degree;
      const double prob =
          sample_neighbors ? 1.0 * _fm_ctx.bound_degree / _p_graph.degree(current) : 1.0;

      // grow to neighbors
      for (const auto [e, v] : _p_graph.neighbors(current)) {
        const bool take = sample_neighbors ? rand.random_bool(prob) : true;
        std::uint8_t free = 0;

        if (take && current_distance + 1 < _fm_ctx.radius &&
            ((!_fm_ctx.overlap_regions && _locked[v].compare_exchange_strong(free, 1)
             ) // obtain overship
             ||
             (_fm_ctx.overlap_regions && !_locked[v]))) { // take everything, except for seed nodes
          search_front.push(v);
        } else if (!_fm_ctx.contract_border) { // otherwise, we build fake
                                               // neighbors later on
          discovered_owned_nodes.emplace_back(v, current_distance + 1, true);
        }
      }
    } else {
      discovered_owned_nodes.emplace_back(current, current_distance, true);
      discovered_ghost_nodes.emplace_back(current, current_distance, false);
    }

    // +1 distance from seed
    if (current_front_size == 0) {
      current_front_size = search_front.size();
      ++current_distance;
    }
  }

  // @todo inter-PE growth

  // build local graphs
  const NodeID real_n = discovered_owned_nodes.size();
  const NodeID n = _fm_ctx.contract_border ? real_n + _p_graph.k() : real_n;

  // build local_graph_mappings[i]
  for (const auto [u, d, b] : discovered_owned_nodes) {
    mapping.push_back(_p_graph.local_to_global_node(u));
    fixed.push_back(b);
  }
  if (_fm_ctx.contract_border) {
    KASSERT(std::none_of(fixed.begin(), fixed.end(), [](const bool v) { return v; }));
    for (BlockID b = 0; b < _p_graph.k(); ++b) {
      fixed.push_back(true);
    }
  }

  // build inverse mapping
  std::unordered_map<GlobalNodeID, NodeID> to_local_map;
  for (std::size_t i = 0; i < real_n; ++i) {
    to_local_map[mapping[i]] = i;
  }

  // build local graph
  StaticArray<shm::EdgeID> local_nodes(n + 1);
  std::vector<shm::NodeID> local_edges;
  StaticArray<shm::NodeWeight> local_node_weights(n);
  std::vector<shm::EdgeWeight> local_edge_weights;
  NodeID next_node = 0;
  NodeID border_size = 0;
  std::vector<shm::EdgeWeight> covered_external_degrees(_p_graph.k());

  if (_fm_ctx.contract_border) {
    border_size = _p_graph.k();
  }

  for (const auto [u, d, b] : discovered_owned_nodes) {
    local_nodes[next_node] = local_edges.size();
    local_node_weights[next_node] = _p_graph.node_weight(u);
    ++next_node;

    if (!b) {
      const bool last_hop = _fm_ctx.contract_border && d + 1 == _fm_ctx.radius;

      for (const auto [e, v] : _p_graph.neighbors(u)) {
        const GlobalNodeID global_v = _p_graph.local_to_global_node(v);
        if (to_local_map.find(global_v) != to_local_map.end()) {
          KASSERT(to_local_map[global_v] < real_n);

          local_edges.push_back(to_local_map[global_v]);
          local_edge_weights.push_back(_p_graph.edge_weight(e));

          if (last_hop) {
            covered_external_degrees[_p_graph.block(v)] += _p_graph.edge_weight(e);
          }
        } else {
          KASSERT(_fm_ctx.contract_border || mpi::get_comm_rank(_p_graph.communicator()) > 0);
        }
      }

      if (last_hop) { // connect to pseudo-nodes
        for (BlockID b = 0; b < _p_graph.k(); ++b) {
          local_edges.push_back(real_n + b);
          local_edge_weights.push_back(external_degree(u, b) - covered_external_degrees[b]);
        }
        std::fill(covered_external_degrees.begin(), covered_external_degrees.end(), 0);
      }
    } else {
      ++border_size;
    }
  }

  for (NodeID u = real_n; u < n + 1; ++u) {
    local_nodes[u] = local_edges.size();
  }

  KASSERT([&] {
    KASSERT(static_cast<NodeID>(local_nodes.size()) == n + 1);
    KASSERT(local_nodes[0] == 0u);
    KASSERT(local_nodes[n] == static_cast<NodeID>(local_edges.size()));
    for (NodeID u = 0; u < n; ++u) {
      KASSERT(local_nodes[u] <= local_nodes[u + 1]);
      KASSERT(local_nodes[u + 1] <= local_edges.size());
    }
    for (const auto &v : local_edges) {
      KASSERT(v < n);
    }
    return true;
  }());

  if (kStatistics) {
    _stats.graphs_n.push_back(n);
    _stats.graphs_m.push_back(local_nodes.back());
    _stats.graphs_border_n.push_back(border_size);
  }

  // build partition
  StaticArray<BlockID> partition(n);
  for (std::size_t i = 0; i < discovered_owned_nodes.size(); ++i) {
    partition[i] = _p_graph.block(discovered_owned_nodes[i].node);
  }

  if (_fm_ctx.contract_border) {
    for (BlockID b = 0; b < _p_graph.k(); ++b) {
      partition[real_n + b] = b;
      local_node_weights[real_n + b] = 1; // should not matter
    }
  }

  // create graph objects
  StaticArray<shm::NodeID> local_edges_prime(local_edges.size());
  std::copy(local_edges.begin(), local_edges.end(), local_edges_prime.begin());
  StaticArray<shm::EdgeWeight> local_edge_weights_prime(local_edge_weights.size());
  std::copy(local_edge_weights.begin(), local_edge_weights.end(), local_edge_weights_prime.begin());

  out_graph = shm::Graph(
      std::move(local_nodes),
      std::move(local_edges_prime),
      std::move(local_node_weights),
      std::move(local_edge_weights_prime)
  );
  out_p_graph =
      shm::PartitionedGraph(shm::no_block_weights, out_graph, _p_ctx.k, std::move(partition));
}

void LocalFMRefiner::init_external_degrees() {
  _external_degrees.resize(_p_graph.n() * _p_graph.k());

  const BlockID k = _p_graph.k();
  _p_graph.pfor_nodes([&](const NodeID u) {
    auto begin = _external_degrees.begin() + u * k;
    auto end = begin + k;
    std::fill(begin, end, 0u);

    for (const auto [e, v] : _p_graph.neighbors(u)) {
      const BlockID v_block = _p_graph.block(v);
      const EdgeWeight e_weight = _p_graph.edge_weight(e);
      *(begin + v_block) += e_weight;
    }
  });
}
} // namespace kaminpar::dist
