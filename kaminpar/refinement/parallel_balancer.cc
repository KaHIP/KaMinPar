#include "refinement/parallel_balancer.h"

namespace kaminpar {
void ParallelBalancer::initialize(const PartitionedGraph &p_graph) { UNUSED(p_graph); }

bool ParallelBalancer::balance(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  _p_graph = &p_graph;
  _p_ctx = &p_ctx;
  _stats.reset();
  ALWAYS_ASSERT(_marker.capacity() >= _p_graph->n());
  _marker.reset();

  const NodeWeight initial_overload = metrics::total_overload(*_p_graph, *_p_ctx);
  if (initial_overload == 0) { return true; }

  const EdgeWeight initial_cut = IFDBG(metrics::edge_cut(*_p_graph));

  init_pq();
  const BlockWeight delta = perform_round();
  const NodeWeight new_overload = initial_overload - delta;

  CLOG << "-> Balancer: overload=[" << initial_overload << " --> " << new_overload << "]";
  DBG << "-> Balancer: cut=" << C(initial_cut, metrics::edge_cut(*_p_graph));
  if (kStatistics) { _stats.print(); }

  return new_overload == 0;
}

BlockWeight ParallelBalancer::perform_round() {
  if (kStatistics) {
    _stats.initial_cut = metrics::edge_cut(*_p_graph);
    _stats.initial_overload = metrics::total_overload(*_p_graph, *_p_ctx);
  }

  // reset feasible target blocks
  for (auto &blocks : _feasible_target_blocks) { blocks.clear(); }

  tbb::enumerable_thread_specific<BlockWeight> overload_delta;

  START_TIMER("Main loop");
  tbb::parallel_for(static_cast<BlockID>(0), _p_graph->k(), [&](const BlockID from) {
    BlockWeight current_overload = block_overload(from);

    if (current_overload > 0 && _feasible_target_blocks.local().empty()) {
      init_feasible_target_blocks();
      DBG << "Block " << from << " with overload: " << current_overload << ": "
          << _feasible_target_blocks.local().size() << " feasible target blocks and " << _pq.size(from)
          << " nodes in PQ: total weight of PQ is " << _pq_weight[from];
    }

    while (current_overload > 0 && !_pq.empty(from)) {
      ASSERT(current_overload == std::max(0, _p_graph->block_weight(from) - _p_ctx->max_block_weight(from)));

      const NodeID u = _pq.peek_max_id(from);
      const NodeWeight u_weight = _p_graph->node_weight(u);
      const double expected_relative_gain = _pq.peek_max_key(from);
      _pq.pop_max(from);
      _pq_weight[from] -= u_weight;
      ASSERT(_marker.get(u));

      auto [to, actual_relative_gain] = compute_gain(u, from);
      if (expected_relative_gain == actual_relative_gain) { // gain still correct --> try to move it
        bool moved_node = false;

        if (to == from) { // internal node --> move to random underloaded block
          moved_node = move_to_random_block(u);
          if (kStatistics) {
            _stats.num_successful_random_moves += moved_node;
            _stats.num_unsuccessful_random_moves += (1 - moved_node);
            ++_stats.num_moved_internal_nodes;
          }
        } else if (move_node_if_possible(u, from, to)) { // border node -> move to promising block
          moved_node = true;
          if (kStatistics) {
            ++_stats.num_moved_border_nodes;
            ++_stats.num_successful_adjacent_moves;
          }
        } else { // border node could not be moved -> try again
          if (kStatistics) {
            ++_stats.num_pq_reinserts;
            ++_stats.num_unsuccessful_adjacent_moves;
          }
        }

        if (moved_node) { // update overload if node was moved
          const BlockWeight delta = std::min(current_overload, u_weight);
          current_overload -= delta;
          overload_delta.local() += delta;

          // try to add neighbors of moved node to PQ
          for (const NodeID v : _p_graph->adjacent_nodes(u)) {
            if (!_marker.get(v) && _p_graph->block(v) == from) { add_to_pq(from, v); }
            _marker.set(v);
          }
        } else {
          add_to_pq(from, u, u_weight, actual_relative_gain);
        }
      } else { // gain changed after insertion --> try again with new gain
        add_to_pq(from, u, _p_graph->node_weight(u), actual_relative_gain);
        if (kStatistics) { ++_stats.num_pq_reinserts; }
      }
    }

    ASSERT(current_overload == std::max(0, _p_graph->block_weight(from) - _p_ctx->max_block_weight(from)));
  });
  STOP_TIMER();

  if (kStatistics) {
    _stats.final_cut = metrics::edge_cut(*_p_graph);
    _stats.final_overload = metrics::total_overload(*_p_graph, *_p_ctx);
  }

  const BlockWeight global_overload_delta = overload_delta.combine(std::plus{});
  return global_overload_delta;
}

bool ParallelBalancer::add_to_pq(const BlockID b, const NodeID u) {
  ASSERT(b == _p_graph->block(u));

  const auto [to, rel_gain] = compute_gain(u, b);
  return add_to_pq(b, u, _p_graph->node_weight(u), rel_gain);
}

bool ParallelBalancer::add_to_pq(const BlockID b, const NodeID u, const NodeWeight u_weight, const double rel_gain) {
  ASSERT(u_weight == _p_graph->node_weight(u));
  ASSERT(b == _p_graph->block(u));

  if (_pq_weight[b] < block_overload(b) || _pq.empty(b) || rel_gain > _pq.peek_min_key(b)) {
    DBG << "Add " << u << " pq weight " << _pq_weight[b] << " rel_gain " << rel_gain;
    _pq.push(b, u, rel_gain);
    _pq_weight[b] += u_weight;

    if (rel_gain > _pq.peek_min_key(b)) {
      const NodeID min_node = _pq.peek_min_id(b);
      const NodeWeight min_weight = _p_graph->node_weight(min_node);
      if (_pq_weight[b] - min_weight >= block_overload(b)) {
        _pq.pop_min(b);
        _pq_weight[b] -= min_weight;
      }
    }

    return true;
  }

  return false;
}

void ParallelBalancer::init_pq() {
  SCOPED_TIMER("Initialize balancer PQ");

  const BlockID k = _p_graph->k();

  tbb::enumerable_thread_specific<std::vector<DynamicBinaryMinHeap<NodeID, double>>> local_pq{
      [&] { return std::vector<DynamicBinaryMinHeap<NodeID, double>>(k); }};
  tbb::enumerable_thread_specific<std::vector<NodeWeight>> local_pq_weight{[&] { return std::vector<NodeWeight>(k); }};

  _marker.reset();

  // build thread-local PQs: one PQ for each thread and block, each PQ for block b has at most roughly |overload[b]|
  // weight
  START_TIMER("Thread-local");
  tbb::parallel_for(static_cast<NodeID>(0), _p_graph->n(), [&](const NodeID u) {
    auto &pq = local_pq.local();
    auto &pq_weight = local_pq_weight.local();

    const BlockID b = _p_graph->block(u);
    const BlockWeight overload = block_overload(b);

    if (overload > 0) { // node in overloaded block
      const auto [max_gainer, rel_gain] = compute_gain(u, b);
      const bool need_more_nodes = (pq_weight[b] < overload);
      if (need_more_nodes || pq[b].empty() || rel_gain > pq[b].peek_key()) {
        if (!need_more_nodes) {
          const NodeWeight u_weight = _p_graph->node_weight(u);
          const NodeWeight min_weight = _p_graph->node_weight(pq[b].peek_id());
          if (pq_weight[b] + u_weight - min_weight >= overload) { pq[b].pop(); }
        }
        pq[b].push(u, rel_gain);
        _marker.set(u);
      }
    }
  });
  STOP_TIMER();

  // build global PQ: one PQ per block, block-level parallelism
  _pq.clear();

  START_TIMER("Merge thread-local PQs");
  tbb::parallel_for(static_cast<BlockID>(0), k, [&](const BlockID b) {
    if (kStatistics && block_overload(b) > 0) { ++_stats.num_overloaded_blocks; }
    _pq_weight[b] = 0;

    for (auto &pq : local_pq) {
      for (const auto &[u, rel_gain] : pq[b].elements()) { add_to_pq(b, u, _p_graph->node_weight(u), rel_gain); }
    }

    if (!_pq.empty(b)) {
      DBG << "PQ " << b << ": weight=" << _pq_weight[b] << ", " << _pq.peek_min_key(b) << " < key < "
          << _pq.peek_max_key(b);
    } else {
      DBG << "PQ " << b << ": empty";
    }
  });
  STOP_TIMER();

  _stats.total_pq_sizes = _pq.size();
}

[[nodiscard]] std::pair<BlockID, double> ParallelBalancer::compute_gain(const NodeID u, const BlockID u_block) const {
  const NodeWeight u_weight = _p_graph->node_weight(u);
  BlockID max_gainer = u_block;
  EdgeWeight max_external_gain = 0;
  EdgeWeight internal_degree = 0;

  auto action = [&](auto &map) {
    // compute external degree to each adjacent block that can take u without becoming overloaded
    for (const auto [e, v] : _p_graph->neighbors(u)) {
      const BlockID v_block = _p_graph->block(v);
      if (u_block != v_block && _p_graph->block_weight(v_block) + u_weight <= _p_ctx->max_block_weight(v_block)) {
        map[v_block] += _p_graph->edge_weight(e);
      } else if (u_block == v_block) {
        internal_degree += _p_graph->edge_weight(e);
      }
    }

    // select neighbor that maximizes gain
    Randomize &rand = Randomize::instance();
    for (const auto [block, gain] : map.entries()) {
      if (gain > max_external_gain || (gain == max_external_gain && rand.random_bool())) {
        max_gainer = block;
        max_external_gain = gain;
      }
    }
    map.clear();
  };

  auto &rating_map = _rating_map.local();
  rating_map.update_upper_bound_size(_p_graph->degree(u));
  rating_map.run_with_map(action, action);

  // compute absolute and relative gain based on internal degree / external gain
  const Gain gain = max_external_gain - internal_degree;
  const double relative_gain = compute_relative_gain(gain, u_weight);
  return {max_gainer, relative_gain};
}

bool ParallelBalancer::move_node_if_possible(const NodeID u, const BlockID from, const BlockID to) {
  const NodeWeight u_weight = _p_graph->node_weight(u);
  BlockWeight old_weight = _p_graph->block_weight(to);

  while (old_weight + u_weight <= _p_ctx->max_block_weight(to)) {
    if (_p_graph->_block_weights[to].compare_exchange_weak(old_weight, old_weight + u_weight)) {
      _p_graph->_block_weights[from] -= u_weight;
      _p_graph->set_block<false>(u, to); // don't update block weight -- we already did that
      return true;
    }
  }

  return false;
}

bool ParallelBalancer::move_to_random_block(const NodeID u) {
  auto &feasible_target_blocks = _feasible_target_blocks.local();
  const BlockID u_block = _p_graph->block(u);

  while (!feasible_target_blocks.empty()) {
    // get random block from feasible block list
    const std::size_t n = feasible_target_blocks.size();
    const std::size_t i = Randomize::instance().random_index(0, n);
    const BlockID b = feasible_target_blocks[i];

    // try to move node to that block, if possible, operation succeeded
    if (move_node_if_possible(u, u_block, b)) { return true; }

    // loop terminated without return, hence moving u to b failed --> we no longer consider b to be a feasible target
    // block and remove it from the list
    std::swap(feasible_target_blocks[i], feasible_target_blocks.back());
    feasible_target_blocks.pop_back();
  }

  // there are no more feasible target blocks -> operation failed
  return false;
}

void ParallelBalancer::init_feasible_target_blocks() {
  if (kStatistics) { ++_stats.num_feasible_target_block_inits; }

  auto &blocks = _feasible_target_blocks.local();
  blocks.clear();
  for (const BlockID b : _p_graph->blocks()) {
    if (_p_graph->block_weight(b) < _p_ctx->perfectly_balanced_block_weight(b)) { blocks.push_back(b); }
  }
}
} // namespace kaminpar
