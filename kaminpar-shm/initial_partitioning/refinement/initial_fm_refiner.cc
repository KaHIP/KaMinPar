/*******************************************************************************
 * Sequential 2-way FM refinement used during initial bipartitioning.
 *
 * @file:   initial_fm_refiner.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/initial_partitioning/refinement/initial_fm_refiner.h"

#include <algorithm>

#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/metrics.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/logger.h"

namespace kaminpar::shm {

namespace {

SET_DEBUG(false);

}

using Queues = std::array<BinaryMinHeap<EdgeWeight>, 2>;

namespace fm {

void SimpleStoppingPolicy::init(const CSRGraph *) {
  reset();
}

bool SimpleStoppingPolicy::should_stop(const InitialFMRefinementContext &fm_ctx) {
  return _num_steps > fm_ctx.num_fruitless_moves;
}

void SimpleStoppingPolicy::reset() {
  _num_steps = 0;
}

void SimpleStoppingPolicy::update(const EdgeWeight) {
  ++_num_steps;
}

void AdaptiveStoppingPolicy::init(const CSRGraph *graph) {
  _beta = std::sqrt(graph->n());
  reset();
}

bool AdaptiveStoppingPolicy::should_stop(const InitialFMRefinementContext &fm_ctx) {
  const double factor = (fm_ctx.alpha / 2.0) - 0.25;
  return (_num_steps > _beta) && ((_Mk == 0) || (_num_steps >= (_variance / (_Mk * _Mk)) * factor));
}

void AdaptiveStoppingPolicy::reset() {
  _num_steps = 0;
  _variance = 0.0;
}

void AdaptiveStoppingPolicy::update(const EdgeWeight gain) {
  ++_num_steps;

  // See Knuth TAOCP vol 2, 3rd edition, page 232 or
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
  if (_num_steps == 1) {
    _MkMinus1 = static_cast<double>(gain);
    _Mk = _MkMinus1;
    _SkMinus1 = 0.0;
  } else {
    _Mk = _MkMinus1 + (gain - _MkMinus1) / _num_steps;
    _Sk = _SkMinus1 + (gain - _MkMinus1) * (gain - _Mk);
    _variance = _Sk / (_num_steps - 1.0);

    // Prepare for next iteration:
    _MkMinus1 = _Mk;
    _SkMinus1 = _Sk;
  }
}

//! Always move the next node from the heavier block. This should improve
//! balance.
struct MaxWeightSelectionPolicy {
  std::size_t operator()(
      const PartitionedCSRGraph &p_graph,
      const PartitionContext &context,
      const Queues &,
      Random &rand
  ) {
    const auto weight0 = p_graph.block_weight(0) - context.perfectly_balanced_block_weight(0);
    const auto weight1 = p_graph.block_weight(1) - context.perfectly_balanced_block_weight(1);
    return weight1 > weight0 || (weight0 == weight1 && rand.random_bool());
  }
};

//! Always select the node with the highest gain / lowest loss.
struct MaxGainSelectionPolicy {
  std::size_t operator()(
      const PartitionedCSRGraph &p_graph,
      const PartitionContext &context,
      const Queues &queues,
      Random &rand
  ) {
    const auto loss0 =
        queues[0].empty() ? std::numeric_limits<EdgeWeight>::max() : queues[0].peek_key();
    const auto loss1 =
        queues[1].empty() ? std::numeric_limits<EdgeWeight>::max() : queues[1].peek_key();

    if (loss0 == loss1) {
      return MaxWeightSelectionPolicy()(p_graph, context, queues, rand);
    }

    return loss1 < loss0;
  }
};

struct MaxOverloadSelectionPolicy {
  std::size_t operator()(
      const PartitionedCSRGraph &p_graph,
      const PartitionContext &context,
      const Queues &queues,
      Random &rand
  ) {
    const NodeWeight overload0 =
        std::max<NodeWeight>(0, p_graph.block_weight(0) - context.max_block_weight(0));
    const NodeWeight overload1 =
        std::max<NodeWeight>(0, p_graph.block_weight(1) - context.max_block_weight(1));

    if (overload0 == 0 && overload1 == 0) {
      return MaxGainSelectionPolicy()(p_graph, context, queues, rand);
    }

    return overload1 > overload0 || (overload1 == overload0 && rand.random_bool());
  }
};

//! Accept better cuts that preserve the minimum block weights and do not increase overload.
struct BalancedMinCutAcceptancePolicy {
  bool operator()(
      const PartitionedCSRGraph &p_graph,
      const PartitionContext &p_ctx,
      const EdgeWeight accepted_overload,
      const EdgeWeight current_overload,
      const EdgeWeight accepted_delta,
      const EdgeWeight delta
  ) {
    return current_overload <= accepted_overload && delta < accepted_delta &&
           metrics::is_min_balanced(p_graph, p_ctx);
  }
};
} // namespace fm

template <typename QueueSelectionPolicy, typename CutAcceptancePolicy, typename StoppingPolicy>
void InitialFMRefiner<QueueSelectionPolicy, CutAcceptancePolicy, StoppingPolicy>::init(
    const CSRGraph &graph
) {
  _graph = &graph;
  _stopping_policy.init(&graph);

  if (_queues[0].capacity() < graph.n()) {
    _queues[0].resize(graph.n());
  }
  if (_queues[1].capacity() < graph.n()) {
    _queues[1].resize(graph.n());
  }
  if (_marker.capacity() < graph.n()) {
    _marker.resize(graph.n());
  }
  if (_weighted_degrees.size() < graph.n()) {
    _weighted_degrees.resize(graph.n());
  }

  init_weighted_degrees();
}

template <typename QueueSelectionPolicy, typename CutAcceptancePolicy, typename StoppingPolicy>
bool InitialFMRefiner<QueueSelectionPolicy, CutAcceptancePolicy, StoppingPolicy>::refine(
    PartitionedCSRGraph &p_graph, const PartitionContext &p_ctx
) {
  _p_ctx = &p_ctx;

  KASSERT(&p_graph.graph() == _graph);
  KASSERT(p_graph.k() == 2u);
  KASSERT(_p_ctx->k == 2u);

  // Avoid redundant edge cut scans by computing initial cut directly inside init_pq()
  if (_r_ctx.num_iterations == 1) {
    EdgeWeight initial_edge_cut = 0;
    const EdgeWeight delta = round(p_graph, &initial_edge_cut);
    _last_edge_cut = initial_edge_cut + delta;
    return false;
  }

  EdgeWeight initial_edge_cut = 0;
  EdgeWeight cur_edge_cut = round(p_graph, &initial_edge_cut);

  // If there is no improvement possible, abort early
  if (initial_edge_cut == 0) {
    _last_edge_cut = 0;
    return false;
  }

  EdgeWeight prev_edge_cut = initial_edge_cut;
  cur_edge_cut += initial_edge_cut;
  for (std::size_t it = 1;
       0 < cur_edge_cut && it < _r_ctx.num_iterations && !abort(prev_edge_cut, cur_edge_cut);
       ++it) {
    prev_edge_cut = cur_edge_cut;
    cur_edge_cut += round(p_graph);
  }

  _last_edge_cut = cur_edge_cut;
  return cur_edge_cut < initial_edge_cut;
}

template <typename QueueSelectionPolicy, typename CutAcceptancePolicy, typename StoppingPolicy>
bool InitialFMRefiner<QueueSelectionPolicy, CutAcceptancePolicy, StoppingPolicy>::abort(
    const EdgeWeight prev_edge_weight, const EdgeWeight cur_edge_weight
) const {
  return (1.0 - 1.0 * cur_edge_weight / prev_edge_weight) < _r_ctx.improvement_abortion_threshold;
}

template <typename QueueSelectionPolicy, typename CutAcceptancePolicy, typename StoppingPolicy>
EdgeWeight InitialFMRefiner<QueueSelectionPolicy, CutAcceptancePolicy, StoppingPolicy>::round(
    PartitionedCSRGraph &p_graph, EdgeWeight *out_initial_cut
) {
  DBG << "Initial refiner initialized with n=" << p_graph.graph().n()
      << ", m=" << p_graph.graph().m() << ", k=" << p_graph.k();

  KASSERT(
      p_graph.k() == 2u,
      "initial 2-way FM refinement can only refine 2-way partitions",
      assert::light
  );

#if KASSERT_ENABLED(ASSERTION_LEVEL_HEAVY)
  const bool initially_feasible = metrics::is_feasible(p_graph, *_p_ctx);
#endif

  _stopping_policy.reset();

  const EdgeWeight computed_initial_cut = init_pq(p_graph);
  if (out_initial_cut != nullptr) {
    *out_initial_cut = computed_initial_cut;
  }
  if (computed_initial_cut == 0) {
    return 0;
  }

  std::size_t active = 0; // block from which we select a node for movement

  EdgeWeight current_overload = metrics::total_overload(p_graph, *_p_ctx);
  EdgeWeight accepted_overload = current_overload;

  EdgeWeight current_delta = 0;
  EdgeWeight accepted_delta = 0;
#if KASSERT_ENABLED(ASSERTION_LEVEL_HEAVY)
  const EdgeWeight initial_edge_cut = metrics::edge_cut_seq(p_graph);
#endif

  DBG << "Starting main refinement loop with #_pq[0]=" << _queues[0].size()
      << " #_pq[1]=" << _queues[1].size();

  QueueSelectionPolicy queue_selection_policy;
  CutAcceptancePolicy cut_acceptance_policy;

  while ((!_queues[0].empty() || !_queues[1].empty()) && !_stopping_policy.should_stop(_r_ctx)) {
    KASSERT(validate_pqs(p_graph), "inconsistent PQ state", assert::heavy);

    active = queue_selection_policy(p_graph, *_p_ctx, _queues, _rand);
    if (_queues[active].empty()) {
      active = 1 - active;
    }
    BinaryMinHeap<EdgeWeight> &queue = _queues[active];

    const NodeID u = queue.peek_id();
    const EdgeWeight delta = queue.peek_key();
    const BlockID from = active;
    const BlockID to = 1 - from;
    KASSERT(!_marker.get(u));
    KASSERT(from == p_graph.block(u));
    _marker.set(u);
    queue.pop();

    DBG << "Performed move, new cut=" << metrics::edge_cut_seq(p_graph);
    p_graph.set_block(u, to);
    current_delta += delta;
    _moves.push_back(u);
#if KASSERT_ENABLED(ASSERTION_LEVEL_HEAVY)
    KASSERT(initial_edge_cut + current_delta == metrics::edge_cut_seq(p_graph), "", assert::heavy);
#endif
    _stopping_policy.update(-delta); // assumes gain, not loss
    current_overload = metrics::total_overload(p_graph, *_p_ctx);

    // update gain of neighboring nodes
    _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight e_weight) {
      if (_marker.get(v)) {
        return;
      }

      const BlockID v_block = p_graph.block(v);
      const EdgeWeight loss_delta = 2 * e_weight * ((to == v_block) ? 1 : -1);

      if (_queues[v_block].contains(v)) {
        const EdgeWeight new_loss = _queues[v_block].key(v) + loss_delta;
        const bool still_boundary_node = new_loss < _weighted_degrees[v];

        if (!still_boundary_node) { // v is no longer a boundary node
          KASSERT(!is_boundary_node(p_graph, v), "", assert::heavy);
          _queues[v_block].remove(v);
        } else { // v is still a boundary node
          KASSERT(is_boundary_node(p_graph, v), "", assert::heavy);
          _queues[v_block].change_priority(v, new_loss);
        }
      } else { // since v was not a boundary node before, it must be one now
        KASSERT(is_boundary_node(p_graph, v), "", assert::heavy);
        _queues[v_block].push(v, _weighted_degrees[v] + loss_delta);
      }
    });

    // accept move if it improves the best edge cut found so far
    if (cut_acceptance_policy(
            p_graph, *_p_ctx, accepted_overload, current_overload, accepted_delta, current_delta
        )) {
      DBG << "Accepted new bipartition: delta=" << current_delta
          << " cut=" << metrics::edge_cut_seq(p_graph);
      _stopping_policy.reset();
      accepted_delta = current_delta;
      accepted_overload = current_overload;
      _moves.clear();
    }
  }

  // rollback to last accepted cut
  for (const NodeID u : _moves) {
    p_graph.set_block(u, 1 - p_graph.block(u));
  };

  // reset datastructures for next run
  for (const std::size_t i : {0, 1}) {
    _queues[i].clear();
  }
  _marker.reset();
  _moves.clear();

#if KASSERT_ENABLED(ASSERTION_LEVEL_HEAVY)
  KASSERT(!initially_feasible || accepted_delta <= 0);
  KASSERT(metrics::edge_cut_seq(p_graph) == initial_edge_cut + accepted_delta);
#endif

  return accepted_delta;
}

template <typename QueueSelectionPolicy, typename CutAcceptancePolicy, typename StoppingPolicy>
EdgeWeight InitialFMRefiner<QueueSelectionPolicy, CutAcceptancePolicy, StoppingPolicy>::init_pq(
    const PartitionedCSRGraph &p_graph
) {
  KASSERT(_queues[0].empty());
  KASSERT(_queues[1].empty());

  const NodeID n = _graph->n();
  const NodeID num_chunks = n / kChunkSize + 1;

  _chunks.clear();
  if (_chunks.size() < num_chunks) {
    _chunks.resize(num_chunks);
  }

  std::iota(_chunks.begin(), _chunks.begin() + num_chunks, 0);
  std::transform(
      _chunks.begin(),
      _chunks.begin() + num_chunks,
      _chunks.begin(),
      [](const NodeID i) { return i * kChunkSize; }
  );
  _rand.shuffle(_chunks.begin(), _chunks.begin() + num_chunks);

  std::int64_t total_external_weight = 0;
  for (NodeID i = 0; i < num_chunks; ++i) {
    const NodeID chunk = _chunks[i];
    const auto &permutation = _permutations.get(_rand);
    if (chunk + kChunkSize <= n) {
      for (const NodeID offset : permutation) {
        total_external_weight += insert_node(p_graph, chunk + offset);
      }
    } else {
      for (const NodeID offset : permutation) {
        const NodeID u = chunk + offset;
        if (u < n) {
          total_external_weight += insert_node(p_graph, u);
        }
      }
    }
  }

  KASSERT(validate_pqs(p_graph), "inconsistent PQ state after init", assert::heavy);
  return static_cast<EdgeWeight>(total_external_weight / 2);
}

template <typename QueueSelectionPolicy, typename CutAcceptancePolicy, typename StoppingPolicy>
EdgeWeight InitialFMRefiner<QueueSelectionPolicy, CutAcceptancePolicy, StoppingPolicy>::insert_node(
    const PartitionedCSRGraph &p_graph, const NodeID u
) {
  const BlockID u_block = p_graph.block(u);
  EdgeWeight weighted_external_degree = 0;
  const auto &g = *_graph;
  const EdgeID e_begin = g.raw_nodes()[u];
  const EdgeID e_end = g.raw_nodes()[u + 1];
  const NodeID *__restrict__ edges = g.raw_edges().data();
  const BlockID *__restrict__ part = p_graph.raw_partition().data();

  if (g.is_edge_weighted()) {
    const EdgeWeight *__restrict__ ew = g.raw_edge_weights().data();
    for (EdgeID e = e_begin; e < e_end; ++e) {
      weighted_external_degree += (part[edges[e]] != u_block) ? ew[e] : 0;
    }
  } else {
    for (EdgeID e = e_begin; e < e_end; ++e) {
      weighted_external_degree += (part[edges[e]] != u_block);
    }
  }

  if (weighted_external_degree != 0) {
    const EdgeWeight gain = _weighted_degrees[u] - 2 * weighted_external_degree;
    _queues[u_block].push(u, gain);
  }
  return weighted_external_degree;
}

template <typename QueueSelectionPolicy, typename CutAcceptancePolicy, typename StoppingPolicy>
EdgeWeight InitialFMRefiner<QueueSelectionPolicy, CutAcceptancePolicy, StoppingPolicy>::
    compute_gain_from_scratch(const PartitionedCSRGraph &p_graph, const NodeID u) {
  const BlockID u_block = p_graph.block(u);
  EdgeWeight weighted_external_degree = 0;
  p_graph.graph().adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
    weighted_external_degree += (p_graph.block(v) != u_block) * weight;
  });
  const EdgeWeight weighted_internal_degree = _weighted_degrees[u] - weighted_external_degree;
  return weighted_internal_degree - weighted_external_degree;
}

template <typename QueueSelectionPolicy, typename CutAcceptancePolicy, typename StoppingPolicy>
void InitialFMRefiner<QueueSelectionPolicy, CutAcceptancePolicy, StoppingPolicy>::
    init_weighted_degrees() {
  const NodeID n = _graph->n();
  const EdgeID *__restrict__ nodes = _graph->raw_nodes().data();
  if (!_graph->is_edge_weighted()) {
    for (NodeID u = 0; u < n; ++u) {
      _weighted_degrees[u] = static_cast<EdgeWeight>(nodes[u + 1] - nodes[u]);
    }
    return;
  }

  const EdgeWeight *__restrict__ ew = _graph->raw_edge_weights().data();
  for (NodeID u = 0; u < n; ++u) {
    EdgeWeight weighted_degree = 0;
    for (EdgeID e = nodes[u]; e < nodes[u + 1]; ++e) {
      weighted_degree += ew[e];
    }
    _weighted_degrees[u] = weighted_degree;
  }
}

template <typename QueueSelectionPolicy, typename CutAcceptancePolicy, typename StoppingPolicy>
bool InitialFMRefiner<QueueSelectionPolicy, CutAcceptancePolicy, StoppingPolicy>::is_boundary_node(
    const PartitionedCSRGraph &p_graph, const NodeID u
) {
  bool boundary_node = false;
  p_graph.graph().adjacent_nodes(u, [&](const NodeID v) {
    if (p_graph.block(u) != p_graph.block(v)) {
      boundary_node = true;
      return true;
    }

    return false;
  });

  return boundary_node;
}

template <typename QueueSelectionPolicy, typename CutAcceptancePolicy, typename StoppingPolicy>
bool InitialFMRefiner<QueueSelectionPolicy, CutAcceptancePolicy, StoppingPolicy>::validate_pqs(
    const PartitionedCSRGraph &p_graph
) {
  for (const NodeID u : p_graph.graph().nodes()) {
    if (is_boundary_node(p_graph, u)) {
      if (_marker.get(u)) {
        KASSERT(!_queues[0].contains(u));
        KASSERT(!_queues[1].contains(u));
      } else {
        KASSERT(_queues[p_graph.block(u)].contains(u));
        KASSERT(!_queues[1 - p_graph.block(u)].contains(u));
        KASSERT(_queues[p_graph.block(u)].key(u) == compute_gain_from_scratch(p_graph, u));
      }
    } else {
      KASSERT(!_queues[0].contains(u));
      KASSERT(!_queues[1].contains(u));
    }
  }

  return true;
}

template class InitialFMRefiner<
    fm::MaxOverloadSelectionPolicy,
    fm::BalancedMinCutAcceptancePolicy,
    fm::SimpleStoppingPolicy>;
template class InitialFMRefiner<
    fm::MaxOverloadSelectionPolicy,
    fm::BalancedMinCutAcceptancePolicy,
    fm::AdaptiveStoppingPolicy>;

} // namespace kaminpar::shm
