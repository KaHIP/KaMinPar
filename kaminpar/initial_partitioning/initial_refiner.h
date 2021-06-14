/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2020 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#pragma once

#include "context.h"
#include "datastructure/binary_heap.h"
#include "datastructure/graph.h"
#include "datastructure/marker.h"
#include "definitions.h"
#include "refinement/i_refiner.h"
#include "utility/metrics.h"
#include "utility/random.h"
#include "utility/timer.h"

#include <algorithm>

namespace kaminpar::ip {
using Queues = std::array<BinaryMinHeap<Gain>, 2>;

class InitialRefiner : public Refiner {
public:
  struct MemoryContext {
    Queues queues{BinaryMinHeap<Gain>{0}, BinaryMinHeap<Gain>{0}};
    Marker<> marker{0};
    std::vector<EdgeWeight> weighted_degrees;

    void resize(const NodeID n) {
      if (queues[0].capacity() < n) { queues[0].resize(n); }
      if (queues[1].capacity() < n) { queues[1].resize(n); }
      if (marker.size() < n) { marker.resize(n); }
      if (weighted_degrees.size() < n) { weighted_degrees.resize(n); }
    }

    std::size_t memory_in_kb() const {
      return marker.memory_in_kb() +                               //
             weighted_degrees.size() * sizeof(EdgeWeight) / 1000 + //
             queues[0].memory_in_kb() + queues[1].memory_in_kb();  //
    }
  };

  NodeWeight expected_total_gain() const final {
    ASSERT(false) << "not implemented";
    return 0;
  }

  virtual MemoryContext free() = 0;
};

class InitialNoopRefiner : public InitialRefiner {
public:
  InitialNoopRefiner(MemoryContext m_ctx) : _m_ctx{std::move(m_ctx)} {}

  void initialize(const Graph &) final {}
  bool refine(PartitionedGraph &, const PartitionContext &) final { return false; }
  MemoryContext free() { return std::move(_m_ctx); }

private:
  MemoryContext _m_ctx;
};

namespace fm {
struct SimpleStoppingPolicy {
  void init(const Graph *) const {}
  bool should_stop(const FMRefinementContext &fm_ctx) const { return _num_steps > fm_ctx.num_fruitless_moves; }
  void reset() { _num_steps = 0; }
  void update(const Gain) { ++_num_steps; }

private:
  std::size_t _num_steps{0};
};

// "Adaptive" random walk stopping policy
// Implementation copied from: KaHyPar -> AdvancedRandomWalkModelStopsSearch, Copyright (C) Sebastian Schlag
struct AdaptiveStoppingPolicy {
  void init(const Graph *graph) { _beta = std::sqrt(graph->n()); }

  bool should_stop(const FMRefinementContext &fm_ctx) const {
    const double factor = (fm_ctx.alpha / 2.0) - 0.25;
    return (_num_steps > _beta) && ((_Mk == 0) || (_num_steps >= (_variance / (_Mk * _Mk)) * factor));
  }

  void reset() {
    _num_steps = 0;
    _variance = 0.0;
  }

  void update(const Gain gain) {
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

      // prepare for next iteration:
      _MkMinus1 = _Mk;
      _SkMinus1 = _Sk;
    }
  }

private:
  double _beta{};
  std::size_t _num_steps{0};
  double _variance{0.0};
  double _Mk{0.0};
  double _MkMinus1{0.0};
  double _Sk{0.0};
  double _SkMinus1{0.0};
};

//! Always move the next node from the heavier block. This should improve balance.
struct MaxWeightSelectionPolicy {
  std::size_t operator()(const PartitionedGraph &p_graph, const PartitionContext &context, const Queues &,
                         Randomize &rand) {
    const auto weight0 = p_graph.block_weight(0) - context.perfectly_balanced_block_weight(0);
    const auto weight1 = p_graph.block_weight(1) - context.perfectly_balanced_block_weight(1);
    return weight1 > weight0 || (weight0 == weight1 && rand.random_bool());
  }
};

//! Always select the node with the highest gain / lowest loss.
struct MaxGainSelectionPolicy {
  std::size_t operator()(const PartitionedGraph &p_graph, const PartitionContext &context, const Queues &queues,
                         Randomize &rand) {
    const auto loss0 = queues[0].empty() ? std::numeric_limits<Gain>::max() : queues[0].peek_key();
    const auto loss1 = queues[1].empty() ? std::numeric_limits<Gain>::max() : queues[1].peek_key();
    if (loss0 == loss1) { return MaxWeightSelectionPolicy()(p_graph, context, queues, rand); }
    return loss1 < loss0;
  }
};

struct MaxOverloadSelectionPolicy {
  std::size_t operator()(const PartitionedGraph &p_graph, const PartitionContext &context, const Queues &queues,
                         Randomize &rand) {
    const NodeWeight overload0 = std::max(0, p_graph.block_weight(0) - context.max_block_weight(0));
    const NodeWeight overload1 = std::max(0, p_graph.block_weight(1) - context.max_block_weight(1));
    if (overload0 == 0 && overload1 == 0) { return MaxGainSelectionPolicy()(p_graph, context, queues, rand); }
    return overload1 > overload0 || (overload1 == overload0 && rand.random_bool());
  }
};

//! Accept better cuts, or the first cut that is balanced in case the initial cut is not balanced.
struct BalancedMinCutAcceptancePolicy {
  bool operator()(const PartitionedGraph &, const PartitionContext &, const EdgeWeight accepted_overload,
                  const EdgeWeight current_overload, const Gain accepted_delta, const Gain delta) {
    return current_overload <= accepted_overload && delta < accepted_delta;
  }
};
} // namespace fm

/*!
 * 2-way FM refinement algorithm that uses two priority queues, one for each block. A round of local search is stopped
 * after 350 fruitless moves, i.e., moves that did not lead to an accepted cut; or after every node was tried to be
 * moved once.
 *
 * @tparam QueueSelectionPolicy Selects the next block from where we move a node.
 * @tparam CutAcceptancePolicy Decides whether we accept the current cut.
 */
template<typename QueueSelectionPolicy, typename CutAcceptancePolicy, typename StoppingPolicy>
class InitialTwoWayFMRefiner : public InitialRefiner {
  static constexpr NodeID kChunkSize = 64;
  static constexpr std::size_t kNumberOfNodePermutations = 32;

  static constexpr bool kDebug = false;

public:
  InitialTwoWayFMRefiner(const NodeID n, const PartitionContext &p_ctx, const RefinementContext &r_ctx,
                         MemoryContext m_ctx = {})
      : _p_ctx{p_ctx},
        _r_ctx{r_ctx},
        _queues{std::move(m_ctx.queues)}, //
        _marker{std::move(m_ctx.marker)},
        _weighted_degrees{std::move(m_ctx.weighted_degrees)} {
    ALWAYS_ASSERT(p_ctx.k == 2) << "2-way refiner cannot be used on a " << p_ctx.k << "-way partition.";

    if (_queues[0].capacity() < n) { _queues[0].resize(n); }
    if (_queues[1].capacity() < n) { _queues[1].resize(n); }
    if (_marker.capacity() < n) { _marker.resize(n); }
    if (_weighted_degrees.size() < n) { _weighted_degrees.resize(n); }
  }

  void initialize(const Graph &graph) final {
    ASSERT(_queues[0].capacity() >= graph.n());
    ASSERT(_queues[1].capacity() >= graph.n());
    ASSERT(_marker.capacity() >= graph.n());
    ASSERT(_weighted_degrees.capacity() >= graph.n());
    _graph = &graph;
    _stopping_policy.init(_graph);
    init_weighted_degrees();
  }

  bool refine(PartitionedGraph &p_graph, const PartitionContext &) final {
    ASSERT(&p_graph.graph() == _graph) << "Must be initialized with the same graph";
    ASSERT(p_graph.k() == 2) << "2-way refiner cannot be used on a " << p_graph.k() << "-way partition.";

    const EdgeWeight initial_edge_cut = metrics::edge_cut(p_graph, tag::seq);
    if (initial_edge_cut == 0) { return false; } // no improvement possible

    EdgeWeight prev_edge_cut = initial_edge_cut;
    EdgeWeight cur_edge_cut = prev_edge_cut;

    cur_edge_cut += round(p_graph); // always do at least one round
    for (std::size_t it = 1; 0 < cur_edge_cut && it < _r_ctx.fm.num_iterations && !abort(prev_edge_cut, cur_edge_cut);
         ++it) {
      prev_edge_cut = cur_edge_cut;
      cur_edge_cut += round(p_graph);
    }

    return cur_edge_cut < initial_edge_cut;
  }

  MemoryContext free() final {
    return {.queues = std::move(_queues),
            .marker = std::move(_marker),
            .weighted_degrees = std::move(_weighted_degrees)};
  }

private:
  bool abort(const EdgeWeight prev_edge_weight, const EdgeWeight cur_edge_weight) const {
    return (1.0 - 1.0 * cur_edge_weight / prev_edge_weight) < _r_ctx.fm.improvement_abortion_threshold;
  }

  /*!
   * Performs one round of local search that is stopped after a configurable number of fruitless moves.
   *
   * @param p_graph Partition of #_graph.
   * @return Whether we were able to improve the cut.
   */
  EdgeWeight round(PartitionedGraph &p_graph) {
    ASSERT(p_graph.k() == 2) << "2-way FM with " << p_graph.k() << "-way partition";
    DBG << "Initial refiner initialized with n=" << p_graph.n() << " m=" << p_graph.m();

#ifdef KAMINPAR_ENABLE_ASSERTIONS
    const bool initially_feasible = metrics::is_feasible(p_graph, _p_ctx);
#endif // KAMINPAR_ENABLE_ASSERTIONS

    _stopping_policy.reset();

    init_pq(p_graph);

    std::vector<NodeID> moves; // moves since last accepted cut
    std::size_t active = 0;    // block from which we select a node for movement

    EdgeWeight current_overload = metrics::total_overload(p_graph, _p_ctx);
    EdgeWeight accepted_overload = current_overload;

    Gain current_delta = 0;
    Gain accepted_delta = 0;
#ifdef KAMINPAR_ENABLE_ASSERTIONS
    const EdgeWeight initial_edge_cut = metrics::edge_cut(p_graph, tag::seq);
#endif // KAMINPAR_ENABLE_ASSERTIONS

    DBG << "Starting main refinement loop with #_pq[0]=" << _queues[0].size() << " #_pq[1]=" << _queues[1].size();

    while ((!_queues[0].empty() || !_queues[1].empty()) && !_stopping_policy.should_stop(_r_ctx.fm)) {
#ifdef KAMINPAR_ENABLE_HEAVY_ASSERTIONS
      VALIDATE_PQS(p_graph);
#endif // KAMINPAR_ENABLE_HEAVY_ASSERTIONS

      active = _queue_selection_policy(p_graph, _p_ctx, _queues, _rand);
      if (_queues[active].empty()) { active = 1 - active; }
      BinaryMinHeap<Gain> &queue = _queues[active];

      const NodeID u = queue.peek_id();
      const Gain delta = queue.peek_key();
      const BlockID from = active;
      const BlockID to = 1 - from;
      ASSERT(!_marker.get(u));
      ASSERT(from == p_graph.block(u));
      _marker.set(u);
      queue.pop();

      DBG << "Performed move, new cut=" << metrics::edge_cut(p_graph, tag::seq);
      p_graph.set_block(u, to);
      current_delta += delta;
      moves.push_back(u);
      ASSERT(initial_edge_cut + current_delta == metrics::edge_cut(p_graph, tag::seq));
      _stopping_policy.update(-delta); // assumes gain, not loss
      current_overload = metrics::total_overload(p_graph, _p_ctx);

      // update gain of neighboring nodes
      for (const auto [e, v] : _graph->neighbors(u)) {
        if (_marker.get(v)) { continue; }

        const EdgeWeight e_weight = _graph->edge_weight(e);
        const BlockID v_block = p_graph.block(v);
        const EdgeWeight loss_delta = 2 * e_weight * ((to == v_block) ? 1 : -1);

        if (_queues[v_block].contains(v)) {
          const EdgeWeight new_loss = _queues[v_block].key(v) + loss_delta;
          const bool still_boundary_node = new_loss < _weighted_degrees[v];

          if (!still_boundary_node) { // v is no longer a boundary node
            HEAVY_ASSERT(!IS_BOUNDARY_NODE(p_graph, v));
            _queues[v_block].remove(v);
          } else { // v is still a boundary node
            HEAVY_ASSERT(IS_BOUNDARY_NODE(p_graph, v));
            _queues[v_block].change_priority(v, new_loss);
          }
        } else { // since v was not a boundary node before, it must be one now
          HEAVY_ASSERT(IS_BOUNDARY_NODE(p_graph, v));
          _queues[v_block].push(v, _weighted_degrees[v] + loss_delta);
        }
      }

      // accept move if it improves the best edge cut found so far
      if (_cut_acceptance_policy(p_graph, _p_ctx, accepted_overload, current_overload, accepted_delta, current_delta)) {
        DBG << "Accepted new bipartition: delta=" << current_delta << " cut=" << metrics::edge_cut(p_graph, tag::seq);
        _stopping_policy.reset();
        accepted_delta = current_delta;
        accepted_overload = current_overload;
        moves.clear();
      }
    }

    // rollback to last accepted cut
    for (const NodeID u : moves) { p_graph.set_block(u, 1 - p_graph.block(u)); };

    // reset datastructures for next run
    for (const std::size_t i : {0, 1}) { _queues[i].clear(); }
    _marker.reset();

    ASSERT(!initially_feasible || accepted_delta <= 0); // only accept bad cuts when starting with bad balance
    ASSERT(metrics::edge_cut(p_graph) == initial_edge_cut + accepted_delta) << V(metrics::edge_cut(p_graph, tag::seq));
    return accepted_delta;
  }

  void init_pq(const PartitionedGraph &p_graph) {
    ASSERT(_queues[0].empty() && _queues[1].empty());

    const std::size_t num_chunks = _graph->n() / kChunkSize + 1;

    std::vector<std::size_t> chunks(num_chunks);
    std::iota(chunks.begin(), chunks.end(), 0);
    std::transform(chunks.begin(), chunks.end(), chunks.begin(), [](const std::size_t i) { return i * kChunkSize; });
    _rand.shuffle(chunks);

    for (const std::size_t chunk : chunks) {
      const auto &permutation = _permutations.get(_rand);
      for (const NodeID i : permutation) {
        const NodeID u = chunk + i;
        if (u < _graph->n()) { insert_node(p_graph, u); }
      }
    }

#ifdef KAMINPAR_ENABLE_HEAVY_ASSERTIONS
    VALIDATE_PQS(p_graph);
#endif // KAMINPAR_ENABLE_HEAVY_ASSERTIONS
  }

  void insert_node(const PartitionedGraph &p_graph, const NodeID u) {
    const EdgeWeight gain = compute_gain_from_scratch(p_graph, u);
    const BlockID u_block = p_graph.block(u);
    if (_weighted_degrees[u] != gain) { _queues[u_block].push(u, gain); }
  }

  EdgeWeight compute_gain_from_scratch(const PartitionedGraph &p_graph, const NodeID u) {
    const BlockID u_block = p_graph.block(u);
    EdgeWeight weighted_external_degree = 0;
    for (const auto [e, v] : p_graph.neighbors(u)) {
      weighted_external_degree += (p_graph.block(v) != u_block) * p_graph.edge_weight(e);
    }
    const EdgeWeight weighted_internal_degree = _weighted_degrees[u] - weighted_external_degree;
    return weighted_internal_degree - weighted_external_degree;
  }

  void init_weighted_degrees() {
    for (const NodeID u : _graph->nodes()) {
      EdgeWeight weighted_degree = 0;
      for (const EdgeID e : _graph->incident_edges(u)) { weighted_degree += _graph->edge_weight(e); }
      _weighted_degrees[u] = weighted_degree;
    }
  }

#ifdef KAMINPAR_ENABLE_HEAVY_ASSERTIONS
  bool IS_BOUNDARY_NODE(const PartitionedGraph &p_graph, const NodeID u) {
    for (const NodeID v : p_graph.adjacent_nodes(u)) {
      if (p_graph.block(u) != p_graph.block(v)) { return true; }
    }
    return false;
  }

  void VALIDATE_PQS(const PartitionedGraph &p_graph) {
    for (const NodeID u : p_graph.nodes()) {
      if (IS_BOUNDARY_NODE(p_graph, u)) {
        if (_marker.get(u)) {
          ASSERT(!_queues[0].contains(u));
          ASSERT(!_queues[1].contains(u));
        } else {
          ASSERT(_queues[p_graph.block(u)].contains(u));
          ASSERT(!_queues[1 - p_graph.block(u)].contains(u));
          ASSERT(_queues[p_graph.block(u)].key(u) == compute_gain_from_scratch(p_graph, u));
        }
      } else {
        ASSERT(!_queues[0].contains(u));
        ASSERT(!_queues[1].contains(u));
      }
    }
  }
#endif // KAMINPAR_ENABLE_HEAVY_ASSERTIONS

  const Graph *_graph; //! Graph for refinement, partition to refine is passed to #refine().
  const PartitionContext &_p_ctx;
  const RefinementContext &_r_ctx;
  Queues _queues{BinaryMinHeap<Gain>{0}, BinaryMinHeap<Gain>{0}};
  Marker<> _marker{0};
  std::vector<EdgeWeight> _weighted_degrees{};
  QueueSelectionPolicy _queue_selection_policy{};
  CutAcceptancePolicy _cut_acceptance_policy{};
  StoppingPolicy _stopping_policy{};
  Randomize &_rand{Randomize::instance()};
  RandomPermutations<NodeID, kChunkSize, kNumberOfNodePermutations> _permutations;
};

extern template class InitialTwoWayFMRefiner<fm::MaxOverloadSelectionPolicy, fm::BalancedMinCutAcceptancePolicy,
                                             fm::SimpleStoppingPolicy>;
extern template class InitialTwoWayFMRefiner<fm::MaxOverloadSelectionPolicy, fm::BalancedMinCutAcceptancePolicy,
                                             fm::AdaptiveStoppingPolicy>;

using InitialSimple2WayFM = InitialTwoWayFMRefiner<fm::MaxOverloadSelectionPolicy, fm::BalancedMinCutAcceptancePolicy,
                                                   fm::SimpleStoppingPolicy>;
using InitialAdaptive2WayFM = InitialTwoWayFMRefiner<fm::MaxOverloadSelectionPolicy, fm::BalancedMinCutAcceptancePolicy,
                                                     fm::AdaptiveStoppingPolicy>;
} // namespace kaminpar::ip