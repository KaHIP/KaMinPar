/*******************************************************************************
 * @file:   initial_refiner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Sequential local improvement graphutils used to improve an initial
 * partition.
 ******************************************************************************/
#pragma once

#include <algorithm>

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/definitions.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/refiner.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/binary_heap.h"
#include "kaminpar-common/datastructures/marker.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::ip {
using Queues = std::array<BinaryMinHeap<EdgeWeight>, 2>;

class InitialRefiner {
public:
  struct MemoryContext {
    Queues queues{BinaryMinHeap<EdgeWeight>{0}, BinaryMinHeap<EdgeWeight>{0}};
    Marker<> marker{0};
    std::vector<EdgeWeight> weighted_degrees;

    void resize(const NodeID n) {
      if (queues[0].capacity() < n) {
        queues[0].resize(n);
      }
      if (queues[1].capacity() < n) {
        queues[1].resize(n);
      }
      if (marker.size() < n) {
        marker.resize(n);
      }
      if (weighted_degrees.size() < n) {
        weighted_degrees.resize(n);
      }
    }

    [[nodiscard]] std::size_t memory_in_kb() const {
      return marker.memory_in_kb() +                               //
             weighted_degrees.size() * sizeof(EdgeWeight) / 1000 + //
             queues[0].memory_in_kb() + queues[1].memory_in_kb();  //
    }
  };

  virtual ~InitialRefiner() = default;

  virtual void initialize(const Graph &graph) = 0;

  virtual bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) = 0;

  virtual MemoryContext free() = 0;
};

class InitialNoopRefiner : public InitialRefiner {
public:
  explicit InitialNoopRefiner(MemoryContext m_ctx) : _m_ctx{std::move(m_ctx)} {}

  void initialize(const Graph &) final {}

  bool refine(PartitionedGraph &, const PartitionContext &) final {
    return false;
  }

  MemoryContext free() override {
    return std::move(_m_ctx);
  }

private:
  MemoryContext _m_ctx;
};

namespace fm {
struct SimpleStoppingPolicy {
  void init(const Graph *) const {}
  [[nodiscard]] bool should_stop(const InitialRefinementContext &fm_ctx) const {
    return _num_steps > fm_ctx.num_fruitless_moves;
  }
  void reset() {
    _num_steps = 0;
  }
  void update(const EdgeWeight) {
    ++_num_steps;
  }

private:
  std::size_t _num_steps{0};
};

// "Adaptive" random walk stopping policy
// Implementation copied from: KaHyPar -> AdvancedRandomWalkModelStopsSearch,
// Copyright (C) Sebastian Schlag
struct AdaptiveStoppingPolicy {
  void init(const Graph *graph) {
    _beta = std::sqrt(graph->n());
  }

  [[nodiscard]] bool should_stop(const InitialRefinementContext &fm_ctx) const {
    const double factor = (fm_ctx.alpha / 2.0) - 0.25;
    return (_num_steps > _beta) &&
           ((_Mk == 0) || (_num_steps >= (_variance / (_Mk * _Mk)) * factor));
  }

  void reset() {
    _num_steps = 0;
    _variance = 0.0;
  }

  void update(const EdgeWeight gain) {
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

//! Always move the next node from the heavier block. This should improve
//! balance.
struct MaxWeightSelectionPolicy {
  std::size_t operator()(
      const PartitionedGraph &p_graph, const PartitionContext &context, const Queues &, Random &rand
  ) {
    const auto weight0 = p_graph.block_weight(0) - context.block_weights.perfectly_balanced(0);
    const auto weight1 = p_graph.block_weight(1) - context.block_weights.perfectly_balanced(1);
    return weight1 > weight0 || (weight0 == weight1 && rand.random_bool());
  }
};

//! Always select the node with the highest gain / lowest loss.
struct MaxGainSelectionPolicy {
  std::size_t operator()(
      const PartitionedGraph &p_graph,
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
      const PartitionedGraph &p_graph,
      const PartitionContext &context,
      const Queues &queues,
      Random &rand
  ) {
    const NodeWeight overload0 =
        std::max<NodeWeight>(0, p_graph.block_weight(0) - context.block_weights.max(0));
    const NodeWeight overload1 =
        std::max<NodeWeight>(0, p_graph.block_weight(1) - context.block_weights.max(1));
    if (overload0 == 0 && overload1 == 0) {
      return MaxGainSelectionPolicy()(p_graph, context, queues, rand);
    }
    return overload1 > overload0 || (overload1 == overload0 && rand.random_bool());
  }
};

//! Accept better cuts, or the first cut that is balanced in case the initial
//! cut is not balanced.
struct BalancedMinCutAcceptancePolicy {
  bool operator()(
      const PartitionedGraph &,
      const PartitionContext &,
      const EdgeWeight accepted_overload,
      const EdgeWeight current_overload,
      const EdgeWeight accepted_delta,
      const EdgeWeight delta
  ) {
    return current_overload <= accepted_overload && delta < accepted_delta;
  }
};
} // namespace fm

/*!
 * 2-way FM refinement graphutils that uses two priority queues, one for each
 * block. A round of local search is stopped after 350 fruitless moves, i.e.,
 * moves that did not lead to an accepted cut; or after every node was tried to
 * be moved once.
 *
 * @tparam QueueSelectionPolicy Selects the next block from where we move a
 * node.
 * @tparam CutAcceptancePolicy Decides whether we accept the current cut.
 */
template <typename QueueSelectionPolicy, typename CutAcceptancePolicy, typename StoppingPolicy>
class InitialTwoWayFMRefiner : public InitialRefiner {
  static constexpr NodeID kChunkSize = 64;
  static constexpr std::size_t kNumberOfNodePermutations = 32;

  static constexpr bool kDebug = false;

public:
  InitialTwoWayFMRefiner(
      const NodeID n,
      const PartitionContext &p_ctx,
      const InitialRefinementContext &r_ctx,
      MemoryContext m_ctx = {}
  )
      : _p_ctx(p_ctx),
        _r_ctx(r_ctx),
        _queues(std::move(m_ctx.queues)), //
        _marker(std::move(m_ctx.marker)),
        _weighted_degrees(std::move(m_ctx.weighted_degrees)) {
    KASSERT(
        p_ctx.k == 2u,
        "2-way refiner cannot be used on a " << p_ctx.k << "-way partition" << assert::light
    );

    if (_queues[0].capacity() < n) {
      _queues[0].resize(n);
    }
    if (_queues[1].capacity() < n) {
      _queues[1].resize(n);
    }
    if (_marker.capacity() < n) {
      _marker.resize(n);
    }
    if (_weighted_degrees.size() < n) {
      _weighted_degrees.resize(n);
    }
  }

  void initialize(const Graph &graph) final {
    KASSERT(_queues[0].capacity() >= graph.n());
    KASSERT(_queues[1].capacity() >= graph.n());
    KASSERT(_marker.capacity() >= graph.n());
    KASSERT(_weighted_degrees.capacity() >= graph.n());

    _graph = &graph;
    _stopping_policy.init(_graph);

    init_weighted_degrees();
  }

  bool refine(PartitionedGraph &p_graph, const PartitionContext &) final {
    KASSERT(&p_graph.graph() == _graph, "must be initialized with the same graph", assert::light);
    KASSERT(
        p_graph.k() == 2u,
        "2-way refiner cannot be used on a " << p_graph.k() << "-way partition",
        assert::light
    );

    const EdgeWeight initial_edge_cut = metrics::edge_cut_seq(p_graph);
    if (initial_edge_cut == 0) {
      return false;
    } // no improvement possible

    EdgeWeight prev_edge_cut = initial_edge_cut;
    EdgeWeight cur_edge_cut = prev_edge_cut;

    cur_edge_cut += round(p_graph); // always do at least one round
    for (std::size_t it = 1;
         0 < cur_edge_cut && it < _r_ctx.num_iterations && !abort(prev_edge_cut, cur_edge_cut);
         ++it) {
      prev_edge_cut = cur_edge_cut;
      cur_edge_cut += round(p_graph);
    }

    return cur_edge_cut < initial_edge_cut;
  }

  MemoryContext free() final {
    return {
        .queues = std::move(_queues),
        .marker = std::move(_marker),
        .weighted_degrees = std::move(_weighted_degrees)};
  }

private:
  [[nodiscard]] bool
  abort(const EdgeWeight prev_edge_weight, const EdgeWeight cur_edge_weight) const {
    return (1.0 - 1.0 * cur_edge_weight / prev_edge_weight) < _r_ctx.improvement_abortion_threshold;
  }

  /*!
   * Performs one round of local search that is stopped after a configurable
   * number of fruitless moves.
   *
   * @param p_graph Partition of #_graph.
   * @return Whether we were able to improve the cut.
   */
  EdgeWeight round(PartitionedGraph &p_graph) {
    KASSERT(p_graph.k() == 2u, "2-way FM with " << p_graph.k() << "-way partition", assert::light);
    DBG << "Initial refiner initialized with n=" << p_graph.n() << " m=" << p_graph.m();

#if KASSERT_ENABLED(ASSERTION_LEVEL_HEAVY)
    const bool initially_feasible = metrics::is_feasible(p_graph, _p_ctx);
#endif

    _stopping_policy.reset();

    init_pq(p_graph);

    std::vector<NodeID> moves; // moves since last accepted cut
    std::size_t active = 0;    // block from which we select a node for movement

    EdgeWeight current_overload = metrics::total_overload(p_graph, _p_ctx);
    EdgeWeight accepted_overload = current_overload;

    EdgeWeight current_delta = 0;
    EdgeWeight accepted_delta = 0;
#if KASSERT_ENABLED(ASSERTION_LEVEL_HEAVY)
    const EdgeWeight initial_edge_cut = metrics::edge_cut(p_graph);
#endif

    DBG << "Starting main refinement loop with #_pq[0]=" << _queues[0].size()
        << " #_pq[1]=" << _queues[1].size();

    while ((!_queues[0].empty() || !_queues[1].empty()) && !_stopping_policy.should_stop(_r_ctx)) {
#if KASSERT_ENABLED(ASSERTION_LEVEL_HEAVY)
      validate_pqs(p_graph);
#endif

      active = _queue_selection_policy(p_graph, _p_ctx, _queues, _rand);
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
      moves.push_back(u);
#if KASSERT_ENABLED(ASSERTION_LEVEL_HEAVY)
      KASSERT(initial_edge_cut + current_delta == metrics::edge_cut(p_graph), "", assert::heavy);
#endif
      _stopping_policy.update(-delta); // assumes gain, not loss
      current_overload = metrics::total_overload(p_graph, _p_ctx);

      // update gain of neighboring nodes
      for (const auto [e, v] : _graph->neighbors(u)) {
        if (_marker.get(v)) {
          continue;
        }

        const EdgeWeight e_weight = _graph->edge_weight(e);
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
      }

      // accept move if it improves the best edge cut found so far
      if (_cut_acceptance_policy(
              p_graph, _p_ctx, accepted_overload, current_overload, accepted_delta, current_delta
          )) {
        DBG << "Accepted new bipartition: delta=" << current_delta
            << " cut=" << metrics::edge_cut_seq(p_graph);
        _stopping_policy.reset();
        accepted_delta = current_delta;
        accepted_overload = current_overload;
        moves.clear();
      }
    }

    // rollback to last accepted cut
    for (const NodeID u : moves) {
      p_graph.set_block(u, 1 - p_graph.block(u));
    };

    // reset datastructures for next run
    for (const std::size_t i : {0, 1}) {
      _queues[i].clear();
    }
    _marker.reset();

#if KASSERT_ENABLED(ASSERTION_LEVEL_HEAVY)
    KASSERT((!initially_feasible || accepted_delta <= 0)
    ); // only accept bad cuts when starting with bad balance
    KASSERT(metrics::edge_cut(p_graph) == initial_edge_cut + accepted_delta);
#endif

    return accepted_delta;
  }

  void init_pq(const PartitionedGraph &p_graph) {
    KASSERT(_queues[0].empty());
    KASSERT(_queues[1].empty());

    const std::size_t num_chunks = _graph->n() / kChunkSize + 1;

    std::vector<std::size_t> chunks(num_chunks);
    std::iota(chunks.begin(), chunks.end(), 0);
    std::transform(chunks.begin(), chunks.end(), chunks.begin(), [](const std::size_t i) {
      return i * kChunkSize;
    });
    _rand.shuffle(chunks);

    for (const std::size_t chunk : chunks) {
      const auto &permutation = _permutations.get(_rand);
      for (const NodeID i : permutation) {
        const NodeID u = chunk + i;
        if (u < _graph->n()) {
          insert_node(p_graph, u);
        }
      }
    }

#if KASSERT_ENABLED(ASSERTION_LEVEL_HEAVY)
    validate_pqs(p_graph);
#endif
  }

  void insert_node(const PartitionedGraph &p_graph, const NodeID u) {
    const EdgeWeight gain = compute_gain_from_scratch(p_graph, u);
    const BlockID u_block = p_graph.block(u);
    if (_weighted_degrees[u] != gain) {
      _queues[u_block].push(u, gain);
    }
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
      for (const EdgeID e : _graph->incident_edges(u)) {
        weighted_degree += _graph->edge_weight(e);
      }
      _weighted_degrees[u] = weighted_degree;
    }
  }

  bool is_boundary_node(const PartitionedGraph &p_graph, const NodeID u) {
    for (const NodeID v : p_graph.adjacent_nodes(u)) {
      if (p_graph.block(u) != p_graph.block(v)) {
        return true;
      }
    }
    return false;
  }

  void validate_pqs(const PartitionedGraph &p_graph) {
    for (const NodeID u : p_graph.nodes()) {
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
  }

  const Graph *_graph; //! Graph for refinement, partition to refine is passed
                       //! to #refine().
  const PartitionContext &_p_ctx;
  const InitialRefinementContext &_r_ctx;
  Queues _queues{BinaryMinHeap<EdgeWeight>{0}, BinaryMinHeap<EdgeWeight>{0}};
  Marker<> _marker{0};
  std::vector<EdgeWeight> _weighted_degrees{};
  QueueSelectionPolicy _queue_selection_policy{};
  CutAcceptancePolicy _cut_acceptance_policy{};
  StoppingPolicy _stopping_policy{};
  Random &_rand{Random::instance()};
  RandomPermutations<NodeID, kChunkSize, kNumberOfNodePermutations> _permutations;
};

extern template class InitialTwoWayFMRefiner<
    fm::MaxOverloadSelectionPolicy,
    fm::BalancedMinCutAcceptancePolicy,
    fm::SimpleStoppingPolicy>;
extern template class InitialTwoWayFMRefiner<
    fm::MaxOverloadSelectionPolicy,
    fm::BalancedMinCutAcceptancePolicy,
    fm::AdaptiveStoppingPolicy>;

using InitialSimple2WayFM = InitialTwoWayFMRefiner<
    fm::MaxOverloadSelectionPolicy,
    fm::BalancedMinCutAcceptancePolicy,
    fm::SimpleStoppingPolicy>;
using InitialAdaptive2WayFM = InitialTwoWayFMRefiner<
    fm::MaxOverloadSelectionPolicy,
    fm::BalancedMinCutAcceptancePolicy,
    fm::AdaptiveStoppingPolicy>;

inline std::unique_ptr<InitialRefiner> create_initial_refiner(
    const Graph &graph,
    const PartitionContext &p_ctx,
    const InitialRefinementContext &r_ctx,
    InitialRefiner::MemoryContext m_ctx
) {
  if (!r_ctx.disabled) {
    switch (r_ctx.stopping_rule) {
    case FMStoppingRule::ADAPTIVE:
      return std::make_unique<InitialSimple2WayFM>(graph.n(), p_ctx, r_ctx, std::move(m_ctx));

    case FMStoppingRule::SIMPLE:
      return std::make_unique<InitialAdaptive2WayFM>(graph.n(), p_ctx, r_ctx, std::move(m_ctx));
    }
  }

  return std::make_unique<InitialNoopRefiner>(std::move(m_ctx));
}
} // namespace kaminpar::shm::ip
