/*******************************************************************************
 * Utility functions to take copies of partitions and re-apply them if desired.
 *
 * @file:   snapshooter.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2023
 ******************************************************************************/
#include "kaminpar-dist/refinement/snapshooter.h"

#include <tbb/parallel_invoke.h>

#include "kaminpar-dist/metrics.h"

namespace kaminpar::dist {
BestPartitionSnapshooter::BestPartitionSnapshooter(const NodeID max_total_n, const BlockID max_k)
    : _best_partition(max_total_n),
      _best_block_weights(max_k) {}

void BestPartitionSnapshooter::init(
    const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
) {
  copy_partition(p_graph, p_ctx);
  _best_cut = metrics::edge_cut(p_graph);
  _best_l1 = metrics::imbalance_l1(p_graph, p_ctx);
  _last_is_best = true;
}

void BestPartitionSnapshooter::rollback(DistributedPartitionedGraph &p_graph) {
  if (!_last_is_best) {
    tbb::parallel_invoke(
        [&] {
          p_graph.pfor_all_nodes([&](const NodeID u) {
            p_graph.set_block<false>(u, _best_partition[u]);
          });
        },
        [&] {
          p_graph.pfor_blocks([&](const BlockID b) {
            p_graph.set_block_weight(b, _best_block_weights[b]);
          });
        }
    );
  }
}

void BestPartitionSnapshooter::update(
    const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
) {
  const EdgeWeight current_cut = metrics::edge_cut(p_graph);
  const double current_l1 = metrics::imbalance_l1(p_graph, p_ctx);
  update(p_graph, p_ctx, current_cut, current_l1);
}

void BestPartitionSnapshooter::update(
    const DistributedPartitionedGraph &p_graph,
    const PartitionContext &p_ctx,
    const EdgeWeight cut,
    const double l1
) {
  // Accept if the previous best partition is imbalanced and we improved its balance
  // OR if we are balanced and got a better cut than before
  if ((_best_l1 > 0 && l1 < _best_l1) || (l1 == 0 && cut <= _best_cut)) {
    copy_partition(p_graph, p_ctx);
    _best_cut = cut;
    _best_l1 = l1;
    _last_is_best = true;
  } else {
    _last_is_best = false;
  }
}

void BestPartitionSnapshooter::copy_partition(
    const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
) {
  tbb::parallel_invoke(
      [&] {
        p_graph.pfor_all_nodes([&](const NodeID u) { _best_partition[u] = p_graph.block(u); });
      },
      [&] {
        p_graph.pfor_blocks([&](const BlockID b) {
          _best_block_weights[b] = p_graph.block_weight(b);
        });
      }
  );
}

void DummyPartitionSnapshooter::init(
    const DistributedPartitionedGraph & /* p_graph */, const PartitionContext & /* p_ctx */
) {}

void DummyPartitionSnapshooter::update(
    const DistributedPartitionedGraph & /* p_graph */, const PartitionContext & /* p_ctx */
) {}

void DummyPartitionSnapshooter::update(
    const DistributedPartitionedGraph & /* p_graph */,
    const PartitionContext & /* p_ctx */,
    EdgeWeight /* cut */,
    double /* l1 */
) {}

void DummyPartitionSnapshooter::rollback(DistributedPartitionedGraph & /* p_graph */) {}
} // namespace kaminpar::dist
