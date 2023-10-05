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
BestPartitionSnapshooter::BestPartitionSnapshooter(
    DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
)
    : _p_graph(p_graph),
      _p_ctx(p_ctx),
      _best_cut(metrics::edge_cut(_p_graph)),
      _best_l1(metrics::imbalance_l1(_p_graph, _p_ctx)),
      _best_partition(_p_graph.total_n()),
      _best_block_weights(_p_graph.k()) {
  copy_partition();
}

void BestPartitionSnapshooter::rollback() {
  if (!_last_is_best) {
    tbb::parallel_invoke(
        [&] {
          _p_graph.pfor_all_nodes([&](const NodeID u) {
            _p_graph.set_block<false>(u, _best_partition[u]);
          });
        },
        [&] {
          _p_graph.pfor_blocks([&](const BlockID b) {
            _p_graph.set_block_weight(b, _best_block_weights[b]);
          });
        }
    );
  }
}

void BestPartitionSnapshooter::update() {
  const EdgeWeight current_cut = metrics::edge_cut(_p_graph);
  const double current_l1 = metrics::imbalance_l1(_p_graph, _p_ctx);

  // Accept if the previous best partition is imbalanced and we improved its balance
  // OR if we are balanced and got a better cut than before
  if ((_best_l1 > 0 && current_l1 < _best_l1) || (current_l1 == 0 && current_cut <= _best_cut)) {
    copy_partition();
    _best_cut = current_cut;
    _best_l1 = current_l1;
    _last_is_best = true;
  } else {
    _last_is_best = false;
  }
}

void BestPartitionSnapshooter::copy_partition() {
  tbb::parallel_invoke(
      [&] {
        _p_graph.pfor_all_nodes([&](const NodeID u) { _best_partition[u] = _p_graph.block(u); });
      },
      [&] {
        _p_graph.pfor_blocks([&](const BlockID b) {
          _best_block_weights[b] = _p_graph.block_weight(b);
        });
      }
  );
}

void DummyPartitionSnapshooter::update() {}

void DummyPartitionSnapshooter::rollback() {}
} // namespace kaminpar::dist
