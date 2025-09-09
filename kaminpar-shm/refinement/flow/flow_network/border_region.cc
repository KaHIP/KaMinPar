#include "kaminpar-shm/refinement/flow/flow_network/border_region.h"

#include <algorithm>

#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

BorderRegionConstructor::BorderRegionConstructor(
    const PartitionContext &p_ctx,
    const FlowNetworkConstructionContext &c_ctx,
    const QuotientGraph &q_graph,
    const PartitionedCSRGraph &p_graph,
    const CSRGraph &graph
)
    : _p_ctx(p_ctx),
      _c_ctx(c_ctx),
      _q_graph(q_graph),
      _p_graph(p_graph),
      _graph(graph),
      _random(random::thread_independent_seeding) {};

const BorderRegion &BorderRegionConstructor::construct(
    const BlockID block1,
    const BlockID block2,
    const BlockWeight block1_weight,
    const BlockWeight block2_weight
) {
  SCOPED_TIMER("Construct Border Region");

  const BlockWeight max_border_region_weight1 = std::min<BlockWeight>(
      block1_weight,
      (1 + _c_ctx.border_region_scaling_factor * _p_ctx.epsilon()) *
              _p_ctx.perfectly_balanced_block_weight(block2) -
          block2_weight
  );

  const BlockWeight max_border_region_weight2 = std::min<BlockWeight>(
      block2_weight,
      (1 + _c_ctx.border_region_scaling_factor * _p_ctx.epsilon()) *
              _p_ctx.perfectly_balanced_block_weight(block1) -
          block1_weight
  );

  compute_border_regions(block1, block2, max_border_region_weight1, max_border_region_weight2);
  if (!_border_region.empty()) {
    expand_border_regions();
  }

  return _border_region;
}

void BorderRegionConstructor::free() {
  _border_region.free();
  _bfs_runner.free();
}

void BorderRegionConstructor::compute_border_regions(
    const BlockID block1,
    const BlockID block2,
    const BlockWeight max_border_region_weight1,
    const BlockWeight max_border_region_weight2
) {
  SCOPED_TIMER("Compute Initial Border Region");

  _border_region.initialize(block1, block2, max_border_region_weight1, max_border_region_weight2);

  QuotientGraph::CutEdges &cut_edges = _q_graph.cut_edges(block1, block2);
  const std::size_t num_cut_edges = cut_edges.size();

  if (_c_ctx.deterministic) {
    _random.reinit(Random::get_seed() + (block1 * _p_graph.k() + block2));
  }
  _random.shuffle(cut_edges.begin(), cut_edges.begin() + num_cut_edges);

  for (std::size_t i = 0; i < num_cut_edges; ++i) {
    const auto [u, v] = cut_edges[i];
    if (_p_graph.block(u) != block1 || _p_graph.block(v) != block2) {
      return;
    }

    const bool u_is_contained = _border_region.contains(u);
    const bool v_is_contained = _border_region.contains(v);
    if (u_is_contained && v_is_contained) {
      return;
    }

    const NodeWeight u_weight = _graph.node_weight(u);
    const NodeWeight v_weight = _graph.node_weight(v);

    const bool u_fits = _border_region.fits_in_region1(u_weight);
    const bool v_fits = _border_region.fits_in_region2(v_weight);
    if (u_is_contained) {
      if (v_fits) {
        _border_region.insert_border_node_into_region2(v, v_weight);
      }
    } else if (v_is_contained) {
      if (u_fits) {
        _border_region.insert_border_node_into_region1(u, u_weight);
      }
    } else if (u_fits && v_fits) {
      _border_region.insert_border_node_into_region1(u, u_weight);
      _border_region.insert_border_node_into_region2(v, v_weight);
    }
  }

  DBG << "Initial border region sizes: block " << _border_region.block1() << " = "
      << _border_region.size_region1() << ", block " << _border_region.block2() << " = "
      << _border_region.size_region2();
}

void BorderRegionConstructor::expand_border_regions() {
  SCOPED_TIMER("Expand Border Region");

  const NodeID max_border_distance = _c_ctx.max_border_distance;
  if (max_border_distance == 0) {
    return;
  }

  expand_border_region(kSourceTag);
  expand_border_region(kSinkTag);

  DBG << "Expanded border region sizes: block " << _border_region.block1() << " = "
      << _border_region.size_region1() << ", block " << _border_region.block2() << " = "
      << _border_region.size_region2();
}

void BorderRegionConstructor::expand_border_region(const bool source_side) {
  _bfs_runner.reset();
  _bfs_runner.add_seeds(
      source_side ? _border_region.initial_nodes_region1() : _border_region.initial_nodes_region2()
  );

  const NodeID max_border_distance = _c_ctx.max_border_distance;
  const BlockID block = source_side ? _border_region.block1() : _border_region.block2();

  _bfs_runner.perform([&](const NodeID u, const NodeID u_distance, auto &queue) {
    const NodeID v_distance = u_distance + 1;

    _graph.adjacent_nodes(u, [&](const NodeID v) {
      if (_p_graph.block(v) != block || _border_region.contains(v)) {
        return;
      }

      const NodeWeight v_weight = _graph.node_weight(v);
      if (_border_region.fits(source_side, v_weight)) {
        _border_region.insert(source_side, v, v_weight);

        if (v_distance < max_border_distance) {
          queue.push_back(v);
        }
      }
    });
  });
}

} // namespace kaminpar::shm
