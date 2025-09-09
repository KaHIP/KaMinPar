#pragma once

#include <span>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/flow_network/quotient_graph.h"
#include "kaminpar-shm/refinement/flow/util/breadth_first_search.h"
#include "kaminpar-shm/refinement/flow/util/node_status.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm {

class BorderRegion {
  static constexpr std::uint8_t kSource = 1;
  static constexpr std::uint8_t kSink = 2;

public:
  BorderRegion()
      : _block1(kInvalidBlockID),
        _block2(kInvalidBlockID),
        _max_weight1(0),
        _max_weight2(0),
        _cur_weight1(0),
        _cur_weight2(0) {};

  BorderRegion(BorderRegion &&) noexcept = default;
  BorderRegion &operator=(BorderRegion &&) noexcept = default;

  BorderRegion(const BorderRegion &) = delete;
  BorderRegion &operator=(const BorderRegion &) = delete;

  void initialize(
      const BlockID block1,
      const BlockID block2,
      const NodeWeight max_weight1,
      const NodeWeight max_weight2
  ) {
    _block1 = block1;
    _block2 = block2;

    _max_weight1 = max_weight1;
    _max_weight2 = max_weight2;

    _cur_weight1 = 0;
    _cur_weight2 = 0;

    _node_status.clear();

    _initial_source_side_border_nodes.clear();
    _initial_sink_side_border_nodes.clear();

    _source_side_nodes.clear();
    _sink_side_nodes.clear();
  }

  void insert_border_node_into_region1(const NodeID u, const NodeWeight u_weight) {
    KASSERT(!contains(u));
    KASSERT(fits_in_region1(u_weight));

    _cur_weight1 += u_weight;
    _node_status[u] = kSource;
    _source_side_nodes.push_back(u);

    _initial_source_side_border_nodes.push_back(u);
  }

  void insert_border_node_into_region2(const NodeID u, const NodeWeight u_weight) {
    KASSERT(!contains(u));
    KASSERT(fits_in_region2(u_weight));

    _cur_weight2 += u_weight;
    _node_status[u] = kSink;
    _sink_side_nodes.push_back(u);

    _initial_sink_side_border_nodes.push_back(u);
  }

  void insert(const bool source_side, const NodeID u, const NodeWeight u_weight) {
    KASSERT(!contains(u));
    KASSERT(source_side ? fits_in_region1(u_weight) : fits_in_region2(u_weight));

    if (source_side) {
      _cur_weight1 += u_weight;
      _node_status[u] = kSource;
      _source_side_nodes.push_back(u);
    } else {
      _cur_weight2 += u_weight;
      _node_status[u] = kSink;
      _sink_side_nodes.push_back(u);
    }
  }

  [[nodiscard]] bool fits_in_region1(const NodeWeight weight) const {
    return _cur_weight1 + weight <= _max_weight1;
  }

  [[nodiscard]] bool fits_in_region2(const NodeWeight weight) const {
    return _cur_weight2 + weight <= _max_weight2;
  }

  [[nodiscard]] bool fits(const bool source_side, const NodeWeight weight) const {
    const NodeWeight cur_weight = source_side ? _cur_weight1 : _cur_weight2;
    const NodeWeight max_weight = source_side ? _max_weight1 : _max_weight2;
    return cur_weight + weight <= max_weight;
  }

  [[nodiscard]] bool region1_contains(const NodeID u) const {
    const std::uint8_t *status = _node_status.get_if_contained(u);
    return status != nullptr && *status == kSource;
  }

  [[nodiscard]] bool region2_contains(const NodeID u) const {
    const std::uint8_t *status = _node_status.get_if_contained(u);
    return status != nullptr && *status == kSink;
  }

  [[nodiscard]] bool contains(const NodeID u) const {
    return _node_status.contains(u);
  }

  [[nodiscard]] BlockID empty() const {
    return size() == 0;
  }

  [[nodiscard]] BlockID size() const {
    return size_region1() + size_region2();
  }

  [[nodiscard]] BlockID size_region1() const {
    return _source_side_nodes.size();
  }

  [[nodiscard]] BlockID size_region2() const {
    return _sink_side_nodes.size();
  }

  [[nodiscard]] BlockID block1() const {
    return _block1;
  }

  [[nodiscard]] BlockID block2() const {
    return _block2;
  }

  [[nodiscard]] NodeWeight max_weight_region1() const {
    return _max_weight1;
  }

  [[nodiscard]] NodeWeight max_weight_region2() const {
    return _max_weight2;
  }

  [[nodiscard]] NodeWeight weight_region1() const {
    return _cur_weight1;
  }

  [[nodiscard]] NodeWeight weight_region2() const {
    return _cur_weight2;
  }

  [[nodiscard]] std::span<const NodeID> nodes_region1() const {
    return _source_side_nodes;
  }

  [[nodiscard]] std::span<const NodeID> nodes_region2() const {
    return _sink_side_nodes;
  }

  [[nodiscard]] std::span<const NodeID> initial_nodes_region1() const {
    return _initial_source_side_border_nodes;
  }

  [[nodiscard]] std::span<const NodeID> initial_nodes_region2() const {
    return _initial_sink_side_border_nodes;
  }

  void free() {
    _block1 = kInvalidBlockID;
    _block2 = kInvalidBlockID;

    _max_weight1 = 0;
    _max_weight2 = 0;

    _cur_weight1 = 0;
    _cur_weight2 = 0;

    _node_status.free();

    _initial_source_side_border_nodes.clear();
    _initial_source_side_border_nodes.shrink_to_fit();

    _initial_sink_side_border_nodes.clear();
    _initial_sink_side_border_nodes.shrink_to_fit();

    _source_side_nodes.clear();
    _source_side_nodes.shrink_to_fit();

    _sink_side_nodes.clear();
    _sink_side_nodes.shrink_to_fit();
  }

private:
  BlockID _block1;
  BlockID _block2;

  NodeWeight _max_weight1;
  NodeWeight _max_weight2;

  NodeWeight _cur_weight1;
  NodeWeight _cur_weight2;

  DynamicFlatMap<NodeID, std::uint8_t> _node_status;

  ScalableVector<NodeID> _initial_source_side_border_nodes;
  ScalableVector<NodeID> _initial_sink_side_border_nodes;

  ScalableVector<NodeID> _source_side_nodes;
  ScalableVector<NodeID> _sink_side_nodes;
};

class BorderRegionConstructor {
  SET_DEBUG(false);

  static constexpr bool kSourceTag = true;
  static constexpr bool kSinkTag = false;

public:
  BorderRegionConstructor(
      const PartitionContext &p_ctx,
      const FlowNetworkConstructionContext &c_ctx,
      const QuotientGraph &q_graph,
      const PartitionedCSRGraph &p_graph,
      const CSRGraph &graph
  );

  BorderRegionConstructor(BorderRegionConstructor &&) noexcept = default;
  BorderRegionConstructor &operator=(BorderRegionConstructor &&) noexcept = delete;

  BorderRegionConstructor(const BorderRegionConstructor &) = delete;
  BorderRegionConstructor &operator=(const BorderRegionConstructor &) = delete;

  [[nodiscard]] const BorderRegion &
  construct(BlockID block1, BlockID block2, BlockWeight block1_weight, BlockWeight block2_weight);

  void free();

private:
  void compute_border_regions(
      BlockID block1,
      BlockID block2,
      BlockWeight max_border_region_weight1,
      BlockWeight max_border_region_weight2
  );

  void expand_border_regions();

  void expand_border_region(bool source_side);

private:
  const PartitionContext &_p_ctx;
  const FlowNetworkConstructionContext &_c_ctx;

  const QuotientGraph &_q_graph;
  const PartitionedCSRGraph &_p_graph;
  const CSRGraph &_graph;

  BorderRegion _border_region;

  BFSRunner _bfs_runner;
  Random _random;
};

} // namespace kaminpar::shm
