/*******************************************************************************
 * Compressed static graph representations.
 *
 * @file:   compressed_graph.cc
 * @author: Daniel Salwasser
 * @date:   01.12.2023
 ******************************************************************************/
#include "compressed_graph.h"

namespace kaminpar::shm {

CompressedGraph CompressedGraphBuilder::compress(const CSRGraph &graph) {
  const bool store_node_weights = graph.is_node_weighted();
  const bool store_edge_weights = graph.is_edge_weighted();

  CompressedGraphBuilder builder;
  builder.init(graph.n(), graph.m(), store_node_weights, store_edge_weights);

  std::vector<NodeID> neighbourhood;
  for (const NodeID node : graph.nodes()) {
    if (store_node_weights) {
      builder.set_node_weight(node, graph.node_weight(node));
    }

    for (const auto [edge, adjacent_node] : graph.neighbors(node)) {
      if (store_edge_weights) {
        builder.set_edge_weight(edge, graph.edge_weight(edge));
      }

      neighbourhood.push_back(adjacent_node);
    }

    builder.add_node(node, neighbourhood);
    neighbourhood.clear();
  }

  return builder.build();
}

void CompressedGraphBuilder::init(
    std::size_t node_count, std::size_t edge_count, bool store_node_weights, bool store_edge_weights
) {
  _nodes.resize(node_count + 1);

  if (store_node_weights) {
    _node_weights.resize(node_count);
  }

  if (store_edge_weights) {
    _edge_weights.resize(edge_count * 2);
  }

  _total_node_weight = 0;
  _total_edge_weight = 0;

  const std::size_t max_bytes = varint_max_length<NodeID>();
  const std::size_t max_size = max_bytes * node_count * 2 + max_bytes * edge_count;
  if constexpr (kHeapProfiling) {
    // As we overcommit memory do not track the amount of bytes used directly. Instead record it
    // manually when building.
    _compressed_edges = (uint8_t *)heap_profiler::std_malloc(max_size);
  } else {
    _compressed_edges = (uint8_t *)malloc(max_size);
  }
  _cur_compressed_edges = _compressed_edges;
  _stored_bytes = 0;

  _edge_count = 0;
  _interval_count = 0;
}

void CompressedGraphBuilder::add_node(const NodeID node, std::vector<NodeID> &neighbourhood) {
  // Store the index into the compressed edge array of the start of the neighbourhood of the node
  // in its entry in the node array.
  _nodes[node] = static_cast<EdgeID>(_cur_compressed_edges - _compressed_edges);

  const NodeID degree = neighbourhood.size();
  const NodeID first_edge_id = _edge_count;

  _edge_count += degree;

  std::uint8_t *first_degree_byte = _cur_compressed_edges;
  if constexpr (CompressedGraph::kIntervalEncoding) {
    _cur_compressed_edges += marked_varint_encode(degree, false, _cur_compressed_edges);
  } else {
    _cur_compressed_edges += varint_encode(degree, _cur_compressed_edges);
  }
  _cur_compressed_edges += varint_encode(first_edge_id, _cur_compressed_edges);

  if (degree == 0) {
    return;
  }

  // Sort the adjacent nodes in ascending order.
  std::sort(neighbourhood.begin(), neighbourhood.end());

  // Find intervals [i, j] of consecutive adjacent nodes i, i + 1, ..., j - 1, j of length at
  // least kIntervalLengthTreshold. Instead of storing all nodes, only store a representation of
  // the left extreme i and the length j - i + 1. Left extremes are compressed using the
  // differences between each left extreme and the previous right extreme minus 2 (because there
  // must be at least one integer between the end of an interval and the beginning of the next
  // one), except the first left extreme which is stored directly. The lengths are decremented by
  // kIntervalLengthTreshold, the minimum length of an interval.
  if constexpr (CompressedGraph::kIntervalEncoding) {
    if (neighbourhood.size() > 1) {
      NodeID interval_count = 0;

      // Store the pointer to the interval count and skip the amount of bytes needed to store the
      // interval count as we can only determine the amount of intervals after finding all of
      // them.
      uint8_t *interval_count_ptr = _cur_compressed_edges;
      _stored_bytes += sizeof(NodeID);

      NodeID previous_right_extreme = 2;
      NodeID interval_len = 1;
      NodeID prev_adjacent_node = *neighbourhood.begin();
      for (auto iter = neighbourhood.begin() + 1; iter != neighbourhood.end(); ++iter) {
        const NodeID adjacent_node = *iter;

        if (prev_adjacent_node + 1 == adjacent_node) {
          interval_len++;

          // The interval ends if there are no more nodes or the next node is not the increment of
          // the current node.
          if (iter + 1 == neighbourhood.end() || *(iter + 1) != adjacent_node + 1) {
            if (interval_len >= CompressedGraph::kIntervalLengthTreshold) {
              const NodeID left_extreme = adjacent_node + 1 - interval_len;
              const NodeID left_extreme_gap = left_extreme + 2 - previous_right_extreme;
              const std::size_t interval_length_gap =
                  interval_len - CompressedGraph::kIntervalLengthTreshold;

              interval_count += 1;
              _cur_compressed_edges += varint_encode(left_extreme_gap, _cur_compressed_edges);
              _cur_compressed_edges += varint_encode(interval_length_gap, _cur_compressed_edges);

              previous_right_extreme = adjacent_node;
              iter = neighbourhood.erase(iter - interval_len + 1, iter + 1);
              if (iter == neighbourhood.end()) {
                break;
              }
            }

            interval_len = 1;
          }
        }

        prev_adjacent_node = adjacent_node;
      }

      // If intervals have been encoded store the interval_count and set the bit in the encoded
      // degree of the node indicating that intervals have been used for the neighbourhood.
      // Otherwise, fix the amount of bytes stored as we don't store the interval count if no
      // intervals have been encoded.
      if (interval_count > 0) {
        *((NodeID *)interval_count_ptr) = interval_count;
        *first_degree_byte |= 0x01000000;
      } else {
        _stored_bytes -= sizeof(NodeID);
      }

      // If all incident edges have been compressed using intervals then gap encoding cannot be
      // applied.
      if (neighbourhood.empty()) {
        return;
      }
    }
  }

  // Store the remaining adjacent nodes using gap encoding. That is instead of storing the nodes
  // v_1, v_2, ..., v_{k - 1}, v_k directly, store the gaps v_1 - u, v_2 - v_1, ..., v_k - v_{k -
  // 1} between the nodes, where u is the source node. Note that all gaps except the first one
  // have to be positive as we sorted the nodes in ascending order. Thus, only for the first gap
  // the sign is additionally stored.
  const NodeID first_adjacent_node = *neighbourhood.begin();
  // TODO: Does the value range cover everything s.t. a underflow cannot happen?
  const std::make_signed_t<NodeID> first_gap = first_adjacent_node - node;
  _cur_compressed_edges += signed_varint_encode(first_gap, _cur_compressed_edges);

  NodeID prev_adjacent_node = first_adjacent_node;
  const auto iter_end = neighbourhood.end();
  for (auto iter = neighbourhood.begin() + 1; iter != iter_end; ++iter) {
    const NodeID adjacent_node = *iter;
    const NodeID gap = adjacent_node - prev_adjacent_node;

    _cur_compressed_edges += varint_encode(gap, _cur_compressed_edges);
    prev_adjacent_node = adjacent_node;
  }
}

void CompressedGraphBuilder::set_node_weight(const NodeID node, const NodeWeight weight) {
  _total_node_weight += weight;
  _node_weights[node] = weight;
}

void CompressedGraphBuilder::set_edge_weight(const EdgeID edge, const EdgeWeight weight) {
  _total_edge_weight += weight;
  _edge_weights[edge] = weight;
}

CompressedGraph CompressedGraphBuilder::build() {
  if constexpr (kHeapProfiling) {
    heap_profiler::HeapProfiler::global().record_alloc(_compressed_edges, _stored_bytes);
  }

  const bool unit_node_weights = static_cast<NodeID>(_total_node_weight + 1) == _nodes.size();
  if (unit_node_weights) {
    _node_weights.free();
  }

  const bool unit_edge_weights = static_cast<EdgeID>(_total_edge_weight) == _edge_count;
  if (unit_edge_weights) {
    _edge_weights.free();
  }

  return CompressedGraph(
      std::move(_nodes),
      StaticArray<std::uint8_t>(_compressed_edges, _stored_bytes),
      std::move(_node_weights),
      std::move(_edge_weights),
      _edge_count,
      _interval_count
  );
}

std::int64_t CompressedGraphBuilder::total_node_weight() const {
  return _total_node_weight;
}

std::int64_t CompressedGraphBuilder::total_edge_weight() const {
  return _total_edge_weight;
}

} // namespace kaminpar::shm