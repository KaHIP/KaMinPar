/*******************************************************************************
 * Compressed static graph representations.
 *
 * @file:   compressed_graph.cc
 * @author: Daniel Salwasser
 * @date:   01.12.2023
 ******************************************************************************/
#include "compressed_graph.h"

#include <kassert/kassert.hpp>

#include "kaminpar-common/heap_profiler.h"

namespace kaminpar::shm {

CompressedGraph::CompressedGraph(
    StaticArray<EdgeID> nodes,
    StaticArray<std::uint8_t> compressed_edges,
    StaticArray<NodeWeight> node_weights,
    StaticArray<EdgeWeight> edge_weights,
    EdgeID edge_count,
    NodeID max_degree,
    std::size_t high_degree_count,
    std::size_t part_count,
    std::size_t interval_count
)
    : _nodes(std::move(nodes)),
      _compressed_edges(std::move(compressed_edges)),
      _node_weights(std::move(node_weights)),
      _edge_weights(std::move(edge_weights)),
      _node_count(static_cast<NodeID>(_nodes.size() - 1)),
      _edge_count(edge_count),
      _max_degree(max_degree),
      _high_degree_count(high_degree_count),
      _part_count(part_count),
      _interval_count(interval_count) {
  KASSERT(kHighDegreeEncoding || _high_degree_count == 0);
  KASSERT(kHighDegreeEncoding || _part_count == 0);
  KASSERT(kIntervalEncoding || interval_count == 0);

  if (_node_weights.empty()) {
    _total_node_weight = static_cast<NodeWeight>(n());
    _max_node_weight = 1;
  } else {
    _total_node_weight =
        std::accumulate(_node_weights.begin(), _node_weights.end(), static_cast<NodeWeight>(0));
    _max_node_weight = *std::max_element(_node_weights.begin(), _node_weights.end());
  }

  if (_edge_weights.empty()) {
    _total_edge_weight = static_cast<EdgeWeight>(m());
  } else {
    _total_edge_weight =
        std::accumulate(_edge_weights.begin(), _edge_weights.end(), static_cast<EdgeWeight>(0));
  }

  init_degree_buckets();
};

void CompressedGraph::init_degree_buckets() {
  KASSERT(std::all_of(_buckets.begin(), _buckets.end(), [](const auto n) { return n == 0; }));

  if (sorted()) {
    for (const NodeID u : nodes()) {
      ++_buckets[degree_bucket(degree(u)) + 1];
    }
    auto last_nonempty_bucket =
        std::find_if(_buckets.rbegin(), _buckets.rend(), [](const auto n) { return n > 0; });
    _number_of_buckets = std::distance(_buckets.begin(), (last_nonempty_bucket + 1).base());
  } else {
    _buckets[1] = n();
    _number_of_buckets = 1;
  }

  std::partial_sum(_buckets.begin(), _buckets.end(), _buckets.begin());
}

void CompressedGraph::update_total_node_weight() {
  if (_node_weights.empty()) {
    _total_node_weight = n();
    _max_node_weight = 1;
  } else {
    _total_node_weight =
        std::accumulate(_node_weights.begin(), _node_weights.end(), static_cast<NodeWeight>(0));
    _max_node_weight = *std::max_element(_node_weights.begin(), _node_weights.end());
  }
}

CompressedGraph CompressedGraphBuilder::compress(const CSRGraph &graph) {
  const bool store_node_weights = graph.node_weighted();
  const bool store_edge_weights = graph.edge_weighted();

  CompressedGraphBuilder builder;
  builder.init(graph.n(), graph.m(), store_node_weights, store_edge_weights);

  std::vector<std::pair<NodeID, EdgeWeight>> neighbourhood;
  for (const NodeID node : graph.nodes()) {
    if (store_node_weights) {
      builder.set_node_weight(node, graph.node_weight(node));
    }

    for (const auto [edge, adjacent_node] : graph.neighbors(node)) {
      neighbourhood.push_back(std::make_pair(adjacent_node, graph.edge_weight(edge)));
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

  _store_node_weights = store_node_weights;
  _store_edge_weights = store_edge_weights;

  _total_node_weight = 0;
  _total_edge_weight = 0;

  const std::size_t max_bytes_node_id = varint_max_length<NodeID>();
  const std::size_t max_bytes_edge_id = varint_max_length<EdgeID>();
  const std::size_t max_part_count = (node_count / CompressedGraph::kHighDegreeThreshold) + 1;
  const std::size_t max_size = max_bytes_node_id * node_count * 2 +
                               max_bytes_node_id * max_part_count + max_bytes_edge_id * edge_count;
  if constexpr (kHeapProfiling) {
    // As we overcommit memory do not track the amount of bytes used directly. Instead record it
    // manually when building.
    _compressed_edges = (uint8_t *)heap_profiler::std_malloc(max_size);
  } else {
    _compressed_edges = (uint8_t *)malloc(max_size);
  }
  _cur_compressed_edges = _compressed_edges;

  _edge_count = 0;
  _max_degree = 0;

  _high_degree_count = 0;
  _part_count = 0;
  _interval_count = 0;
}

void CompressedGraphBuilder::add_node(
    const NodeID node, std::vector<std::pair<NodeID, EdgeWeight>> &neighbourhood
) {
  // Store the index into the compressed edge array of the start of the neighbourhood of the node
  // in its entry in the node array.
  _nodes[node] = static_cast<EdgeID>(_cur_compressed_edges - _compressed_edges);

  const NodeID degree = neighbourhood.size();
  const EdgeID first_edge_id = _edge_count;
  const bool split_neighbourhood =
      CompressedGraph::kHighDegreeThreshold && degree > CompressedGraph::kHighDegreeThreshold;

  _cur_compressed_edges += varint_encode(degree, _cur_compressed_edges);

  std::uint8_t *marked_byte = _cur_compressed_edges;
  if constexpr (CompressedGraph::kIntervalEncoding) {
    if (!split_neighbourhood) {
      _cur_compressed_edges += marked_varint_encode(first_edge_id, false, _cur_compressed_edges);
    } else {
      _cur_compressed_edges += varint_encode(first_edge_id, _cur_compressed_edges);
    }
  } else {
    _cur_compressed_edges += varint_encode(first_edge_id, _cur_compressed_edges);
  }

  if (degree == 0) {
    return;
  }

  _max_degree = std::max(_max_degree, degree);
  if (!_store_edge_weights) {
    _edge_count += degree;
  }

  // Sort the adjacent nodes in ascending order.
  std::sort(neighbourhood.begin(), neighbourhood.end());

  if constexpr (CompressedGraph::kHighDegreeEncoding) {
    if (split_neighbourhood) {
      const NodeID part_count = ((degree % CompressedGraph::kHighDegreeThreshold) == 0)
                                    ? (degree / CompressedGraph::kHighDegreeThreshold)
                                    : ((degree / CompressedGraph::kHighDegreeThreshold) + 1);
      const NodeID last_part_length = ((degree % CompressedGraph::kHighDegreeThreshold) == 0)
                                          ? CompressedGraph::kHighDegreeThreshold
                                          : (degree % CompressedGraph::kHighDegreeThreshold);

      uint8_t *part_ptr = _cur_compressed_edges;
      _cur_compressed_edges += sizeof(NodeID) * part_count;

      for (NodeID i = 0; i < part_count; ++i) {
        auto part_begin = neighbourhood.begin() + i * CompressedGraph::kHighDegreeThreshold;
        const NodeID part_length =
            (i + 1 == part_count) ? last_part_length : CompressedGraph::kHighDegreeThreshold;

        std::uint8_t *cur_part_ptr = part_ptr + sizeof(NodeID) * i;
        *((NodeID *)cur_part_ptr) = static_cast<NodeID>(_cur_compressed_edges - part_ptr);

        std::vector<std::pair<NodeID, EdgeWeight>> part_neighbourhood(
            part_begin, part_begin + part_length
        );
        add_edges(node, nullptr, part_neighbourhood);
      }

      _part_count += part_count;
      _high_degree_count += 1;
      return;
    }
  }

  add_edges(node, marked_byte, neighbourhood);
}

void CompressedGraphBuilder::set_node_weight(const NodeID node, const NodeWeight weight) {
  KASSERT(_store_node_weights);

  _total_node_weight += weight;
  _node_weights[node] = weight;
}

CompressedGraph CompressedGraphBuilder::build() {
  std::size_t stored_bytes = static_cast<std::size_t>(_cur_compressed_edges - _compressed_edges);
  RECORD("compressed_edges")
  StaticArray<std::uint8_t> compressed_edges(_compressed_edges, stored_bytes);

  if constexpr (kHeapProfiling) {
    heap_profiler::HeapProfiler::global().record_alloc(_compressed_edges, stored_bytes);
  }

  _nodes[_nodes.size() - 1] = stored_bytes;

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
      std::move(compressed_edges),
      std::move(_node_weights),
      std::move(_edge_weights),
      _edge_count,
      _max_degree,
      _high_degree_count,
      _part_count,
      _interval_count
  );
}

std::int64_t CompressedGraphBuilder::total_node_weight() const {
  return _total_node_weight;
}

std::int64_t CompressedGraphBuilder::total_edge_weight() const {
  return _total_edge_weight;
}

void CompressedGraphBuilder::add_edges(
    NodeID node,
    std::uint8_t *marked_byte,
    std::vector<std::pair<NodeID, EdgeWeight>> &neighbourhood
) {
  const auto store_edge_weight = [&](const EdgeWeight edge_weight) {
    _total_edge_weight += edge_weight;
    _edge_weights[_edge_count++] = edge_weight;
  };

  // Find intervals [i, j] of consecutive adjacent nodes i, i + 1, ..., j - 1, j of length at
  // least kIntervalLengthTreshold. Instead of storing all nodes, only store a representation of
  // the left extreme i and the length j - i + 1. Left extremes are compressed using the
  // differences between each left extreme and the previous right extreme minus 2 (because there
  // must be at least one integer between the end of an interval and the beginning of the next
  // one), except the first left extreme which is stored directly. The lengths are decremented by
  // kIntervalLengthTreshold, the minimum length of an interval.
  if constexpr (CompressedGraph::kIntervalEncoding) {
    NodeID interval_count = 0;

    // Store the pointer to the interval count and skip the amount of bytes needed to store the
    // interval count as we can only determine the amount of intervals after finding all of
    // them.
    std::uint8_t *interval_count_ptr = _cur_compressed_edges;
    _cur_compressed_edges += sizeof(NodeID);

    if (neighbourhood.size() > 1) {
      NodeID previous_right_extreme = 2;
      NodeID interval_len = 1;
      NodeID prev_adjacent_node = (*neighbourhood.begin()).first;

      for (auto iter = neighbourhood.begin() + 1; iter != neighbourhood.end(); ++iter) {
        const NodeID adjacent_node = (*iter).first;

        if (prev_adjacent_node + 1 == adjacent_node) {
          interval_len++;

          // The interval ends if there are no more nodes or the next node is not the increment of
          // the current node.
          if (iter + 1 == neighbourhood.end() || (*(iter + 1)).first != adjacent_node + 1) {
            if (interval_len >= CompressedGraph::kIntervalLengthTreshold) {
              const NodeID left_extreme = adjacent_node + 1 - interval_len;
              const NodeID left_extreme_gap = left_extreme + 2 - previous_right_extreme;
              const NodeID interval_length_gap =
                  interval_len - CompressedGraph::kIntervalLengthTreshold;

              interval_count += 1;
              _cur_compressed_edges += varint_encode(left_extreme_gap, _cur_compressed_edges);
              _cur_compressed_edges += varint_encode(interval_length_gap, _cur_compressed_edges);

              if (_store_edge_weights) {
                for (NodeID i = 0; i < interval_len; ++i) {
                  const EdgeWeight edge_weight = (*(iter + 1 + i - interval_len)).second;
                  store_edge_weight(edge_weight);
                }
              }

              previous_right_extreme = adjacent_node;
              iter = neighbourhood.erase(iter - interval_len + 1, iter + 1) - 1;
              if (iter == neighbourhood.end()) {
                break;
              }
            }

            interval_len = 1;
          }
        }

        prev_adjacent_node = adjacent_node;
      }
    }

    // If intervals have been encoded store the interval_count and set the bit in the encoded
    // degree of the node indicating that intervals have been used for the neighbourhood.
    // Otherwise, fix the amount of bytes stored as we don't store the interval count if no
    // intervals have been encoded.
    if (marked_byte == nullptr) {
      *((NodeID *)interval_count_ptr) = interval_count;
    } else if (interval_count > 0) {
      *((NodeID *)interval_count_ptr) = interval_count;
      *marked_byte |= 0b01000000;
    } else {
      _cur_compressed_edges -= sizeof(NodeID);
    }

    if (interval_count > 0) {
      _interval_count += 1;
    }

    // If all incident edges have been compressed using intervals then gap encoding cannot be
    // applied.
    if (neighbourhood.empty()) {
      return;
    }
  }

  // Store the remaining adjacent nodes using gap encoding. That is instead of storing the nodes
  // v_1, v_2, ..., v_{k - 1}, v_k directly, store the gaps v_1 - u, v_2 - v_1, ..., v_k - v_{k -
  // 1} between the nodes, where u is the source node. Note that all gaps except the first one
  // have to be positive as we sorted the nodes in ascending order. Thus, only for the first gap
  // the sign is additionally stored.
  const auto [first_adjacent_node, first_edge_weight] = *neighbourhood.begin();
  // TODO: Does the value range cover everything s.t. a underflow cannot happen?
  const std::make_signed_t<NodeID> first_gap = first_adjacent_node - node;
  _cur_compressed_edges += signed_varint_encode(first_gap, _cur_compressed_edges);
  if (_store_edge_weights) {
    store_edge_weight(first_edge_weight);
  }

  RLEncoder<NodeID> rl_encoder(_cur_compressed_edges);

  NodeID prev_adjacent_node = first_adjacent_node;
  const auto iter_end = neighbourhood.end();
  for (auto iter = neighbourhood.begin() + 1; iter != iter_end; ++iter) {
    const auto [adjacent_node, edge_weight] = *iter;
    const NodeID gap = adjacent_node - prev_adjacent_node;

    if constexpr (CompressedGraph::kRunLengthEncoding) {
      _cur_compressed_edges += rl_encoder.add(gap);
    } else {
      _cur_compressed_edges += varint_encode(gap, _cur_compressed_edges);
    }

    if (_store_edge_weights) {
      store_edge_weight(edge_weight);
    }

    prev_adjacent_node = adjacent_node;
  }

  if constexpr (CompressedGraph::kRunLengthEncoding) {
    rl_encoder.flush();
  }
}

} // namespace kaminpar::shm
