/*******************************************************************************
 * Compressed static graph representation.
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
    CompactStaticArray<EdgeID> nodes,
    StaticArray<std::uint8_t> compressed_edges,
    StaticArray<NodeWeight> node_weights,
    StaticArray<EdgeWeight> edge_weights,
    EdgeID edge_count,
    NodeID max_degree,
    bool sorted,
    std::size_t num_high_degree_nodes,
    std::size_t num_high_degree_parts,
    std::size_t num_interval_nodes,
    std::size_t num_intervals
)
    : _nodes(std::move(nodes)),
      _compressed_edges(std::move(compressed_edges)),
      _node_weights(std::move(node_weights)),
      _edge_weights(std::move(edge_weights)),
      _edge_count(edge_count),
      _max_degree(max_degree),
      _sorted(sorted),
      _num_high_degree_nodes(num_high_degree_nodes),
      _num_high_degree_parts(num_high_degree_parts),
      _num_interval_nodes(num_interval_nodes),
      _num_intervals(num_intervals) {
  KASSERT(kHighDegreeEncoding || _num_high_degree_nodes == 0);
  KASSERT(kHighDegreeEncoding || _num_high_degree_parts == 0);
  KASSERT(kIntervalEncoding || _num_interval_nodes == 0);
  KASSERT(kIntervalEncoding || _num_intervals == 0);

  if (_node_weights.empty()) {
    _total_node_weight = static_cast<NodeWeight>(n());
    _max_node_weight = 1;
  } else {
    _total_node_weight = parallel::accumulate(_node_weights, static_cast<NodeWeight>(0));
    _max_node_weight = parallel::max_element(_node_weights);
  }

  if (_edge_weights.empty()) {
    _total_edge_weight = static_cast<EdgeWeight>(m());
  } else {
    _total_edge_weight = parallel::accumulate(_edge_weights, static_cast<EdgeWeight>(0));
  }

  init_degree_buckets();
};

void CompressedGraph::init_degree_buckets() {
  KASSERT(std::all_of(_buckets.begin(), _buckets.end(), [](const auto n) { return n == 0; }));

  if (sorted()) {
    constexpr std::size_t kNumBuckets = kNumberOfDegreeBuckets<NodeID> + 1;
    tbb::enumerable_thread_specific<std::array<NodeID, kNumBuckets>> buckets_ets([&] {
      return std::array<NodeID, kNumBuckets>{};
    });

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, n()), [&](const auto &r) {
      auto &buckets = buckets_ets.local();
      for (NodeID u = r.begin(); u != r.end(); ++u) {
        ++buckets[degree_bucket(degree(u)) + 1];
      }
    });

    std::fill(_buckets.begin(), _buckets.end(), 0);
    for (auto &local_buckets : buckets_ets) {
      for (std::size_t i = 0; i < kNumBuckets; ++i) {
        _buckets[i] += local_buckets[i];
      }
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
    _total_node_weight = parallel::accumulate(_node_weights, static_cast<NodeWeight>(0));
    _max_node_weight = parallel::max_element(_node_weights);
  }
}

void CompressedGraph::remove_isolated_nodes(const NodeID isolated_nodes) {
  KASSERT(sorted());

  if (isolated_nodes == 0) {
    return;
  }

  const NodeID new_n = n() - isolated_nodes;
  _nodes.restrict(new_n + 1);
  if (!_node_weights.empty()) {
    _node_weights.restrict(new_n);
  }

  update_total_node_weight();

  // Update degree buckets
  for (std::size_t i = 0; i < _buckets.size() - 1; ++i) {
    _buckets[1 + i] -= isolated_nodes;
  }

  // If the graph has only isolated nodes then there are no buckets afterwards
  if (_number_of_buckets == 1) {
    _number_of_buckets = 0;
  }
}

void CompressedGraph::integrate_isolated_nodes() {
  KASSERT(sorted());

  const NodeID nonisolated_nodes = n();
  _nodes.unrestrict();
  _node_weights.unrestrict();

  const NodeID isolated_nodes = n() - nonisolated_nodes;
  update_total_node_weight();

  // Update degree buckets
  for (std::size_t i = 0; i < _buckets.size() - 1; ++i) {
    _buckets[1 + i] += isolated_nodes;
  }

  // If the graph has only isolated nodes then there is one bucket afterwards
  if (_number_of_buckets == 0) {
    _number_of_buckets = 1;
  }
}

std::size_t CompressedGraphBuilder::compressed_edge_array_max_size(
    const NodeID node_count, const EdgeID edge_count
) {
  std::size_t max_size =
      node_count * varint_max_length<EdgeID>() + 2 * edge_count * varint_max_length<NodeID>();

  if constexpr (CompressedGraph::kHighDegreeEncoding) {
    if constexpr (CompressedGraph::kIntervalEncoding) {
      max_size += 2 * node_count * varint_max_length<NodeID>();
    } else {
      max_size += node_count * varint_max_length<NodeID>();
    }

    max_size += (edge_count / CompressedGraph::kHighDegreePartLength) * varint_max_length<NodeID>();
  }

  return max_size;
}

CompressedGraph CompressedGraphBuilder::compress(const CSRGraph &graph) {
  const bool store_node_weights = graph.is_node_weighted();
  const bool store_edge_weights = graph.is_edge_weighted();

  CompressedGraphBuilder builder;
  builder.init(graph.n(), graph.m(), store_node_weights, store_edge_weights, graph.sorted());

  std::vector<std::pair<NodeID, EdgeWeight>> neighbourhood;
  neighbourhood.reserve(graph.max_degree());

  for (const NodeID node : graph.nodes()) {
    if (store_node_weights) {
      builder.set_node_weight(node, graph.node_weight(node));
    }

    for (const auto [incident_edge, adjacent_node] : graph.neighbors(node)) {
      neighbourhood.emplace_back(adjacent_node, graph.edge_weight(incident_edge));
    }

    builder.add_node(node, neighbourhood);
    neighbourhood.clear();
  }

  return builder.build();
}

void CompressedGraphBuilder::init(
    const NodeID node_count,
    const EdgeID edge_count,
    const bool store_node_weights,
    const bool store_edge_weights,
    const bool sorted
) {
  KASSERT(node_count != std::numeric_limits<NodeID>::max() - 1);

  const std::size_t max_size = compressed_edge_array_max_size(node_count, edge_count);
  _nodes.resize(math::byte_width(max_size), node_count + 1);

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

  _sorted = sorted;

  if constexpr (kHeapProfiling) {
    // As we overcommit memory do not track the amount of bytes used directly. Instead record it
    // manually when building the compressed graph.
    _compressed_edges = (uint8_t *)heap_profiler::std_malloc(max_size);
  } else {
    _compressed_edges = (uint8_t *)malloc(max_size);
  }
  _cur_compressed_edges = _compressed_edges;

  _edge_count = 0;
  _max_degree = 0;

  _first_isolated_node = true;
  _effective_last_edge_offset = 0;

  _num_high_degree_nodes = 0;
  _num_high_degree_parts = 0;
  _num_interval_nodes = 0;
  _num_intervals = 0;
}

void CompressedGraphBuilder::add_node(
    const NodeID node, std::vector<std::pair<NodeID, EdgeWeight>> &neighbourhood
) {
  // Store the offset into the compressed edge array of the start of the neighbourhood for the node
  // in its entry in the node array.
  _nodes.write(node, static_cast<EdgeID>(_cur_compressed_edges - _compressed_edges));

  const NodeID degree = neighbourhood.size();
  if (degree == 0) {
    // If isolated nodes are continuously stored at the end of the nodes array, gap encoding for the
    // first edge id with respect to the source node can be used while determining the degree
    // through the first edge id. For this to work, at the first isolated node (or at the end of the
    // edge array if no isolated node exists, see build-method) we have to store the (effective)
    // first edge id of the isolated node as a gap with respect to the isolated node. Further, the
    // index in the node array of all following isolated nodes have to be shifted such that they are
    // seen as isolated nodes.
    if constexpr (CompressedGraph::kIsolatedNodesSeparation) {
      if (_first_isolated_node) {
        _first_isolated_node = false;
        _effective_last_edge_offset =
            static_cast<EdgeID>(_cur_compressed_edges - _compressed_edges);

        const EdgeID first_edge_gap = _edge_count - node;
        if constexpr (CompressedGraph::kIntervalEncoding) {
          _cur_compressed_edges +=
              marked_varint_encode(first_edge_gap, false, _cur_compressed_edges);
        } else {
          _cur_compressed_edges += varint_encode(first_edge_gap, _cur_compressed_edges);
        }
      } else {
        _nodes.write(node, _effective_last_edge_offset);
      }
    }

    return;
  }

  KASSERT(!CompressedGraph::kIsolatedNodesSeparation || _first_isolated_node);
  _max_degree = std::max(_max_degree, degree);

  // Store a pointer to the first byte of the first edge of this neighborhood. This byte encodes in
  // one of its bits whether interval encoding is used for this node, i.e., whether the nodes has
  // intervals in its neighbourhood.
  std::uint8_t *marked_byte = _cur_compressed_edges;

  // Store only the first edge for the source node. The degree can be obtained by determining the
  // difference between the first edge ids of a node and the next node. Additionally, store the
  // first edge as a gap when the isolated nodes are continuously stored at the end of the nodes
  // array.
  const EdgeID first_edge = _edge_count;
  if constexpr (CompressedGraph::kIsolatedNodesSeparation) {
    const EdgeID first_edge_gap = _edge_count - node;

    if constexpr (CompressedGraph::kIntervalEncoding) {
      _cur_compressed_edges += marked_varint_encode(first_edge_gap, false, _cur_compressed_edges);
    } else {
      _cur_compressed_edges += varint_encode(first_edge_gap, _cur_compressed_edges);
    }
  } else {
    if constexpr (CompressedGraph::kIntervalEncoding) {
      _cur_compressed_edges += marked_varint_encode(first_edge, false, _cur_compressed_edges);
    } else {
      _cur_compressed_edges += varint_encode(first_edge, _cur_compressed_edges);
    }
  }

  // Only increment the edge count if edge weights are not stored as otherwise the edge count is
  // incremented with each edge weight being added.
  if (!_store_edge_weights) {
    _edge_count += degree;
  }

  // Sort the adjacent nodes in ascending order.
  std::sort(neighbourhood.begin(), neighbourhood.end(), [](const auto &a, const auto &b) {
    return a.first < b.first;
  });

  // If high-degree encoding is used then split the neighborhood if the degree crosses a threshold.
  // The neighborhood is split into equally sized parts (except possible the last part) and each
  // part is encoded independently. Furthermore, the offset at which the part is encoded is also
  // stored.
  if constexpr (CompressedGraph::kHighDegreeEncoding) {
    const bool split_neighbourhood = degree >= CompressedGraph::kHighDegreeThreshold;

    if (split_neighbourhood) {
      const NodeID part_count = math::div_ceil(degree, CompressedGraph::kHighDegreePartLength);
      const NodeID last_part_length = ((degree % CompressedGraph::kHighDegreePartLength) == 0)
                                          ? CompressedGraph::kHighDegreePartLength
                                          : (degree % CompressedGraph::kHighDegreePartLength);

      uint8_t *part_ptr = _cur_compressed_edges;
      _cur_compressed_edges += sizeof(NodeID) * part_count;

      for (NodeID i = 0; i < part_count; ++i) {
        const bool last_part = (i + 1) == part_count;
        const NodeID part_length =
            last_part ? last_part_length : CompressedGraph::kHighDegreePartLength;

        auto part_begin = neighbourhood.begin() + i * CompressedGraph::kHighDegreePartLength;
        auto part_end = part_begin + part_length;

        std::uint8_t *cur_part_ptr = part_ptr + sizeof(NodeID) * i;
        *((NodeID *)cur_part_ptr) = static_cast<NodeID>(_cur_compressed_edges - part_ptr);

        std::vector<std::pair<NodeID, EdgeWeight>> part_neighbourhood(part_begin, part_end);
        add_edges(node, nullptr, part_neighbourhood);
      }

      _num_high_degree_nodes += 1;
      _num_high_degree_parts += part_count;
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
  // Store in the last entry of the node array the offset one after the last byte belonging to the
  // last node.
  _nodes.write(_nodes.size() - 1, static_cast<EdgeID>(_cur_compressed_edges - _compressed_edges));

  // Store at the end of the compressed edge array the (gap of the) id of the last edge. This
  // ensures that the the degree of the last node can be computed from the difference between the
  // last two first edge ids.
  const EdgeID last_edge = _edge_count;
  if constexpr (CompressedGraph::kIsolatedNodesSeparation) {
    if (_first_isolated_node) {
      const EdgeID last_edge_gap = last_edge - (_nodes.size() - 1);

      if constexpr (CompressedGraph::kIntervalEncoding) {
        _cur_compressed_edges += marked_varint_encode(last_edge_gap, false, _cur_compressed_edges);
      } else {
        _cur_compressed_edges += varint_encode(last_edge_gap, _cur_compressed_edges);
      }
    } else {
      _nodes.write(_nodes.size() - 1, _effective_last_edge_offset);
    }
  } else {
    if constexpr (CompressedGraph::kIntervalEncoding) {
      _cur_compressed_edges += marked_varint_encode(last_edge, false, _cur_compressed_edges);
    } else {
      _cur_compressed_edges += varint_encode(last_edge, _cur_compressed_edges);
    }
  }

  // Add an additional 15 bytes to the compressed edge array when stream encoding is enabled to
  // avoid a possible segmentation fault as the stream decoder reads 16-byte chunks.
  if constexpr (CompressedGraph::kStreamEncoding) {
    _cur_compressed_edges += 15;
  }

  const std::size_t stored_bytes =
      static_cast<std::size_t>(_cur_compressed_edges - _compressed_edges);
  RECORD("compressed_edges")
  StaticArray<std::uint8_t> compressed_edges(stored_bytes, _compressed_edges);

  if constexpr (kHeapProfiling) {
    heap_profiler::HeapProfiler::global().record_alloc(_compressed_edges, stored_bytes);
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
      std::move(compressed_edges),
      std::move(_node_weights),
      std::move(_edge_weights),
      _edge_count,
      _max_degree,
      _sorted,
      _num_high_degree_nodes,
      _num_high_degree_parts,
      _num_interval_nodes,
      _num_intervals
  );
}

std::size_t CompressedGraphBuilder::edge_array_size() const {
  return static_cast<std::size_t>(_cur_compressed_edges - _compressed_edges);
}

std::int64_t CompressedGraphBuilder::total_node_weight() const {
  return _total_node_weight;
}

std::int64_t CompressedGraphBuilder::total_edge_weight() const {
  return _total_edge_weight;
}

void CompressedGraphBuilder::add_edges(
    const NodeID node,
    std::uint8_t *marked_byte,
    std::vector<std::pair<NodeID, EdgeWeight>> &neighbourhood
) {
  const auto store_edge_weight = [&](const EdgeWeight edge_weight) {
    _total_edge_weight += edge_weight;
    _edge_weights[_edge_count++] = edge_weight;
  };

  NodeID local_degree = neighbourhood.size();

  // Find intervals [i, j] of consecutive adjacent nodes i, i + 1, ..., j - 1, j of length at least
  // kIntervalLengthTreshold. Instead of storing all nodes, only encode the left extreme i and the
  // length j - i + 1. Left extremes are stored using the differences between each left extreme and
  // the previous right extreme minus 2 (because there must be at least one integer between the end
  // of an interval and the beginning of the next one), except the first left extreme, which is
  // stored directly. The lengths are decremented by kIntervalLengthTreshold, the minimum length of
  // an interval.
  if constexpr (CompressedGraph::kIntervalEncoding) {
    NodeID interval_count = 0;

    // Save the pointer to the interval count and skip the amount of bytes needed to store the
    // interval count as we can only determine the amount of intervals after finding all of
    // them.
    std::uint8_t *interval_count_ptr = _cur_compressed_edges;
    _cur_compressed_edges += sizeof(NodeID);

    if (local_degree >= CompressedGraph::kIntervalLengthTreshold) {
      NodeID interval_len = 1;
      NodeID previous_right_extreme = 2;
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

              _cur_compressed_edges += varint_encode(left_extreme_gap, _cur_compressed_edges);
              _cur_compressed_edges += varint_encode(interval_length_gap, _cur_compressed_edges);

              for (NodeID i = 0; i < interval_len; ++i) {
                std::pair<NodeID, EdgeWeight> &incident_edge = *(iter + 1 + i - interval_len);

                // Set the adjacent node to a special value, which indicates for the gap encoder
                // that the node has been encoded through an interval.
                incident_edge.first = std::numeric_limits<NodeID>::max();

                if (_store_edge_weights) {
                  store_edge_weight(incident_edge.second);
                }
              }

              previous_right_extreme = adjacent_node;

              local_degree -= interval_len;
              interval_count += 1;
            }

            interval_len = 1;
          }
        }

        prev_adjacent_node = adjacent_node;
      }
    }

    // If intervals have been encoded store the interval count and set the bit in the marked byte
    // indicating that interval encoding has been used for the neighbourhood if the marked byte is
    // given. Otherwise, fix the amount of bytes stored as we don't store the interval count if no
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
      _num_interval_nodes += 1;
      _num_intervals += interval_count;
    }

    // If all incident edges have been compressed using intervals then gap encoding cannot be
    // applied.
    if (local_degree == 0) {
      return;
    }
  }

  // Store the remaining adjacent nodes using gap encoding. That is instead of directly storing the
  // nodes v_1, v_2, ..., v_{k - 1}, v_k, store the gaps v_1 - u, v_2 - v_1 - 1, ..., v_k - v_{k -
  // 1} - 1 between the nodes, where u is the source node. Note that all gaps except the first one
  // have to be positive as we sorted the nodes in ascending order. Thus, only for the first gap
  // the sign is additionally stored.
  auto iter = neighbourhood.begin();

  // Go to the first adjacent node that has not been encoded through an interval.
  while ((*iter).first == std::numeric_limits<NodeID>::max()) {
    ++iter;
  }

  const auto [first_adjacent_node, first_edge_weight] = *iter++;
  const SignedID first_gap = first_adjacent_node - static_cast<SignedID>(node);
  _cur_compressed_edges += signed_varint_encode(first_gap, _cur_compressed_edges);

  if (_store_edge_weights) {
    store_edge_weight(first_edge_weight);
  }

  VarIntRunLengthEncoder<NodeID> rl_encoder(_cur_compressed_edges);
  VarIntStreamEncoder<NodeID> sv_encoder(_cur_compressed_edges, local_degree - 1);

  NodeID prev_adjacent_node = first_adjacent_node;
  while (iter != neighbourhood.end()) {
    const auto [adjacent_node, edge_weight] = *iter++;

    // Skip the adjacent node since it has been encoded through an interval.
    if (adjacent_node == std::numeric_limits<NodeID>::max()) {
      continue;
    }

    const NodeID gap = adjacent_node - prev_adjacent_node - 1;
    if constexpr (CompressedGraph::kRunLengthEncoding) {
      _cur_compressed_edges += rl_encoder.add(gap);
    } else if constexpr (CompressedGraph::kStreamEncoding) {
      _cur_compressed_edges += sv_encoder.add(gap);
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
  } else if constexpr (CompressedGraph::kStreamEncoding) {
    sv_encoder.flush();
  }
}

} // namespace kaminpar::shm
