#include "kaminpar-shm/refinement/flow/flow_network/flow_network.h"

#include <numeric>
#include <span>
#include <utility>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-shm/refinement/flow/util/reverse_edge_index.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

FlowNetworkConstructor::FlowNetworkConstructor(
    const FlowNetworkConstructionContext &c_ctx,
    const PartitionedCSRGraph &p_graph,
    const CSRGraph &graph
)
    : _c_ctx(c_ctx),
      _p_graph(p_graph),
      _graph(graph) {};

FlowNetwork FlowNetworkConstructor::construct_flow_network(
    const BorderRegion &border_region,
    const BlockWeight block1_weight,
    const BlockWeight block2_weight,
    const bool run_sequentially
) {
  if (run_sequentially || _c_ctx.deterministic) {
    return sequential_construct_flow_network(border_region, block1_weight, block2_weight);
  } else {
    return parallel_construct_flow_network(border_region, block1_weight, block2_weight);
  }
}

FlowNetwork FlowNetworkConstructor::sequential_construct_flow_network(
    const BorderRegion &border_region,
    const BlockWeight block1_weight,
    const BlockWeight block2_weight
) {
  SCOPED_TIMER("Construct Flow Network");

  constexpr NodeID kSource = 0;
  constexpr NodeID kSink = 1;
  constexpr NodeID kFirstNodeID = 2;

  const BlockID block1 = border_region.block1();
  const BlockID block2 = border_region.block2();

  const NodeID num_border_region_nodes = border_region.size();

  using HashTable = DynamicRememberingFlatMap<NodeID, NodeID>;
  const std::size_t mapping_capacity = HashTable::required_capacity(num_border_region_nodes);
  HashTable global_to_local_mapping(mapping_capacity);
  HashTable local_to_global_mapping(mapping_capacity);

  TIMED_SCOPE("Initialize mappings") {
    NodeID cur_node = kFirstNodeID;

    for (BlockID terminal = 0; terminal < 2; ++terminal) {
      const std::span<const NodeID> border_region_nodes =
          (terminal == 0) ? border_region.nodes_region1() : border_region.nodes_region2();

      for (const NodeID u : border_region_nodes) {
        const NodeID u_local = cur_node++;
        global_to_local_mapping[u] = u_local;
        local_to_global_mapping[u_local] = u;
      }
    }
  };

  const NodeID num_nodes = 2 + num_border_region_nodes;
  StaticArray<EdgeID> nodes(num_nodes + 1, static_array::noinit);
  StaticArray<NodeWeight> node_weights(num_nodes, static_array::noinit);

  StaticArray<EdgeWeight> source_weights(num_nodes, static_array::noinit);
  StaticArray<EdgeWeight> sink_weights(num_nodes, static_array::noinit);

  TIMED_SCOPE("Compute node offsets") {
    NodeID cur_node = kFirstNodeID;

    EdgeID num_source_edges = 0;
    EdgeID num_sink_edges = 0;
    for (BlockID terminal = 0; terminal < 2; ++terminal) {
      const std::span<const NodeID> border_region_nodes =
          (terminal == 0) ? border_region.nodes_region1() : border_region.nodes_region2();

      for (const NodeID u : border_region_nodes) {
        const NodeID u_local = cur_node;
        KASSERT(u_local == global_to_local_mapping[u]);

        EdgeID num_neighbors = 0;
        EdgeWeight source_weight = 0;
        EdgeWeight sink_weight = 0;
        _graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
          if (global_to_local_mapping.contains(v)) {
            num_neighbors += 1;
            return;
          }

          const BlockID v_block = _p_graph.block(v);
          source_weight += (v_block == block1) ? w : 0;
          sink_weight += (v_block == block2) ? w : 0;
        });

        sink_weights[u_local] = sink_weight;
        if (sink_weight > 0) {
          num_neighbors += 1;
          num_sink_edges += 1;
        }

        source_weights[u_local] = source_weight;
        if (source_weight > 0) {
          num_neighbors += 1;
          num_source_edges += 1;
        }

        const NodeWeight u_weight = _graph.node_weight(u);
        nodes[u_local] = num_neighbors;
        node_weights[u_local] = u_weight;

        cur_node += 1;
      }
    }

    nodes[kSource] = num_source_edges;
    node_weights[kSource] = block1_weight - border_region.weight_region1();

    nodes[kSink] = num_sink_edges;
    node_weights[kSink] = block2_weight - border_region.weight_region2();

    nodes.back() = 0;
    std::partial_sum(nodes.begin(), nodes.end(), nodes.begin());
  };

  const EdgeID num_edges = nodes.back();
  StaticArray<NodeID> edges(num_edges, static_array::noinit);
  StaticArray<EdgeWeight> edge_weights(num_edges, static_array::noinit);
  StaticArray<EdgeID> reverse_edges(num_edges, static_array::noinit);

  EdgeWeight cut_value = 0;
  TIMED_SCOPE("Compute edges") {
    NodeID cur_node = kFirstNodeID;

    EdgeID cur_source_edge = nodes[kSource];
    EdgeID cur_sink_edge = nodes[kSink];
    for (BlockID terminal = 0; terminal < 2; ++terminal) {
      const bool source_side = terminal == 0;
      const bool sink_side = terminal == 1;

      const std::span<const NodeID> border_region_nodes =
          (terminal == 0) ? border_region.nodes_region1() : border_region.nodes_region2();
      for (const NodeID u : border_region_nodes) {
        const NodeID u_local = cur_node;
        KASSERT(u_local == global_to_local_mapping[u]);

        EdgeID u_edge = nodes[u_local];
        _graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
          if (auto it = global_to_local_mapping.get_if_contained(v);
              it != global_to_local_mapping.end()) {
            const NodeID v_local = *it;
            if (u_local >= v_local) {
              return;
            }

            u_edge -= 1;
            edges[u_edge] = v_local;
            edge_weights[u_edge] = w;

            const EdgeID v_edge = --nodes[v_local];
            edges[v_edge] = u_local;
            edge_weights[v_edge] = w;

            reverse_edges[u_edge] = v_edge;
            reverse_edges[v_edge] = u_edge;

            if (source_side) {
              cut_value += border_region.region2_contains(v) ? w : 0;
            } else if (sink_side) {
              cut_value += border_region.region1_contains(v) ? w : 0;
            }
          }
        });

        const EdgeWeight sink_edge_weight = sink_weights[u_local];
        if (sink_edge_weight > 0) {
          u_edge -= 1;
          edges[u_edge] = kSink;
          edge_weights[u_edge] = sink_edge_weight;

          cur_sink_edge -= 1;
          edges[cur_sink_edge] = u_local;
          edge_weights[cur_sink_edge] = sink_edge_weight;

          reverse_edges[u_edge] = cur_sink_edge;
          reverse_edges[cur_sink_edge] = u_edge;

          cut_value += source_side ? sink_edge_weight : 0;
        }

        const EdgeWeight source_edge_weight = source_weights[u_local];
        if (source_edge_weight > 0) {
          u_edge -= 1;
          edges[u_edge] = kSource;
          edge_weights[u_edge] = source_edge_weight;

          cur_source_edge -= 1;
          edges[cur_source_edge] = u_local;
          edge_weights[cur_source_edge] = source_edge_weight;

          reverse_edges[u_edge] = cur_source_edge;
          reverse_edges[cur_source_edge] = u_edge;

          cut_value += sink_side ? source_edge_weight : 0;
        }

        nodes[u_local] = u_edge;
        cur_node += 1;
      }
    }

    nodes[kSource] = cur_source_edge;
    nodes[kSink] = cur_sink_edge;
  };

  CSRGraph flow_network_graph(
      CSRGraph::seq(),
      std::move(nodes),
      std::move(edges),
      std::move(node_weights),
      std::move(edge_weights)
  );

  KASSERT(
      debug::validate_graph(flow_network_graph),
      "constructed an invalid flow network",
      assert::heavy
  );
  KASSERT(
      debug::is_valid_reverse_edge_index(flow_network_graph, reverse_edges),
      "constructed an invalid reverse edge index",
      assert::heavy
  );
  KASSERT(
      [&] {
        for (const NodeID u : flow_network_graph.nodes()) {
          if (u == kSource || u == kSink) {
            continue;
          }

          if (global_to_local_mapping[local_to_global_mapping[u]] != u) {
            return false;
          }
        }

        return true;
      }(),
      "constructed an global to local mapping",
      assert::heavy
  );
  KASSERT(
      [&] {
        for (BlockID terminal = 0; terminal < 2; ++terminal) {
          const std::span<const NodeID> border_region_nodes =
              (terminal == 0) ? border_region.nodes_region1() : border_region.nodes_region2();

          for (const NodeID u : border_region_nodes) {
            if (local_to_global_mapping[global_to_local_mapping[u]] != u) {
              return false;
            }
          }
        }

        return true;
      }(),
      "constructed an local to global mapping",
      assert::heavy
  );

  return FlowNetwork(
      kSource,
      kSink,
      std::move(flow_network_graph),
      std::move(reverse_edges),
      std::move(global_to_local_mapping),
      std::move(local_to_global_mapping),
      block1_weight,
      block2_weight,
      cut_value
  );
}

namespace {

class NeighborhoodBuffer {
  static constexpr EdgeID kSize = 1024;

public:
  NeighborhoodBuffer(
      std::span<EdgeID> nodes,
      std::span<NodeID> edges,
      std::span<EdgeID> reverse_edges,
      std::span<EdgeWeight> edge_weights
  )
      : _nodes(nodes),
        _edges(edges),
        _reverse_edges(reverse_edges),
        _edge_weights(edge_weights) {}

  void add(const NodeID v, const EdgeID e_reverse, const EdgeWeight w) {
    KASSERT(_num_buffered_edges < kSize);

    _edge_buffer[_num_buffered_edges] = v;
    _reverse_edge_buffer[_num_buffered_edges] = e_reverse;
    _edge_weight_buffer[_num_buffered_edges] = w;

    _num_buffered_edges += 1;
  }

  void flush(const NodeID u) {
    const NodeID num_neighbors = _num_buffered_edges;
    const EdgeID u_edge = __atomic_sub_fetch(&_nodes[u], num_neighbors, __ATOMIC_RELAXED);

    for (NodeID i = 0; i < num_neighbors; ++i) {
      const EdgeID e = u_edge + i;

      _edges[e] = _edge_buffer[i];
      _edge_weights[e] = _edge_weight_buffer[i];

      const EdgeID e_reverse = _reverse_edge_buffer[i];
      _reverse_edges[e] = e_reverse;
      _reverse_edges[e_reverse] = e;
    }

    _num_buffered_edges = 0;
  }

  [[nodiscard]] bool empty() const {
    return _num_buffered_edges == 0;
  }

  [[nodiscard]] bool full() const {
    return _num_buffered_edges == kSize;
  }

private:
  std::span<EdgeID> _nodes;
  std::span<NodeID> _edges;
  std::span<EdgeID> _reverse_edges;
  std::span<EdgeWeight> _edge_weights;

  NodeID _num_buffered_edges;
  std::array<NodeID, kSize> _edge_buffer;
  std::array<EdgeID, kSize> _reverse_edge_buffer;
  std::array<EdgeWeight, kSize> _edge_weight_buffer;
};

} // namespace

[[nodiscard]] FlowNetwork FlowNetworkConstructor::parallel_construct_flow_network(
    const BorderRegion &border_region,
    const BlockWeight block1_weight,
    const BlockWeight block2_weight
) {
  SCOPED_TIMER("Construct Flow Network");

  constexpr NodeID kSource = 0;
  constexpr NodeID kSink = 1;
  constexpr NodeID kFirstNodeID = 2;

  const BlockID block1 = border_region.block1();
  const BlockID block2 = border_region.block2();

  const NodeID num_border_region_nodes = border_region.size();

  using HashTable = DynamicRememberingFlatMap<NodeID, NodeID>;
  const std::size_t mapping_capacity = HashTable::required_capacity(num_border_region_nodes);
  HashTable global_to_local_mapping(mapping_capacity);
  HashTable local_to_global_mapping(mapping_capacity);

  TIMED_SCOPE("Initialize mappings") {
    NodeID cur_node = kFirstNodeID;
    for (BlockID terminal = 0; terminal < 2; ++terminal) {
      const std::span<const NodeID> border_region_nodes =
          (terminal == 0) ? border_region.nodes_region1() : border_region.nodes_region2();

      for (const NodeID u : border_region_nodes) {
        const NodeID u_local = cur_node++;
        global_to_local_mapping[u] = u_local;
        local_to_global_mapping[u_local] = u;
      }
    }
  };

  const NodeID num_nodes = 2 + num_border_region_nodes;
  StaticArray<EdgeID> nodes(num_nodes + 1, static_array::noinit);
  StaticArray<NodeWeight> node_weights(num_nodes, static_array::noinit);

  StaticArray<EdgeWeight> source_weights(num_nodes, static_array::seq);
  StaticArray<EdgeWeight> sink_weights(num_nodes, static_array::seq);

  TIMED_SCOPE("Compute node offsets") {
    tbb::enumerable_thread_specific<EdgeID> num_source_edges_ets;
    tbb::enumerable_thread_specific<EdgeID> num_sink_edges_ets;
    for (BlockID terminal = 0; terminal < 2; ++terminal) {
      const bool source_side = terminal == 0;
      const std::span<const NodeID> border_region_nodes =
          source_side ? border_region.nodes_region1() : border_region.nodes_region2();

      const NodeID u_local_offset = kFirstNodeID + (source_side ? 0 : border_region.size_region1());
      tbb::parallel_for<std::size_t>(0, border_region_nodes.size(), [&](const std::size_t i) {
        const NodeID u = border_region_nodes[i];
        const NodeID u_local = u_local_offset + i;

        EdgeID num_neighbors = 0;
        EdgeWeight source_weight = 0;
        EdgeWeight sink_weight = 0;
        _graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
          if (global_to_local_mapping.contains(v)) {
            num_neighbors += 1;
            return;
          }

          const BlockID v_block = _p_graph.block(v);
          source_weight += (v_block == block1) ? w : 0;
          sink_weight += (v_block == block2) ? w : 0;
        });

        if (sink_weight > 0) {
          num_neighbors += 1;
          num_sink_edges_ets.local() += 1;
          sink_weights[u_local] = sink_weight;
        }

        if (source_weight > 0) {
          num_neighbors += 1;
          num_source_edges_ets.local() += 1;
          source_weights[u_local] = source_weight;
        }

        const NodeWeight u_weight = _graph.node_weight(u);
        nodes[u_local] = num_neighbors;
        node_weights[u_local] = u_weight;
      });
    }

    nodes[kSource] = num_source_edges_ets.combine(std::plus<>());
    node_weights[kSource] = block1_weight - border_region.weight_region1();

    nodes[kSink] = num_sink_edges_ets.combine(std::plus<>());
    node_weights[kSink] = block2_weight - border_region.weight_region2();

    nodes.back() = 0;
    parallel::prefix_sum(nodes.begin(), nodes.end(), nodes.begin());
  };

  const EdgeID num_edges = nodes.back();
  StaticArray<NodeID> edges(num_edges, static_array::noinit);
  StaticArray<EdgeID> reverse_edges(num_edges, static_array::noinit);
  StaticArray<EdgeWeight> edge_weights(num_edges, static_array::noinit);

  tbb::enumerable_thread_specific<EdgeID> cut_value_ets;
  TIMED_SCOPE("Compute edges") {
    tbb::enumerable_thread_specific<NeighborhoodBuffer> source_neighborhood_buffer_ets([&] {
      return NeighborhoodBuffer(nodes, edges, reverse_edges, edge_weights);
    });
    tbb::enumerable_thread_specific<NeighborhoodBuffer> sink_neighborhood_buffer_ets([&] {
      return NeighborhoodBuffer(nodes, edges, reverse_edges, edge_weights);
    });
    tbb::enumerable_thread_specific<NeighborhoodBuffer> neighborhood_buffer_ets([&] {
      return NeighborhoodBuffer(nodes, edges, reverse_edges, edge_weights);
    });

    for (BlockID terminal = 0; terminal < 2; ++terminal) {
      const bool source_side = terminal == 0;
      const bool sink_side = terminal == 1;

      const std::span<const NodeID> border_region_nodes =
          source_side ? border_region.nodes_region1() : border_region.nodes_region2();
      const NodeID b_n = border_region_nodes.size();

      const NodeID u_local_offset = kFirstNodeID + (source_side ? 0 : border_region.size_region1());
      tbb::parallel_for(tbb::blocked_range<std::size_t>(0, b_n), [&](const auto &range) {
        EdgeWeight local_cut_value = 0;

        NeighborhoodBuffer &source_buffer = source_neighborhood_buffer_ets.local();
        NeighborhoodBuffer &sink_buffer = sink_neighborhood_buffer_ets.local();

        NeighborhoodBuffer &buffer = neighborhood_buffer_ets.local();
        for (std::size_t i = range.begin(); i != range.end(); ++i) {
          const NodeID u = border_region_nodes[i];
          const NodeID u_local = u_local_offset + i;

          _graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
            if (auto it = global_to_local_mapping.get_if_contained(v);
                it != global_to_local_mapping.end()) {
              const NodeID v_local = *it;
              if (u_local >= v_local) {
                return;
              }

              const EdgeID v_edge = __atomic_sub_fetch(&nodes[v_local], 1, __ATOMIC_RELAXED);
              edges[v_edge] = u_local;
              edge_weights[v_edge] = w;

              buffer.add(v_local, v_edge, w);
              if (buffer.full()) [[unlikely]] {
                buffer.flush(u_local);
              }

              local_cut_value += (source_side && border_region.region2_contains(v)) ? w : 0;
              local_cut_value += (sink_side && border_region.region1_contains(v)) ? w : 0;
            }
          });

          if (!buffer.empty()) [[likely]] {
            buffer.flush(u_local);
          }

          const EdgeWeight sink_edge_weight = sink_weights[u_local];
          if (sink_edge_weight > 0) {
            const EdgeID u_edge = __atomic_sub_fetch(&nodes[u_local], 1, __ATOMIC_RELAXED);
            edges[u_edge] = kSink;
            edge_weights[u_edge] = sink_edge_weight;

            sink_buffer.add(u_local, u_edge, sink_edge_weight);
            if (sink_buffer.full()) [[unlikely]] {
              sink_buffer.flush(kSink);
            }

            local_cut_value += source_side ? sink_edge_weight : 0;
          }

          const EdgeWeight source_edge_weight = source_weights[u_local];
          if (source_edge_weight > 0) {
            const EdgeID u_edge = __atomic_sub_fetch(&nodes[u_local], 1, __ATOMIC_RELAXED);
            edges[u_edge] = kSource;
            edge_weights[u_edge] = source_edge_weight;

            source_buffer.add(u_local, u_edge, source_edge_weight);
            if (source_buffer.full()) [[unlikely]] {
              source_buffer.flush(kSource);
            }
            local_cut_value += sink_side ? source_edge_weight : 0;
          }
        }

        cut_value_ets.local() += local_cut_value;
      });
    }

    tbb::parallel_for(source_neighborhood_buffer_ets.range(), [&](const auto &local_buffers) {
      for (auto &local_buffer : local_buffers) {
        if (!local_buffer.empty()) {
          local_buffer.flush(kSource);
        }
      }
    });
    tbb::parallel_for(sink_neighborhood_buffer_ets.range(), [&](const auto &local_buffers) {
      for (auto &local_buffer : local_buffers) {
        if (!local_buffer.empty()) {
          local_buffer.flush(kSink);
        }
      }
    });
  };

  CSRGraph flow_network_graph(
      std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights)
  );
  KASSERT(
      debug::validate_graph(flow_network_graph),
      "constructed an invalid flow network",
      assert::heavy
  );
  KASSERT(
      debug::is_valid_reverse_edge_index(flow_network_graph, reverse_edges),
      "constructed an invalid reverse edge index",
      assert::heavy
  );

  return FlowNetwork(
      kSource,
      kSink,
      std::move(flow_network_graph),
      std::move(reverse_edges),
      std::move(global_to_local_mapping),
      std::move(local_to_global_mapping),
      block1_weight,
      block2_weight,
      cut_value_ets.combine(std::plus<>())
  );
}

} // namespace kaminpar::shm
