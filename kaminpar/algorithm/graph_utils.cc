#include "graph_utils.h"

#include "datastructure/graph.h"
#include "datastructure/map.h"
#include "datastructure/marker.h"
#include "datastructure/static_array.h"
#include "definitions.h"
#include "parallel.h"
#include "utility/timer.h"

#include <mutex>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

namespace kaminpar {
void copy_subgraph_partitions(PartitionedGraph &p_graph,
                              const scalable_vector<StaticArray<BlockID>> &p_subgraph_partitions, const BlockID k_prime,
                              const BlockID input_k, const scalable_vector<NodeID> &mapping) {
  scalable_vector<BlockID> k0(p_graph.k() + 1, k_prime / p_graph.k());
  k0[0] = 0;

  scalable_vector<BlockID> final_ks(k_prime, 1);

  // we are done partitioning? --> use final_ks
  if (k_prime == input_k) { std::copy(p_graph.final_ks().begin(), p_graph.final_ks().end(), k0.begin() + 1); }

  parallel::prefix_sum(k0.begin(), k0.end(), k0.begin()); // blocks of old block i start at k0[i]

  // we are not done partitioning?
  if (k_prime != input_k) {
    ALWAYS_ASSERT(math::is_power_of_2(k_prime));
    const BlockID k_per_block = k_prime / p_graph.k();
    tbb::parallel_for(static_cast<BlockID>(0), p_graph.k(), [&](const BlockID b) {
      const BlockID base = k0[b];
      final_ks[base] = p_graph.final_k(b);

      for (BlockID k = k_per_block; k > 1; k /= 2) {
        for (BlockID kp = 0; kp < k_per_block; kp += k) {
          std::tie(final_ks[base + kp], final_ks[base + kp + k / 2]) = math::split_integral(final_ks[base + kp]);
        }
      }
    });
  }

  p_graph.change_k(k_prime);
  tbb::parallel_for(static_cast<NodeID>(0), p_graph.n(), [&](const NodeID &u) {
    const BlockID b = p_graph._partition[u];
    const NodeID s_u = mapping[u];
    p_graph._partition[u] = k0[b] + p_subgraph_partitions[b][s_u];
  });

  p_graph.set_final_ks(std::move(final_ks));
  p_graph.reinit_block_weights();
}

/*
 * Builds a block-induced subgraph for each block of a partitioned graph. Return type contains a mapping that maps
 * nodes from p_graph to nodes in the respective subgraph; we need this because the order in which nodes in subgraphs
 * appear is non-deterministic due to parallelization.
 */
SubgraphExtractionResult extract_subgraphs(const PartitionedGraph &p_graph, SubgraphMemory &subgraph_memory) {
  const Graph &graph = p_graph.graph();

  const auto t_allocation = SIMPLE_TIMER_START();
  START_TIMER(TIMER_ALLOCATION);
  scalable_vector<NodeID> mapping(p_graph.n());
  scalable_vector<SubgraphMemoryStartPosition> start_positions(p_graph.k() + 1);

  using AtomicNodeCounter = parallel::IntegralAtomicWrapper<NodeID>;
  std::vector<AtomicNodeCounter, tbb::cache_aligned_allocator<AtomicNodeCounter>> bucket_index(p_graph.n());

  scalable_vector<Graph> subgraphs(p_graph.k());
  STOP_TIMER();
  SIMPLE_TIMER_STOP("Subgraph extraction allocation", t_allocation);

  // count number of nodes and edges in each block
  START_TIMER("Count block size");
  tbb::enumerable_thread_specific<scalable_vector<NodeID>> tl_num_nodes_in_block{
      [&] { return scalable_vector<NodeID>(p_graph.k()); }};
  tbb::enumerable_thread_specific<scalable_vector<EdgeID>> tl_num_edges_in_block{
      [&] { return scalable_vector<EdgeID>(p_graph.k()); }};

  tbb::parallel_for(tbb::blocked_range<NodeID>(0, graph.n()), [&](auto &r) {
    auto &num_nodes_in_block = tl_num_nodes_in_block.local();
    auto &num_edges_in_block = tl_num_edges_in_block.local();

    for (NodeID u = r.begin(); u != r.end(); ++u) {
      const BlockID u_block = p_graph.block(u);
      ++num_nodes_in_block[u_block];
      for (const NodeID v : graph.adjacent_nodes(u)) {
        if (p_graph.block(v) == u_block) { ++num_edges_in_block[u_block]; }
      }
    }
  });
  STOP_TIMER();

  START_TIMER("Merge block sizes");
  tbb::parallel_for(static_cast<BlockID>(0), p_graph.k(), [&](const BlockID b) {
    NodeID num_nodes = p_graph.final_k(b); // padding for sequential subgraph extraction
    EdgeID num_edges = 0;
    for (auto &local_num_nodes : tl_num_nodes_in_block) { num_nodes += local_num_nodes[b]; }
    for (auto &local_num_edges : tl_num_edges_in_block) { num_edges += local_num_edges[b]; }
    start_positions[b + 1].nodes_start_pos = num_nodes;
    start_positions[b + 1].edges_start_pos = num_edges;
  });
  parallel::prefix_sum(start_positions.begin(), start_positions.end(), start_positions.begin());
  STOP_TIMER();

  //  for (const BlockID b : p_graph.blocks()) {
  //    LOG << "b " << b << " nstart " << start_positions[b].nodes_start_pos << " mstart " << start_positions[b].edges_start_pos;
  //  }
  //    LOG << "b " << p_graph.k() << " nstart " << start_positions[p_graph.k()].nodes_start_pos << " mstart " << start_positions[p_graph.k()].edges_start_pos;

  // build temporary bucket array in nodes array
  START_TIMER("Build bucket array");
  tbb::parallel_for(static_cast<NodeID>(0), p_graph.n(), [&](const NodeID u) {
    const BlockID b = p_graph.block(u);
    const NodeID pos_in_subgraph = bucket_index[b]++;
    const NodeID pos = start_positions[b].nodes_start_pos + pos_in_subgraph;
    subgraph_memory.nodes[pos] = u;
    mapping[u] = pos_in_subgraph; // concurrent random access write
  });
  STOP_TIMER();

  const bool is_node_weighted = p_graph.graph().is_node_weighted();
  const bool is_edge_weighted = p_graph.graph().is_edge_weighted();

  // build graph
  START_TIMER("Construct subgraphs");
  tbb::parallel_for(static_cast<BlockID>(0), p_graph.k(), [&](const BlockID b) {
    const NodeID nodes_start_pos = start_positions[b].nodes_start_pos;
    EdgeID e = 0;                                  // edge = in subgraph
    for (NodeID u = 0; u < bucket_index[b]; ++u) { // u = in subgraph
      const NodeID pos = nodes_start_pos + u;
      const NodeID u_prime = subgraph_memory.nodes[pos]; // u_prime = in graph
      subgraph_memory.nodes[pos] = e;
      if (is_node_weighted) { subgraph_memory.node_weights[pos] = graph.node_weight(u_prime); }

      const EdgeID e0 = start_positions[b].edges_start_pos;

      for (const auto [e_prime, v_prime] : graph.neighbors(u_prime)) { // e_prime, v_prime = in graph
        if (p_graph.block(v_prime) == b) {                             // only keep internal edges
          if (is_edge_weighted) { subgraph_memory.edge_weights[e0 + e] = graph.edge_weight(e_prime); }
          subgraph_memory.edges[e0 + e] = mapping[v_prime];
          ++e;
        }
      }
    }

    subgraph_memory.nodes[nodes_start_pos + bucket_index[b]] = e;
  });
  STOP_TIMER();

  START_TIMER("Create graph objects");
  tbb::parallel_for(static_cast<BlockID>(0), p_graph.k(), [&](const BlockID b) {
    const NodeID n0 = start_positions[b].nodes_start_pos;
    const EdgeID m0 = start_positions[b].edges_start_pos;
    const NodeID n = start_positions[b + 1].nodes_start_pos - n0 - p_graph.final_k(b);
    const EdgeID m = start_positions[b + 1].edges_start_pos - m0;

    StaticArray<EdgeID> nodes(n0, n + 1, subgraph_memory.nodes);
    StaticArray<NodeID> edges(m0, m, subgraph_memory.edges);
    StaticArray<NodeWeight> node_weights(is_node_weighted * n0, is_node_weighted * n, subgraph_memory.node_weights);
    StaticArray<EdgeWeight> edge_weights(is_edge_weighted * m0, is_edge_weighted * m, subgraph_memory.edge_weights);
    subgraphs[b] = Graph{std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights)};
  });
  STOP_TIMER();

  HEAVY_ASSERT([&] {
    for (const BlockID b : p_graph.blocks()) {
      LOG << "Validate " << b;
      ALWAYS_ASSERT(validate_graph(subgraphs[b]));
    }
    return true;
  });

  return {std::move(subgraphs), std::move(mapping), std::move(start_positions)};
}

bool validate_graph(const Graph &graph) {
  LOG << "Validate n=" << graph.n() << " m=" << graph.m();

  for (NodeID u = 0; u < graph.n(); ++u) {
    ALWAYS_ASSERT(graph.raw_nodes()[u] <= graph.raw_nodes()[u + 1])
        << V(u) << V(graph.raw_nodes()[u]) << V(graph.raw_nodes()[u + 1]);
  }

  for (const NodeID u : graph.nodes()) {
    for (const auto [e, v] : graph.neighbors(u)) {
      ALWAYS_ASSERT(v < graph.n());
      bool found_reverse = false;
      for (const auto [e_prime, u_prime] : graph.neighbors(v)) {
        ALWAYS_ASSERT(u_prime < graph.n());
        if (u != u_prime) { continue; }
        ALWAYS_ASSERT(graph.edge_weight(e) == graph.edge_weight(e_prime))
            << V(e) << V(graph.edge_weight(e)) << V(e_prime) << V(graph.edge_weight(e_prime)) << " Edge from " << u
            << " --> " << v << " --> " << u_prime;
        found_reverse = true;
        break;
      }
      ALWAYS_ASSERT(found_reverse) << u << " --> " << v << " exists with edge " << e << " but no reverse edge found!";
    }
  }
  return true;
}

SequentialSubgraphExtractionResult extract_subgraphs_sequential(const PartitionedGraph &p_graph,
                                                                const SubgraphMemoryStartPosition memory_position,
                                                                SubgraphMemory &subgraph_memory,
                                                                TemporarySubgraphMemory &tmp_subgraph_memory) {
  ALWAYS_ASSERT(p_graph.k() == 2) << "Only suitable for bipartitions!";
  ALWAYS_ASSERT(tmp_subgraph_memory.in_use == false);
  tmp_subgraph_memory.in_use = true;

  const bool is_node_weighted = p_graph.graph().is_node_weighted();
  const bool is_edge_weighted = p_graph.graph().is_edge_weighted();

  const BlockID final_k = p_graph.final_k(0) + p_graph.final_k(1);
  tmp_subgraph_memory.ensure_size_nodes(p_graph.n() + final_k, is_node_weighted);

  auto &nodes = tmp_subgraph_memory.nodes;
  auto &edges = tmp_subgraph_memory.edges;
  auto &node_weights = tmp_subgraph_memory.node_weights;
  auto &edge_weights = tmp_subgraph_memory.edge_weights;
  auto &mapping = tmp_subgraph_memory.mapping;

  std::array<NodeID, 2> s_n{0, 0};
  std::array<EdgeID, 2> s_m{0, 0};

  // find graph sizes
  for (const NodeID u : p_graph.nodes()) {
    const BlockID b = p_graph.block(u);
    tmp_subgraph_memory.mapping[u] = s_n[b]++;

    for (const auto [e, v] : p_graph.neighbors(u)) {
      if (p_graph.block(v) == b) { ++s_m[b]; }
    }
  }

  // start position of subgraph[1] in common memory ds
  const NodeID n1 = s_n[0] + p_graph.final_k(0);
  const EdgeID m1 = s_m[0];

  nodes[0] = 0;
  nodes[n1] = 0;
  tmp_subgraph_memory.ensure_size_edges(s_m[0] + s_m[1], is_edge_weighted);

  // build extract graphs in temporary memory buffer
  std::array<EdgeID, 2> next_edge_id{0, 0};

  for (const NodeID u : p_graph.nodes()) {
    const BlockID b = p_graph.block(u);

    const NodeID n0 = b * n1; // either 0 or s_n[0] + final_k(0)
    const EdgeID m0 = b * m1; // either 0 or s_m[0]

    for (const auto [e, v] : p_graph.neighbors(u)) {
      if (p_graph.block(v) == b) {
        edges[m0 + next_edge_id[b]] = mapping[v];
        if (is_edge_weighted) { edge_weights[m0 + next_edge_id[b]] = p_graph.edge_weight(e); }
        ++next_edge_id[b];
      }
    }

    nodes[n0 + mapping[u] + 1] = next_edge_id[b];
    if (is_node_weighted) { node_weights[n0 + mapping[u]] = p_graph.node_weight(u); }
  }

  // copy graphs to subgraph_memory at memory_position
  // THIS OPERATION OVERWRITES p_graph!
  std::copy(nodes.begin(), nodes.begin() + p_graph.n() + final_k,
            subgraph_memory.nodes.begin() + memory_position.nodes_start_pos);
  std::copy(edges.begin(), edges.begin() + s_m[0] + s_m[1],
            subgraph_memory.edges.begin() + memory_position.edges_start_pos);
  if (is_node_weighted) {
    std::copy(node_weights.begin(), node_weights.begin() + p_graph.n() + final_k,
              subgraph_memory.node_weights.begin() + memory_position.nodes_start_pos);
  }
  if (is_edge_weighted) {
    std::copy(edge_weights.begin(), edge_weights.begin() + s_m[0] + s_m[1],
              subgraph_memory.edge_weights.begin() + memory_position.edges_start_pos);
  }

  tmp_subgraph_memory.in_use = false;

  std::array<SubgraphMemoryStartPosition, 2> subgraph_positions;
  subgraph_positions[0].nodes_start_pos = memory_position.nodes_start_pos;
  subgraph_positions[0].edges_start_pos = memory_position.edges_start_pos;
  subgraph_positions[1].nodes_start_pos = memory_position.nodes_start_pos + n1;
  subgraph_positions[1].edges_start_pos = memory_position.edges_start_pos + m1;

  auto create_graph = [&](const NodeID n0, const NodeID n, const EdgeID m0, const EdgeID m) {
    StaticArray<EdgeID> s_nodes(memory_position.nodes_start_pos + n0, n + 1, subgraph_memory.nodes);
    StaticArray<NodeID> s_edges(memory_position.edges_start_pos + m0, m, subgraph_memory.edges);
    StaticArray<NodeWeight> s_node_weights(is_node_weighted * (memory_position.nodes_start_pos + n0),
                                           is_node_weighted * n, subgraph_memory.node_weights);
    StaticArray<EdgeWeight> s_edge_weights(is_edge_weighted * (memory_position.edges_start_pos + m0),
                                           is_edge_weighted * m, subgraph_memory.edge_weights);
    return Graph{tag::seq, std::move(s_nodes), std::move(s_edges), std::move(s_node_weights),
                 std::move(s_edge_weights)};
  };

  std::array<Graph, 2> subgraphs{create_graph(0, s_n[0], 0, s_m[0]), create_graph(n1, s_n[1], m1, s_m[1])};

  return {std::move(subgraphs), std::move(subgraph_positions)};
}

struct Edge {
  NodeID target;
  EdgeWeight weight;
};

static constexpr std::size_t kChunkSize = (1 << 15);

using LocalEdgeMemoryChunk = scalable_vector<Edge>;

struct LocalEdgeMemory {
  LocalEdgeMemory() { current_chunk.reserve(kChunkSize); }

  scalable_vector<LocalEdgeMemoryChunk> chunks;
  LocalEdgeMemoryChunk current_chunk;

  std::size_t get_current_position() const { return chunks.size() * kChunkSize + current_chunk.size(); }

  void push(const NodeID c_v, const EdgeWeight weight) {
    if (current_chunk.size() == kChunkSize) { flush(); }
    current_chunk.emplace_back(c_v, weight);
  }

  const auto &get(const std::size_t position) const { return chunks[position / kChunkSize][position % kChunkSize]; }

  void flush() {
    chunks.push_back(std::move(current_chunk));
    current_chunk.clear();
    current_chunk.reserve(kChunkSize);
  }
};

struct BufferNode {
  NodeID c_u;
  std::size_t position;
  LocalEdgeMemory *chunks;
};

ContractionResult contract(const Graph &graph, const scalable_vector<NodeID> &clustering,
                           const bool leader_is_idempotent, ContractionMemoryContext m_ctx) {
  scalable_vector<parallel::IntegralAtomicWrapper<NodeID>> &buckets_index = m_ctx.buckets_index;
  scalable_vector<NodeID> &buckets = m_ctx.buckets;

  START_TIMER(TIMER_ALLOCATION);
  scalable_vector<NodeID> mapping(graph.n());
  auto &leader_mapping = leader_is_idempotent ? mapping : m_ctx.leader_mapping;
  if (!leader_is_idempotent) {
    leader_mapping.clear();
    leader_mapping.resize(graph.n());
  }
  STOP_TIMER();

  START_TIMER("Preprocessing");

  //
  // Compute a mapping from the nodes of the current graph to the nodes of the coarse graph
  // I.e., node_mapping[node u] = coarse node c_u
  //
  // Note that clustering satisfies this invariant (I): if clustering[x] = y for some node x, then clustering[y] = y
  //

  // Set node_mapping[x] = 1 iff. there is a cluster with leader x
  tbb::parallel_for(static_cast<NodeID>(0), graph.n(), [&](const NodeID u) {
    ASSERT(clustering[u] < graph.n());
    leader_mapping[clustering[u]] = 1;
  });

  // Compute prefix sum to get coarse node IDs (starting at 1!)
  parallel::prefix_sum(leader_mapping.begin(), leader_mapping.end(), leader_mapping.begin());
  const NodeID c_n = leader_mapping.back(); // number of nodes in the coarse graph

  // Assign coarse node ID to all nodes; this works due to (I)
  tbb::parallel_for(static_cast<NodeID>(0), graph.n(),
                    [&](const NodeID u) { mapping[u] = leader_mapping[clustering[u]]; });

  // So far, coarse node IDs start at 1 due to the prefix sum; hence, we must decrement them
  tbb::parallel_for(static_cast<NodeID>(0), graph.n(), [&](const NodeID u) { --mapping[u]; });

  TIMED_SCOPE(TIMER_ALLOCATION) {
    buckets_index.clear();
    buckets_index.resize(c_n + 1);
    buckets.clear();
    buckets.resize(graph.n());
  };

  //
  // Sort nodes into buckets: place all nodes belonging to coarse node i into the i-th bucket
  //
  // Count the number of nodes in each bucket, then compute the position of the bucket in the global buckets array
  // using a prefix sum, roughly 2/5-th of time on europe.osm with 2/3-th to 1/3-tel for loop to prefix sum
  tbb::parallel_for(static_cast<NodeID>(0), graph.n(),
                    [&](const NodeID u) { buckets_index[mapping[u]].fetch_add(1, std::memory_order_relaxed); });
  parallel::prefix_sum(buckets_index.begin(), buckets_index.end(), buckets_index.begin());
  ASSERT(buckets_index.back() <= graph.n());
  // Sort nodes into   buckets, roughly 3/5-th of time on europe.osm
  tbb::parallel_for(static_cast<NodeID>(0), graph.n(), [&](const NodeID u) {
    const std::size_t pos = buckets_index[mapping[u]].fetch_sub(1, std::memory_order_relaxed) - 1;
    buckets[pos] = u;
  });

  STOP_TIMER();

  //
  // Build nodes array of the coarse graph
  // - firstly, we count the degree of each coarse node
  // - secondly, we obtain the nodes array using a prefix sum
  //
  START_TIMER(TIMER_ALLOCATION);
  StaticArray<EdgeID> c_nodes{c_n + 1};
  StaticArray<NodeWeight> c_node_weights{c_n};
  StaticArray<EdgeID> unsorted_c_edges_index(c_n);
  STOP_TIMER();

  // we don't know the number of coarse edges yet, but there are hopefully much fewer than graph.m(); hence, we allocate
  // but don't initialize the memory

  tbb::enumerable_thread_specific<RatingMap<EdgeWeight>> collector{[&] { return RatingMap<EdgeWeight>(c_n); }};

  //
  // We build the coarse graph in multiple steps:
  // (1) During the first step, we compute
  //     - the node weight of each coarse node
  //     - the degree of each coarse node
  //     We can't build c_edges and c_edge_weights yet, because positioning edges in those arrays depends on c_nodes,
  //     which we only have after computing a prefix sum over all coarse node degrees
  //     Hence, we store edges and edge weights in unsorted auxiliary arrays during the first pass
  // (2) We finalize c_nodes arrays by computing a prefix sum over all coarse node degrees
  // (3) We copy coarse edges and coarse edge weights from the auxiliary arrays to c_edges and c_edge_weights
  //

  tbb::enumerable_thread_specific<LocalEdgeMemory> shared_edge_buffer;
  tbb::enumerable_thread_specific<std::vector<BufferNode>> shared_node_buffer;

  START_TIMER("Graph construction 1");
  tbb::parallel_for(static_cast<NodeID>(0), c_n, [&](const NodeID c_u) {
    auto &local_collector = collector.local();

    const std::size_t first = buckets_index[c_u];
    const std::size_t last = buckets_index[c_u + 1];

    // we need an upper bound on the number of coarse edges to choose the right hash map -- sum all node degrees
    Degree upper_bound_degree = 0;
    for (std::size_t i = first; i < last; ++i) {
      const NodeID u = buckets[i];
      upper_bound_degree += graph.degree(u);
    }
    local_collector.update_upper_bound_size(upper_bound_degree);

    // second pass over c_u bucket: compute actual degree, node weight, edges, edge weights
    NodeWeight c_u_weight = 0;
    for (std::size_t i = first; i < last; ++i) {
      const NodeID u = buckets[i];
      ASSERT(mapping[u] == c_u);

      c_u_weight += graph.node_weight(u); // coarse node weight

      // collect coarse edges
      for (const auto [e, v] : graph.neighbors(u)) {
        const NodeID c_v = mapping[v];
        if (c_u != c_v) { local_collector[c_v] += graph.edge_weight(e); }
      }
    }

    c_node_weights[c_u] = c_u_weight;          // coarse node weights are done now
    c_nodes[c_u + 1] = local_collector.size(); // node degree (used to build c_nodes)

    // since we don't know the value of c_nodes[c_u] yet (so far, it only holds the nodes degree), we can't place the
    // edges of c_u in the c_edges and c_edge_weights arrays; hence, we store them in auxiliary arrays and note their
    // position in the auxiliary arrays
    auto &local_edge_buffer = shared_edge_buffer.local();
    auto &local_node_buffer = shared_node_buffer.local();
    const std::size_t position = local_edge_buffer.get_current_position();
    local_node_buffer.emplace_back(c_u, position, &local_edge_buffer);
    for (const auto [c_v, weight] : local_collector) { local_edge_buffer.push(c_v, weight); }
    local_collector.clear();
  });

  parallel::prefix_sum(c_nodes.begin(), c_nodes.end(), c_nodes.begin());
  STOP_TIMER();

  ASSERT(c_nodes[0] == 0) << V(c_nodes);
  const EdgeID c_m = c_nodes.back();

  //
  // Construct rest of the coarse graph: edges, edge weights
  //

  START_TIMER("Merge");
  scalable_vector<BufferNode> all_buffered_nodes(c_n);
  parallel::IntegralAtomicWrapper<std::size_t> global_pos = 0;

  tbb::parallel_invoke(
      [&] {
        tbb::parallel_for(shared_edge_buffer.range(), [&](auto &r) {
          for (auto &buffer : r) {
            if (!buffer.current_chunk.empty()) { buffer.flush(); }
          }
        });
      },
      [&] {
        tbb::parallel_for(shared_node_buffer.range(), [&](const auto &r) {
          for (const auto &buffer : r) {
            const std::size_t local_pos = global_pos.fetch_add(buffer.size());
            std::copy(buffer.begin(), buffer.end(), all_buffered_nodes.begin() + local_pos);
          }
        });
      });
  STOP_TIMER();

  START_TIMER(TIMER_ALLOCATION);
  StaticArray<NodeID> c_edges{c_m};
  StaticArray<EdgeWeight> c_edge_weights{c_m};
  STOP_TIMER();

  // build coarse graph
  START_TIMER("Graph construction 2");
  tbb::parallel_for(static_cast<NodeID>(0), c_n, [&](const NodeID i) {
    const auto &buffered_node = all_buffered_nodes[i];
    const auto *chunks = buffered_node.chunks;
    const NodeID c_u = buffered_node.c_u;

    const Degree c_u_degree = c_nodes[c_u + 1] - c_nodes[c_u];
    const EdgeID first_target_index = c_nodes[c_u];
    const EdgeID first_source_index = buffered_node.position;

    for (std::size_t j = 0; j < c_u_degree; ++j) {
      const auto to = first_target_index + j;
      const auto [c_v, weight] = chunks->get(first_source_index + j);
      c_edges[to] = c_v;
      c_edge_weights[to] = weight;
    }
  });
  STOP_TIMER();

  return {Graph{std::move(c_nodes), std::move(c_edges), std::move(c_node_weights), std::move(c_edge_weights)},
          std::move(mapping), std::move(m_ctx)};
}

/*
 * Builds a node permutation perm[x] such that the following condition is satisfied:
 * let
 * - n0 be the number of nodes with degree zero
 * - and ni be the number of nodes with degree in 2^(i - 1)..(2^i)-1
 * then
 * - perm[0..n0-1] contains all nodes with degree zero
 * - and perm[ni..n(i + 1)-1] contains all nodes with degree 2^(i - 1)..(2^i)-1
 */
NodePermutations sort_by_degree_buckets(const StaticArray<EdgeID> &nodes, const bool deg0_position) {
  auto find_bucket = [&](const Degree deg) {
    return (deg0_position && deg == 0) ? kNumberOfDegreeBuckets - 1 : degree_bucket(deg);
  };

  const NodeID n = nodes.size() - 1;
  const int p = std::min<int>(tbb::this_task_arena::max_concurrency(), n);

  START_TIMER(TIMER_ALLOCATION);
  StaticArray<NodeID> permutation{n};
  StaticArray<NodeID> inverse_permutation{n};
  STOP_TIMER();

  using Buckets = std::array<NodeID, kNumberOfDegreeBuckets + 1>;
  std::vector<Buckets, tbb::cache_aligned_allocator<Buckets>> local_buckets(p + 1);

  tbb::parallel_for(static_cast<int>(0), p, [&](const int id) {
    if (id >= p) { return; }

    auto &my_buckets = local_buckets[id + 1];
    const NodeID chunk = n / p;
    const NodeID rem = n % p;
    const NodeID from = id * chunk + std::min(id, static_cast<int>(rem));
    const NodeID to = from + ((id < static_cast<int>(rem)) ? chunk + 1 : chunk);

    for (NodeID u = from; u < to; ++u) {
      const Degree bucket = find_bucket(nodes[u + 1] - nodes[u]);
      permutation[u] = my_buckets[bucket]++;
    }
  });

  // Build a table of prefix numbers to correct the position of each node in the final permutation
  // After the previous loop, permutation[u] contains the position of u in the thread-local bucket.
  // (i) account for smaller buckets --> add prefix computed in global_buckets
  // (ii) account for the same bucket in smaller processor IDs --> add prefix computed in local_buckets
  Buckets global_buckets{};
  for (int id = 1; id < p + 1; ++id) {
    for (std::size_t i = 0; i + 1 < global_buckets.size(); ++i) { global_buckets[i + 1] += local_buckets[id][i]; }
  }
  parallel::prefix_sum(global_buckets.begin(), global_buckets.end(), global_buckets.begin());
  for (std::size_t i = 0; i < global_buckets.size(); ++i) {
    for (int id = 0; id + 1 < p; ++id) { local_buckets[id + 1][i] += local_buckets[id][i]; }
  }

  START_TIMER("Build permutation");
  tbb::parallel_for(static_cast<int>(0), p, [&](const int id) {
    if (id >= p) { return; }
    auto &my_buckets = local_buckets[id];
    const NodeID chunk = n / p;
    const NodeID rem = n % p;
    const NodeID from = id * chunk + std::min(id, static_cast<int>(rem));
    const NodeID to = from + ((id < static_cast<int>(rem)) ? chunk + 1 : chunk);
    for (NodeID u = from; u < to; ++u) {
      const Degree bucket = find_bucket(nodes[u + 1] - nodes[u]);
      permutation[u] += global_buckets[bucket] + my_buckets[bucket];
    }
  });
  STOP_TIMER();

  START_TIMER("Part 2");
  tbb::parallel_for(static_cast<std::size_t>(1), nodes.size(), [&](const NodeID u_plus_one) {
    const NodeID u = u_plus_one - 1;
    inverse_permutation[permutation[u]] = u;
  });
  STOP_TIMER();

  return {std::move(permutation), std::move(inverse_permutation)};
}

/*
 * Applies a node permutation `permutation` to a graph given as adjacency array.
 */
void build_permuted_graph(const StaticArray<EdgeID> &old_nodes, const StaticArray<NodeID> &old_edges,
                          const StaticArray<NodeWeight> &old_node_weights,
                          const StaticArray<EdgeWeight> &old_edge_weights, const NodePermutations &permutations,
                          StaticArray<EdgeID> &new_nodes, StaticArray<NodeID> &new_edges,
                          StaticArray<NodeWeight> &new_node_weights, StaticArray<EdgeWeight> &new_edge_weights) {
  ASSERT((old_node_weights.empty() && old_edge_weights.empty()) ||
         (old_node_weights.size() + 1 == old_nodes.size() && old_edge_weights.size() == old_edges.size()));
  const bool is_weighted = old_node_weights.size() + 1 == old_nodes.size();

  const NodeID n = old_nodes.size() - 1;
  ASSERT(n + 1 == new_nodes.size());

  // Build p_nodes, p_node_weights
  tbb::parallel_for(static_cast<NodeID>(0), n, [&](const NodeID u) {
    const NodeID old_u = permutations.new_to_old[u];

    new_nodes[u] = old_nodes[old_u + 1] - old_nodes[old_u];
    if (is_weighted) { new_node_weights[u] = old_node_weights[old_u]; }
  });
  parallel::prefix_sum(new_nodes.begin(), new_nodes.end(), new_nodes.begin());

  // Build p_edges, p_edge_weights
  tbb::parallel_for(static_cast<NodeID>(0), n, [&](const NodeID u) {
    const NodeID old_u = permutations.new_to_old[u];

    for (EdgeID e = old_nodes[old_u]; e < old_nodes[old_u + 1]; ++e) {
      const NodeID v = old_edges[e];
      const EdgeID p_e = --new_nodes[u];
      new_edges[p_e] = permutations.old_to_new[v];
      if (is_weighted) { new_edge_weights[p_e] = old_edge_weights[e]; }
    }
  });
}

std::pair<NodeID, NodeWeight> find_isolated_nodes_info(const StaticArray<EdgeID> &nodes,
                                                       const StaticArray<NodeWeight> &node_weights) {
  ASSERT(node_weights.empty() || node_weights.size() + 1 == nodes.size());

  tbb::enumerable_thread_specific<NodeID> isolated_nodes;
  tbb::enumerable_thread_specific<NodeWeight> isolated_nodes_weights;
  const bool is_weighted = !node_weights.empty();

  const NodeID n = nodes.size() - 1;
  tbb::parallel_for(tbb::blocked_range<NodeID>(0, n), [&](const tbb::blocked_range<NodeID> &r) {
    NodeID &local_isolated_nodes = isolated_nodes.local();
    NodeWeight &local_isolated_weights = isolated_nodes_weights.local();

    for (NodeID u = r.begin(); u != r.end(); ++u) {
      if (nodes[u] == nodes[u + 1]) {
        ++local_isolated_nodes;
        local_isolated_weights += is_weighted ? node_weights[u] : 1;
      }
    }
  });

  return {isolated_nodes.combine(std::plus{}), isolated_nodes_weights.combine(std::plus{})};
}

NodePermutations rearrange_and_remove_isolated_nodes(const bool remove_isolated_nodes, PartitionContext &p_ctx,
                                                     StaticArray<EdgeID> &nodes, StaticArray<NodeID> &edges,
                                                     StaticArray<NodeWeight> &node_weights,
                                                     StaticArray<EdgeWeight> &edge_weights,
                                                     NodeWeight total_node_weight) {
  START_TIMER(TIMER_ALLOCATION);
  StaticArray<EdgeID> tmp_nodes(nodes.size());
  StaticArray<NodeID> tmp_edges(edges.size());
  StaticArray<NodeWeight> tmp_node_weights(node_weights.size());
  StaticArray<EdgeWeight> tmp_edge_weights(edge_weights.size());
  STOP_TIMER();

  // if we are about to remove all isolated nodes, we place them to the end of the graph data structure
  // this way, we can just cut them off without doing further work
  START_TIMER("Rearrange input graph");
  NodePermutations permutations = sort_by_degree_buckets(nodes, remove_isolated_nodes);
  build_permuted_graph(nodes, edges, node_weights, edge_weights, permutations, tmp_nodes, tmp_edges, tmp_node_weights,
                       tmp_edge_weights);
  std::swap(nodes, tmp_nodes);
  std::swap(edges, tmp_edges);
  std::swap(node_weights, tmp_node_weights);
  std::swap(edge_weights, tmp_edge_weights);
  STOP_TIMER();

  if (remove_isolated_nodes) {
    if (total_node_weight == -1) {
      if (node_weights.size() == 0) {
        total_node_weight = nodes.size() - 1;
      } else {
        total_node_weight = parallel::accumulate(node_weights);
      }
    }

    const auto [isolated_nodes, isolated_nodes_weight] = find_isolated_nodes_info(nodes, node_weights);

    const NodeID old_n = nodes.size() - 1;
    const NodeID new_n = old_n - isolated_nodes;
    const NodeWeight new_weight = total_node_weight - isolated_nodes_weight;

    const BlockID k = p_ctx.k;
    const double old_epsilon = p_ctx.epsilon;
    const double old_max_block_weight = (1 + p_ctx.epsilon) * std::ceil(1.0 * total_node_weight / k);
    const double new_epsilon = old_max_block_weight / std::ceil(1.0 * new_weight / k) - 1;
    p_ctx.epsilon = new_epsilon;

    LOG << "Removed " << old_n - new_n << " isolated nodes with total weight " << isolated_nodes_weight;
    LOG << " -> adapted epsilon=" << old_epsilon << " to epsilon'=" << new_epsilon;

    nodes.restrict(new_n + 1);
    if (!node_weights.empty()) { node_weights.restrict(new_n); }
  }

  return permutations;
}

PartitionedGraph revert_isolated_nodes_removal(PartitionedGraph p_graph, const NodeID num_isolated_nodes,
                                               const PartitionContext &p_ctx) {
  const Graph &graph = p_graph.graph();
  const NodeID num_nonisolated_nodes = graph.n() - num_isolated_nodes;

  StaticArray<BlockID> partition(graph.n()); // n() should include isolated nodes now
  // copy partition of non-isolated nodes
  tbb::parallel_for(static_cast<NodeID>(0), static_cast<NodeID>(num_nonisolated_nodes),
                    [&](const NodeID u) { partition[u] = p_graph.block(u); });

  // now append the isolated ones
  const BlockID k = p_graph.k();
  auto block_weights = p_graph.take_block_weights();
  BlockID b = 0;

  // TODO parallelize this
  for (NodeID u = num_nonisolated_nodes; u < num_nonisolated_nodes + num_isolated_nodes; ++u) {
    while (b + 1 < k && block_weights[b] + graph.node_weight(u) > p_ctx.max_block_weight(b)) { ++b; }
    partition[u] = b;
    block_weights[b] += graph.node_weight(u);
  }

  return {graph, k, std::move(partition)};
}

std::pair<NodeID, NodeID> find_furthest_away_node(const Graph &graph, const NodeID start_node, Queue<NodeID> &queue,
                                                  Marker<> &marker) {
  queue.push_tail(start_node);
  marker.set<true>(start_node);

  NodeID current_distance = 0;
  NodeID last_node = start_node;
  NodeID remaining_nodes_in_level = 1;
  NodeID nodes_in_next_level = 0;

  while (!queue.empty()) {
    const NodeID u = queue.head();
    queue.pop_head();
    last_node = u;

    for (const NodeID v : graph.adjacent_nodes(u)) {
      if (marker.get(v)) continue;
      queue.push_tail(v);
      marker.set<true>(v);
      ++nodes_in_next_level;
    }

    // keep track of distance from start_node
    ASSERT(remaining_nodes_in_level > 0);
    --remaining_nodes_in_level;
    if (remaining_nodes_in_level == 0) {
      ++current_distance;
      remaining_nodes_in_level = nodes_in_next_level;
      nodes_in_next_level = 0;
    }
  }
  ASSERT(current_distance > 0);
  --current_distance;

  // bfs did not scan the whole graph, i.e., we have disconnected components
  if (marker.first_unmarked_element() < graph.n()) {
    last_node = marker.first_unmarked_element();
    current_distance = std::numeric_limits<NodeID>::max(); // infinity
  }

  marker.reset();
  queue.clear();
  return {last_node, current_distance};
}
} // namespace kaminpar
