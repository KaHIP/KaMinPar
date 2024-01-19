/*******************************************************************************
 * Contracts clusterings and constructs the coarse graph.
 *
 * @file:   cluster_contraction.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/graphutils/cluster_contraction.h"

#include <kassert/kassert.hpp>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "kaminpar-shm/datastructures/graph.h"

#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/datastructures/ts_navigable_linked_list.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::graph {
using namespace contraction;

template <typename Graph, typename Clustering>
Result contract_generic_graph(
    const Graph &graph,
    const ContractionCoarseningContext con_ctx,
    const Clustering &clustering,
    MemoryContext m_ctx
) {
  auto &buckets_index = m_ctx.buckets_index;
  auto &buckets = m_ctx.buckets;
  auto &leader_mapping = m_ctx.leader_mapping;
  auto &all_buffered_nodes = m_ctx.all_buffered_nodes;

  START_HEAP_PROFILER("Mapping allocation");
  START_TIMER("Allocation");

  RECORD("mapping") scalable_vector<NodeID> mapping(graph.n());
  RECORD_LOCAL_DATA_STRUCT("scalable_vector<NodeID>", mapping.capacity() * sizeof(NodeID));

  if (leader_mapping.size() < graph.n()) {
    leader_mapping.resize(graph.n());
  }
  RECORD("leader_mapping");
  RECORD_LOCAL_DATA_STRUCT(
      "scalable_vector<parallel::Atomic<NodeID>>",
      leader_mapping.capacity() * sizeof(parallel::Atomic<NodeID>)
  );

  if (buckets.size() < graph.n()) {
    buckets.resize(graph.n());
  }
  RECORD("buckets");
  RECORD_LOCAL_DATA_STRUCT("scalable_vector<NodeID>", buckets.capacity() * sizeof(NodeID));

  STOP_TIMER();
  STOP_HEAP_PROFILER();

  START_TIMER("Preprocessing");

  //
  // Compute a mapping from the nodes of the current graph to the nodes of the coarse graph. i.e.
  // node_mapping[node u] = coarse node c_u
  //
  // Note that clustering satisfies this invariant (I): if clustering[x] = y for some node x, then
  // clustering[y] = y
  //

  // Set node_mapping[x] = 1 iff. there is a cluster with leader x
  graph.pfor_nodes([&](const NodeID u) { leader_mapping[u] = 0; });
  graph.pfor_nodes([&](const NodeID u) {
    leader_mapping[clustering[u]].store(1, std::memory_order_relaxed);
  });

  // Compute prefix sum to get coarse node IDs (starting at 1!)
  parallel::prefix_sum(
      leader_mapping.begin(), leader_mapping.begin() + graph.n(), leader_mapping.begin()
  );
  const NodeID c_n = leader_mapping[graph.n() - 1]; // number of nodes in the coarse graph

  // Assign coarse node ID to all nodes; this works due to (I)
  graph.pfor_nodes([&](const NodeID u) { mapping[u] = leader_mapping[clustering[u]]; });
  graph.pfor_nodes([&](const NodeID u) { --mapping[u]; });

  STOP_TIMER();

  START_HEAP_PROFILER("Buckets allocation");
  TIMED_SCOPE("Allocation") {
    buckets_index.clear();
    buckets_index.resize(c_n + 1);
  };
  RECORD("buckets_index");
  RECORD_LOCAL_DATA_STRUCT(
      "scalable_vector<parallel::Atomic<NodeID>>",
      buckets_index.capacity() * sizeof(parallel::Atomic<NodeID>)
  );
  STOP_HEAP_PROFILER();

  START_TIMER("Preprocessing");

  //
  // Sort nodes into buckets: place all nodes belonging to coarse node i into the i-th bucket
  //
  // Count the number of nodes in each bucket, then compute the position of the bucket in the global
  // buckets array using a prefix sum, roughly 2/5-th of time on europe.osm with 2/3-th to 1/3-tel
  // for loop to prefix sum
  //

  graph.pfor_nodes([&](const NodeID u) {
    buckets_index[mapping[u]].fetch_add(1, std::memory_order_relaxed);
  });

  parallel::prefix_sum(buckets_index.begin(), buckets_index.end(), buckets_index.begin());
  KASSERT(buckets_index.back() <= graph.n());

  // Sort nodes into buckets, roughly 3/5-th of time on europe.osm
  tbb::parallel_for(static_cast<NodeID>(0), graph.n(), [&](const NodeID u) {
    const std::size_t pos = buckets_index[mapping[u]].fetch_sub(1, std::memory_order_relaxed) - 1;
    buckets[pos] = u;
  });

  STOP_TIMER(); // Preprocessing

  START_HEAP_PROFILER("Coarse graph node allocation");
  RECORD("c_nodes") StaticArray<EdgeID> c_nodes{c_n + 1};
  RECORD("c_node_weights") StaticArray<NodeWeight> c_node_weights{c_n};
  STOP_HEAP_PROFILER();

  if (con_ctx.use_compact_ids) {
    //
    // We build the coarse graph in two steps. First, we compute the degree and node weight of each
    // coarse node. We store the degree in the node array and the node weights in the node weight
    // array. We compute a prefix sum over all coarse node degrees to have the correct offsets in
    // the node array. Additionally, we compute the max edge weight. Then, we allocate the edge and
    // edge weight array with compact IDs. In the second step, we compute the edges and edge weights
    // again and store them in the corresponding arrays.
    //

    START_HEAP_PROFILER("Construct coarse nodes");
    START_TIMER("Construct coarse nodes");

    tbb::enumerable_thread_specific<RatingMap<EdgeWeight, NodeID>> collector{[&] {
      return RatingMap<EdgeWeight, NodeID>(c_n);
    }};
    tbb::enumerable_thread_specific<std::size_t> max_edge_weight_ets;

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, c_n), [&](const auto &r) {
      auto &local_collector = collector.local();

      for (NodeID c_u = r.begin(); c_u != r.end(); ++c_u) {
        const NodeID first = buckets_index[c_u];
        const NodeID last = buckets_index[c_u + 1];

        // Build coarse graph
        const auto collect_edges = [&](auto &map) {
          NodeWeight c_u_weight = 0;
          for (NodeID i = first; i < last; ++i) {
            const NodeID u = buckets[i];
            KASSERT(mapping[u] == c_u);

            c_u_weight += graph.node_weight(u);

            // Collect coarse edges
            graph.neighbors(u, [&](const EdgeID e, const NodeID v) {
              const NodeID c_v = mapping[v];
              if (c_u != c_v) {
                map[c_v] += graph.edge_weight(e);
              }
            });
          }

          c_nodes[c_u + 1] = map.size(); // Store node degree which is used to build c_nodes
          c_node_weights[c_u] = c_u_weight;

          std::size_t max_edge_weight = max_edge_weight_ets.local();
          for (const auto [c_v, weight] : map.entries()) {
            max_edge_weight = std::max(max_edge_weight, math::abs(weight));
          }

          max_edge_weight_ets.local() = max_edge_weight;
          map.clear();
        };

        // To select the right map, we compute a upper bound on the coarse node degree by summing
        // the degree of all fine nodes.
        NodeID upper_bound_degree = 0;
        for (NodeID i = first; i < last; ++i) {
          const NodeID u = buckets[i];
          upper_bound_degree += graph.degree(u);
        }

        local_collector.execute(upper_bound_degree, collect_edges);
      }
    });

    parallel::prefix_sum(c_nodes.begin(), c_nodes.end(), c_nodes.begin());

    std::size_t max_edge_weight = 0;
    for (const std::size_t edge_weight : max_edge_weight_ets) {
      max_edge_weight = std::max(max_edge_weight, edge_weight);
    }

    STOP_TIMER();
    STOP_HEAP_PROFILER();

    KASSERT(c_nodes[0] == 0u);
    const EdgeID c_m = c_nodes.back();

    START_HEAP_PROFILER("Coarse graph edges allocation");
    START_TIMER("Allocation");
    RECORD("c_edges") CompactStaticArray<NodeID> c_edges(math::byte_width(c_n), c_m);
    RECORD("c_edge_weights")
    CompactStaticArray<EdgeWeight> c_edge_weights(math::byte_width(max_edge_weight), c_m);
    STOP_TIMER();
    STOP_HEAP_PROFILER();

    START_HEAP_PROFILER("Construct coarse edges");
    START_TIMER("Construct coarse edges");

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, c_n), [&](const auto &r) {
      auto &local_collector = collector.local();

      for (NodeID c_u = r.begin(); c_u != r.end(); ++c_u) {
        const NodeID first = buckets_index[c_u];
        const NodeID last = buckets_index[c_u + 1];

        // Build coarse graph
        const auto collect_edges = [&](auto &map) {
          for (NodeID i = first; i < last; ++i) {
            const NodeID u = buckets[i];
            KASSERT(mapping[u] == c_u);

            // Collect coarse edges
            graph.neighbors(u, [&](const EdgeID e, const NodeID v) {
              const NodeID c_v = mapping[v];
              if (c_u != c_v) {
                map[c_v] += graph.edge_weight(e);
              }
            });
          }

          EdgeID edge = c_nodes[c_u];
          for (const auto [c_v, weight] : map.entries()) {
            c_edges.write(edge, c_v);
            c_edge_weights.write(edge, weight);
            ++edge;
          }

          map.clear();
        };

        const NodeID degree = c_nodes[c_u + 1] - c_nodes[c_u];
        local_collector.execute(degree, collect_edges);
      }
    });

    STOP_TIMER();
    STOP_HEAP_PROFILER();

    return {
        shm::Graph(std::make_unique<CompactCSRGraph>(
            std::move(c_nodes),
            std::move(c_edges),
            std::move(c_node_weights),
            std::move(c_edge_weights)
        )),
        std::move(mapping),
        std::move(m_ctx)
    };
  } else {
    //
    // We build the coarse graph in multiple iterations. In each iteration we compute for a part of
    // the coarse nodes the degree and node weight of each node. We can't insert the edges and edge
    // weights into the arrays yet, because positioning edges in those arrays depends on c_nodes,
    // which we only have after computing a prefix sum over the coarse node degrees in the current
    // part. Thus, we store the edges and edge weights in a temporary buffer and compute the prefix
    // sum and insert the edges and edge weights into the arrays after processing the part.
    //

    const EdgeID edge_count = graph.m();

    // Split the coarse nodes into chunks, such that the edges of the fine graph are roughly split
    // among the chunks.
    START_TIMER("Compute coarse node chunks");
    std::vector<std::pair<NodeID, NodeID>> cluster_chunks;

    // If there are too few fine edges then don't split the coarse nodes into chunks, as this
    // provides little memory benefits.
    const bool split_coarse_nodes = edge_count >= 100000 && con_ctx.edge_buffer_fill_fraction < 1;
    if (!split_coarse_nodes) {
      cluster_chunks.emplace_back(0, c_n);
    } else {
      // Compute the fine degrees of the coarse nodes in parallel and store them temporarily in the
      // (unused) c_nodes array.
      tbb::parallel_for<NodeID>(0, c_n, [&](const NodeID c_u) {
        const NodeID first = buckets_index[c_u];
        const NodeID last = buckets_index[c_u + 1];

        NodeID fine_degree = 0;
        for (NodeID u = first; u < last; ++u) {
          const NodeID v = buckets[u];
          fine_degree += graph.degree(v);
        }

        c_nodes[c_u + 1] = fine_degree;
      });

      const EdgeID max_chunk_edge_count = edge_count * con_ctx.edge_buffer_fill_fraction;

      NodeID chunk_start = 0;
      NodeID chunk_edge_count = 0;
      for (NodeID c_u = 0; c_u < c_n; ++c_u) {
        const NodeID fine_degree = c_nodes[c_u + 1];
        chunk_edge_count += fine_degree;

        // With this coarse node the chunk would have more fine edges than the maximum allowed
        // limit. Thus, create a new chunk.
        if (chunk_edge_count >= max_chunk_edge_count) {
          // It might happen that the chunk only consists of a high degree node which crosses the
          // limit.
          if (chunk_start == c_u) {
            cluster_chunks.emplace_back(c_u, c_u + 1);
            chunk_start = c_u + 1;
          } else {
            cluster_chunks.emplace_back(chunk_start, c_u);
            chunk_start = c_u;
          }

          chunk_edge_count = 0;
        }
        // Create a chunk for the last coarse nodes, if the total count with the last coarse node
        // together does not cross the limit,
        else if (c_u + 1 == c_n) {
          cluster_chunks.emplace_back(chunk_start, c_n);
        }
      }
    }

    STOP_TIMER();

    // Overcomit memory for the edge and edge weight array as we only know the amount of edges of
    // the coarse graph afterwards.
    NodeID *c_edges;
    EdgeWeight *c_edge_weights;
    if constexpr (kHeapProfiling) {
      // As we overcommit memory do not track the amount of bytes used directly. Instead record it
      // manually afterwards.
      c_edges = (NodeID *)heap_profiler::std_malloc(edge_count * sizeof(NodeID));
      c_edge_weights = (EdgeWeight *)heap_profiler::std_malloc(edge_count * sizeof(EdgeWeight));
    } else {
      c_edges = (NodeID *)std::malloc(edge_count * sizeof(NodeID));
      c_edge_weights = (EdgeWeight *)std::malloc(edge_count * sizeof(EdgeWeight));
    }

    START_HEAP_PROFILER("Construct coarse graph");
    START_TIMER("Construct coarse graph");

    tbb::enumerable_thread_specific<RatingMap<EdgeWeight, NodeID>> collector{[&] {
      return RatingMap<EdgeWeight, NodeID>(c_n);
    }};
    NavigableLinkedList<NodeID, Edge, scalable_vector> edge_buffer_ets;

    for (const auto [cluster_start, cluster_end] : cluster_chunks) {
      tbb::parallel_for(tbb::blocked_range<NodeID>(cluster_start, cluster_end), [&](const auto &r) {
        auto &local_collector = collector.local();
        auto &local_edge_buffer = edge_buffer_ets.local();

        for (NodeID c_u = r.begin(); c_u != r.end(); ++c_u) {
          local_edge_buffer.mark(c_u);

          const NodeID first = buckets_index[c_u];
          const NodeID last = buckets_index[c_u + 1];

          // Build coarse graph
          const auto collect_edges = [&](auto &map) {
            NodeWeight c_u_weight = 0;
            for (NodeID i = first; i < last; ++i) {
              const NodeID u = buckets[i];
              KASSERT(mapping[u] == c_u);

              c_u_weight += graph.node_weight(u); // coarse node weight

              // collect coarse edges
              graph.neighbors(u, [&](const EdgeID e, const NodeID v) {
                const NodeID c_v = mapping[v];
                if (c_u != c_v) {
                  map[c_v] += graph.edge_weight(e);
                }
              });
            }

            c_node_weights[c_u] = c_u_weight; // coarse node weights are done now
            c_nodes[c_u + 1] = map.size();    // node degree (used to build c_nodes)

            // Since we don't know the value of c_nodes[c_u] yet (so far, it only holds the nodes
            // degree), we can't place the edges of c_u in the c_edges and c_edge_weights arrays.
            // Hence, we store them in auxiliary arrays and note their position in the auxiliary
            // arrays
            for (const auto [c_v, weight] : map.entries()) {
              local_edge_buffer.push_back({c_v, weight});
            }

            map.clear();
          };

          // To select the right map, we need a upper bound on the coarse node degree. If we
          // previously split the coarse nodes into chunks, we have already computed them and stored
          // them in the c_nodes array.
          NodeID upper_bound_degree;
          if (split_coarse_nodes) {
            upper_bound_degree = c_nodes[c_u + 1];
          } else {
            upper_bound_degree = 0;

            for (NodeID i = first; i < last; ++i) {
              const NodeID u = buckets[i];
              upper_bound_degree += graph.degree(u);
            }
          }

          local_collector.execute(upper_bound_degree, collect_edges);
        }
      });

      parallel::prefix_sum(
          c_nodes.begin() + cluster_start,
          c_nodes.begin() + cluster_end + 1,
          c_nodes.begin() + cluster_start
      );

      parallel::Atomic<std::size_t> global_pos = 0;
      std::size_t num_markers = 0;
      for (const auto &local_list : edge_buffer_ets) {
        num_markers += local_list.markers().size();
      }
      if (all_buffered_nodes.size() < num_markers) {
        all_buffered_nodes.resize(num_markers);
      }

      tbb::parallel_for(edge_buffer_ets.range(), [&](auto &r) {
        for (auto &local_list : r) {
          local_list.flush();
        }
      });

      tbb::parallel_for(edge_buffer_ets.range(), [&](const auto &r) {
        for (const auto &local_list : r) {
          const auto &markers = local_list.markers();
          const std::size_t local_pos = global_pos.fetch_add(markers.size());
          std::copy(markers.begin(), markers.end(), all_buffered_nodes.begin() + local_pos);
        }
      });

      tbb::parallel_for<NodeID>(cluster_start, cluster_end, [&](const NodeID i) {
        const auto &marker = all_buffered_nodes[i - cluster_start];
        const auto *list = marker.local_list;
        const NodeID c_u = marker.key;

        const NodeID c_u_degree = c_nodes[c_u + 1] - c_nodes[c_u];
        const EdgeID first_target_index = c_nodes[c_u];
        const EdgeID first_source_index = marker.position;

        for (EdgeID j = 0; j < c_u_degree; ++j) {
          const auto to = first_target_index + j;
          const auto [c_v, weight] = list->get(first_source_index + j);
          c_edges[to] = c_v;
          c_edge_weights[to] = weight;
        }
      });

      edge_buffer_ets.clear();
      all_buffered_nodes.clear();
    }

    STOP_TIMER();
    STOP_HEAP_PROFILER();

    KASSERT(c_nodes[0] == 0u);
    const EdgeID c_m = c_nodes.back();

    START_HEAP_PROFILER("Coarse graph edges allocation");
    RECORD("c_edges") StaticArray<NodeID> finalized_c_edges(c_m, c_edges);
    RECORD("c_edge_weights") StaticArray<EdgeWeight> finalized_c_edge_weights(c_m, c_edge_weights);
    if constexpr (kHeapProfiling) {
      heap_profiler::HeapProfiler::global().record_alloc(c_edges, c_m * sizeof(NodeID));
      heap_profiler::HeapProfiler::global().record_alloc(c_edge_weights, c_m * sizeof(EdgeWeight));
    }
    STOP_HEAP_PROFILER();

    return {
        shm::Graph(std::make_unique<CSRGraph>(
            std::move(c_nodes),
            std::move(finalized_c_edges),
            std::move(c_node_weights),
            std::move(finalized_c_edge_weights)
        )),
        std::move(mapping),
        std::move(m_ctx)
    };
  }
}

template <typename Clustering>
Result contract(
    const Graph &graph,
    const ContractionCoarseningContext con_ctx,
    const Clustering &clustering,
    MemoryContext m_ctx
) {
  if (auto *csr_graph = dynamic_cast<CSRGraph *>(graph.underlying_graph()); csr_graph != nullptr) {
    return contract_generic_graph(*csr_graph, con_ctx, clustering, std::move(m_ctx));
  }

  if (auto *compact_csr_graph = dynamic_cast<CompactCSRGraph *>(graph.underlying_graph());
      compact_csr_graph != nullptr) {
    return contract_generic_graph(*compact_csr_graph, con_ctx, clustering, std::move(m_ctx));
  }

  if (auto *compressed_graph = dynamic_cast<CompressedGraph *>(graph.underlying_graph());
      compressed_graph != nullptr) {
    return contract_generic_graph(*compressed_graph, con_ctx, clustering, std::move(m_ctx));
  }

  __builtin_unreachable();
}

template Result contract<scalable_vector<NodeID>>(
    const Graph &graph,
    const ContractionCoarseningContext con_ctx,
    const scalable_vector<NodeID> &clustering,
    MemoryContext m_ctx
);

template Result contract<scalable_vector<parallel::Atomic<NodeID>>>(
    const Graph &graph,
    const ContractionCoarseningContext con_ctx,
    const scalable_vector<parallel::Atomic<NodeID>> &clustering,
    MemoryContext m_ctx
);

} // namespace kaminpar::shm::graph
