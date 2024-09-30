/*******************************************************************************
 * Contraction implementation that uses an edge buffer to store edges before
 * building the final graph.
 *
 * @file:   buffered_cluster_contraction.cc
 * @author: Daniel Seemaier
 * @author: Daniel Salwasser
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/coarsening/contraction/buffered_cluster_contraction.h"

#include <memory>

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/coarsening/contraction/cluster_contraction_preprocessing.h"

#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::contraction {
namespace {
template <typename Graph>
std::unique_ptr<CoarseGraph> contract_clustering_buffered(
    const Graph &graph,
    const NodeID c_n,
    StaticArray<NodeID> mapping,
    const ContractionCoarseningContext &con_ctx,
    MemoryContext &m_ctx
) {
  auto &buckets_index = m_ctx.buckets_index;
  auto &buckets = m_ctx.buckets;
  auto &all_buffered_nodes = m_ctx.all_buffered_nodes;

  START_TIMER("Allocation");
  START_HEAP_PROFILER("Coarse graph node allocation");
  RECORD("c_nodes") StaticArray<EdgeID> c_nodes(c_n + 1);
  RECORD("c_node_weights") StaticArray<NodeWeight> c_node_weights(c_n);
  STOP_HEAP_PROFILER();
  STOP_TIMER();

  // We build the coarse graph in multiple iterations. In each iteration we compute for a part of
  // the coarse nodes the degree and node weight of each node. We can't insert the edges and edge
  // weights into the arrays yet, because positioning edges in those arrays depends on c_nodes,
  // which we only have after computing a prefix sum over the coarse node degrees in the current
  // part. Thus, we store the edges and edge weights in a temporary buffer and compute the prefix
  // sum and insert the edges and edge weights into the arrays after processing the part.

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
    }

    // Create a chunk for the last coarse nodes, if the last coarse node did not cross the limit.
    if (chunk_start != c_n) {
      cluster_chunks.emplace_back(chunk_start, c_n);
    }
  }

  STOP_TIMER();

  // Overcomit memory for the edge and edge weight array as we only know the amount of edges of
  // the coarse graph afterwards.
  auto c_edges = heap_profiler::overcommit_memory<NodeID>(edge_count);
  auto c_edge_weights = heap_profiler::overcommit_memory<EdgeWeight>(edge_count);

  START_HEAP_PROFILER("Construct coarse graph");
  START_TIMER("Construct coarse graph");

  tbb::enumerable_thread_specific<RatingMap<EdgeWeight, NodeID>> collector{[&] {
    return RatingMap<EdgeWeight, NodeID>(c_n);
  }};
  NavigableLinkedList<NodeID, Edge, ScalableVector> edge_buffer_ets;

  for (const auto &[cluster_start, cluster_end] : cluster_chunks) {
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
            graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
              const NodeID c_v = mapping[v];
              if (c_u != c_v) {
                map[c_v] += w;
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

    tbb::parallel_invoke(
        [&] {
          tbb::parallel_for(edge_buffer_ets.range(), [&](auto &r) {
            for (auto &local_list : r) {
              local_list.flush();
            }
          });
        },
        [&] {
          tbb::parallel_for(edge_buffer_ets.range(), [&](const auto &r) {
            for (const auto &local_list : r) {
              const auto &markers = local_list.markers();
              const std::size_t local_pos = global_pos.fetch_add(markers.size());
              std::copy(markers.begin(), markers.end(), all_buffered_nodes.begin() + local_pos);
            }
          });
        }
    );

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
        c_edges.get()[to] = c_v;
        c_edge_weights.get()[to] = weight;
      }
    });

    edge_buffer_ets.clear();
    all_buffered_nodes.free();
  }

  STOP_TIMER();
  STOP_HEAP_PROFILER();

  KASSERT(c_nodes[0] == 0u);
  const EdgeID c_m = c_nodes.back();

  START_HEAP_PROFILER("Coarse graph edges allocation");
  if constexpr (kHeapProfiling) {
    heap_profiler::HeapProfiler::global().record_alloc(c_edges.get(), c_m * sizeof(NodeID));
    heap_profiler::HeapProfiler::global().record_alloc(
        c_edge_weights.get(), c_m * sizeof(EdgeWeight)
    );
  }

  RECORD("c_edges") StaticArray<NodeID> finalized_c_edges(c_m, std::move(c_edges));
  RECORD("c_edge_weights")
  StaticArray<EdgeWeight> finalized_c_edge_weights(c_m, std::move(c_edge_weights));
  STOP_HEAP_PROFILER();

  return std::make_unique<CoarseGraphImpl>(
      shm::Graph(std::make_unique<CSRGraph>(
          std::move(c_nodes),
          std::move(finalized_c_edges),
          std::move(c_node_weights),
          std::move(finalized_c_edge_weights)
      )),
      std::move(mapping)
  );
}
} // namespace

std::unique_ptr<CoarseGraph> contract_clustering_buffered(
    const Graph &graph,
    StaticArray<NodeID> clustering,
    const ContractionCoarseningContext &con_ctx,
    MemoryContext &m_ctx
) {
  auto [c_n, mapping] = compute_mapping(graph, std::move(clustering), m_ctx);
  fill_cluster_buckets(c_n, graph, mapping, m_ctx.buckets_index, m_ctx.buckets);
  return graph.reified([&](auto &graph) {
    return contract_clustering_buffered(graph, c_n, std::move(mapping), con_ctx, m_ctx);
  });
}
} // namespace kaminpar::shm::contraction
