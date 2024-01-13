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

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/graph.h"

#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/datastructures/ts_navigable_linked_list.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::graph {
using namespace contraction;

namespace {
template <typename Graph, typename Clustering>
Result contract_generic_clustering(
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
  RECORD_DATA_STRUCT("scalable_vector", mapping.capacity() * sizeof(NodeID));

  if (leader_mapping.size() < graph.n()) {
    leader_mapping.resize(graph.n());
  }
  RECORD("leader_mapping");
  RECORD_DATA_STRUCT(
      "scalable_vector", leader_mapping.capacity() * sizeof(parallel::Atomic<NodeID>)
  );

  if (buckets.size() < graph.n()) {
    buckets.resize(graph.n());
  }
  RECORD("buckets");
  RECORD_DATA_STRUCT("scalable_vector", buckets.capacity() * sizeof(NodeID));

  STOP_TIMER();
  STOP_HEAP_PROFILER();

  START_TIMER("Preprocessing");

  //
  // Compute a mapping from the nodes of the current graph to the nodes of the
  // coarse graph I.e., node_mapping[node u] = coarse node c_u
  //
  // Note that clustering satisfies this invariant (I): if clustering[x] = y for
  // some node x, then clustering[y] = y
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
  RECORD_DATA_STRUCT(
      "scalable_vector", buckets_index.capacity() * sizeof(parallel::Atomic<NodeID>)
  );
  STOP_HEAP_PROFILER();

  START_TIMER("Preprocessing");

  //
  // Sort nodes into buckets: place all nodes belonging to coarse node i into
  // the i-th bucket
  //
  // Count the number of nodes in each bucket, then compute the position of the
  // bucket in the global buckets array using a prefix sum, roughly 2/5-th of
  // time on europe.osm with 2/3-th to 1/3-tel for loop to prefix sum
  graph.pfor_nodes([&](const NodeID u) {
    buckets_index[mapping[u]].fetch_add(1, std::memory_order_relaxed);
  });

  parallel::prefix_sum(buckets_index.begin(), buckets_index.end(), buckets_index.begin());
  KASSERT(buckets_index.back() <= graph.n());

  // Sort nodes into   buckets, roughly 3/5-th of time on europe.osm
  tbb::parallel_for(static_cast<NodeID>(0), graph.n(), [&](const NodeID u) {
    const std::size_t pos = buckets_index[mapping[u]].fetch_sub(1, std::memory_order_relaxed) - 1;
    buckets[pos] = u;
  });

  STOP_TIMER(); // Preprocessing

  //
  // Build nodes array of the coarse graph
  // - firstly, we count the degree of each coarse node
  // - secondly, we obtain the nodes array using a prefix sum
  //
  START_HEAP_PROFILER("Coarse graph nodes allocation");
  START_TIMER("Allocation");
  RECORD("c_nodes") StaticArray<EdgeID> c_nodes{c_n + 1};
  RECORD("c_node_weights") StaticArray<NodeWeight> c_node_weights{c_n};
  STOP_TIMER();
  STOP_HEAP_PROFILER();

  //
  // We build the coarse graph in multiple steps:
  // (1) During the first step, we compute
  //     - the node weight of each coarse node
  //     - the degree of each coarse node
  //     We can't build c_edges and c_edge_weights yet, because positioning
  //     edges in those arrays depends on c_nodes, which we only have after
  //     computing a prefix sum over all coarse node degrees Hence, we store
  //     edges and edge weights in unsorted auxiliary arrays during the first
  //     pass
  // (2) We finalize c_nodes arrays by computing a prefix sum over all coarse
  // node degrees (3) We copy coarse edges and coarse edge weights from the
  // auxiliary arrays to c_edges and c_edge_weights
  //

  START_HEAP_PROFILER("Construct coarse edges");
  START_TIMER("Construct coarse edges");

  tbb::enumerable_thread_specific<RatingMap<EdgeWeight, NodeID>> collector{[&] {
    return RatingMap<EdgeWeight, NodeID>(c_n);
  }};

  // A temporary buffer to store the edges in the first step to avoid computing them in the second
  // step again. It is only used when the temporary buffer option is enabled in the contraction
  // context.
  NavigableLinkedList<NodeID, Edge, scalable_vector> edge_buffer_ets;

  const auto collect_contracted_nodes = [&](auto &&compute_upper_bound,
                                            auto &&collect_contracted_node) {
    tbb::parallel_for(tbb::blocked_range<NodeID>(0, c_n), [&](const auto &r) {
      auto &local_collector = collector.local();

      for (NodeID c_u = r.begin(); c_u != r.end(); ++c_u) {
        const std::size_t first = buckets_index[c_u];
        const std::size_t last = buckets_index[c_u + 1];

        // build coarse graph
        auto collect_edges = [&](auto &map) {
          NodeWeight c_u_weight = 0;
          for (std::size_t i = first; i < last; ++i) {
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

          collect_contracted_node(c_u, c_u_weight, map);
          map.clear();
        };

        local_collector.execute(compute_upper_bound(c_u, first, last), collect_edges);
      }
    });
  };

  // To select the right map, we compute a upper bound on the coarse node degree by summing the
  // degree of all fine nodes.
  const auto computer_upper_bound_degree =
      [&](const NodeID c_u, const std::size_t first, const std::size_t last) {
        NodeID upper_bound_degree = 0;

        for (std::size_t i = first; i < last; ++i) {
          const NodeID u = buckets[i];
          upper_bound_degree += graph.degree(u);
        }

        return upper_bound_degree;
      };

  tbb::enumerable_thread_specific<std::size_t> max_edge_weight_ets;

  if (con_ctx.use_edge_buffer) {
    collect_contracted_nodes(
        std::forward<decltype(computer_upper_bound_degree)>(computer_upper_bound_degree),
        [&](const NodeID c_u, const NodeWeight c_u_weight, auto &map) {
          auto &local_edge_buffer = edge_buffer_ets.local();
          local_edge_buffer.mark(c_u);

          c_nodes[c_u + 1] = map.size(); // Store node degree which is used to build c_nodes
          c_node_weights[c_u] = c_u_weight;

          // Since we don't know the value of c_nodes[c_u] yet (so far, it only holds the nodes
          // degree), we can't place the edges of c_u in the c_edges and c_edge_weights arrays;
          // hence, we store them in auxiliary arrays and note their position in the auxiliary
          // arrays.
          std::size_t max_edge_weight = max_edge_weight_ets.local();
          for (const auto [c_v, weight] : map.entries()) {
            local_edge_buffer.push_back({c_v, weight});
            max_edge_weight = std::max(max_edge_weight, math::abs(weight));
          }

          max_edge_weight_ets.local() = max_edge_weight;
        }
    );
  } else {
    collect_contracted_nodes(
        std::forward<decltype(computer_upper_bound_degree)>(computer_upper_bound_degree),
        [&](const NodeID c_u, const NodeWeight c_u_weight, auto &map) {
          c_nodes[c_u + 1] = map.size(); // Store node degree which is used to build c_nodes
          c_node_weights[c_u] = c_u_weight;

          std::size_t max_edge_weight = max_edge_weight_ets.local();
          for (const auto [c_v, weight] : map.entries()) {
            max_edge_weight = std::max(max_edge_weight, math::abs(weight));
          }

          max_edge_weight_ets.local() = max_edge_weight;
        }
    );
  }

  parallel::prefix_sum(c_nodes.begin(), c_nodes.end(), c_nodes.begin());

  std::size_t max_edge_weight = 0;
  for (const std::size_t edge_weight : max_edge_weight_ets) {
    max_edge_weight = std::max(max_edge_weight, edge_weight);
  }

  STOP_TIMER(); // Graph construction
  STOP_HEAP_PROFILER();

  //
  // Construct rest of the coarse graph: edges, edge weights
  //

  const auto build_coarse_graph = [&](auto &c_edges, auto &c_edge_weights) {
    SCOPED_HEAP_PROFILER("Construct coarse graph");
    SCOPED_TIMER("Construct coarse graph");

    if (con_ctx.use_edge_buffer) {
      all_buffered_nodes = ts_navigable_list::combine<NodeID, Edge, scalable_vector>(
          edge_buffer_ets, std::move(all_buffered_nodes)
      );

      tbb::parallel_for(static_cast<NodeID>(0), c_n, [&](const NodeID i) {
        const auto &marker = all_buffered_nodes[i];
        const auto *list = marker.local_list;
        const NodeID c_u = marker.key;

        const NodeID c_u_degree = c_nodes[c_u + 1] - c_nodes[c_u];
        const EdgeID first_target_index = c_nodes[c_u];
        const EdgeID first_source_index = marker.position;

        for (std::size_t j = 0; j < c_u_degree; ++j) {
          const auto to = first_target_index + j;
          const auto [c_v, weight] = list->get(first_source_index + j);

          if constexpr (std::is_same_v<decltype(c_edges), StaticArray<NodeID> &>) {
            c_edges[to] = c_v;
            c_edge_weights[to] = weight;
          } else {
            c_edges.write(to, c_v);
            c_edge_weights.write(to, weight);
          }
        }
      });
    } else {
      collect_contracted_nodes(
          // Since we computed the contracted node degrees in the first step we can use them as an
          // exact upper bound for the rating map.
          [&](const NodeID c_u, const std::size_t first, const std::size_t last) {
            return c_nodes[c_u + 1] - c_nodes[c_u];
          },
          [&](const NodeID c_u, const NodeWeight c_u_weight, auto &map) {
            EdgeID edge = c_nodes[c_u];

            for (const auto [c_v, weight] : map.entries()) {
              if constexpr (std::is_same_v<decltype(c_edges), StaticArray<NodeID> &>) {
                c_edges[edge] = c_v;
                c_edge_weights[edge] = weight;
              } else {
                c_edges.write(edge, c_v);
                c_edge_weights.write(edge, weight);
              }

              ++edge;
            }
          }
      );
    }
  };

  KASSERT(c_nodes[0] == 0u);
  const EdgeID c_m = c_nodes.back();

  if (con_ctx.use_compact_ids) {
    START_HEAP_PROFILER("Coarse graph edges allocation");
    START_TIMER("Allocation");
    RECORD("c_edges") CompactStaticArray<NodeID> c_edges(math::byte_width(c_n), c_m);
    RECORD("c_edge_weights")
    CompactStaticArray<EdgeWeight> c_edge_weights(math::byte_width(max_edge_weight), c_m);
    STOP_TIMER();
    STOP_HEAP_PROFILER();

    build_coarse_graph(c_edges, c_edge_weights);

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
    START_HEAP_PROFILER("Coarse graph edges allocation");
    START_TIMER("Allocation");
    RECORD("c_edges") StaticArray<NodeID> c_edges{c_m};
    RECORD("c_edge_weights") StaticArray<EdgeWeight> c_edge_weights{c_m};
    STOP_TIMER();
    STOP_HEAP_PROFILER();

    build_coarse_graph(c_edges, c_edge_weights);

    return {
        shm::Graph(std::make_unique<CSRGraph>(
            std::move(c_nodes),
            std::move(c_edges),
            std::move(c_node_weights),
            std::move(c_edge_weights)
        )),
        std::move(mapping),
        std::move(m_ctx)
    };
  }
}
} // namespace

Result contract(
    const Graph &graph,
    const ContractionCoarseningContext con_ctx,
    const scalable_vector<NodeID> &clustering,
    MemoryContext m_ctx
) {
  if (auto *csr_graph = dynamic_cast<CSRGraph *>(graph.underlying_graph()); csr_graph != nullptr) {
    return contract_generic_clustering(*csr_graph, con_ctx, clustering, std::move(m_ctx));
  }

  if (auto *compact_csr_graph = dynamic_cast<CompactCSRGraph *>(graph.underlying_graph());
      compact_csr_graph != nullptr) {
    return contract_generic_clustering(*compact_csr_graph, con_ctx, clustering, std::move(m_ctx));
  }

  if (auto *compressed_graph = dynamic_cast<CompressedGraph *>(graph.underlying_graph());
      compressed_graph != nullptr) {
    return contract_generic_clustering(*compressed_graph, con_ctx, clustering, std::move(m_ctx));
  }

  __builtin_unreachable();
}

Result contract(
    const Graph &graph,
    const ContractionCoarseningContext con_ctx,
    const scalable_vector<parallel::Atomic<NodeID>> &clustering,
    MemoryContext m_ctx
) {
  if (auto *csr_graph = dynamic_cast<CSRGraph *>(graph.underlying_graph()); csr_graph != nullptr) {
    return contract_generic_clustering(*csr_graph, con_ctx, clustering, std::move(m_ctx));
  }

  if (auto *compact_csr_graph = dynamic_cast<CompactCSRGraph *>(graph.underlying_graph());
      compact_csr_graph != nullptr) {
    return contract_generic_clustering(*compact_csr_graph, con_ctx, clustering, std::move(m_ctx));
  }

  if (auto *compressed_graph = dynamic_cast<CompressedGraph *>(graph.underlying_graph());
      compressed_graph != nullptr) {
    return contract_generic_clustering(*compressed_graph, con_ctx, clustering, std::move(m_ctx));
  }

  __builtin_unreachable();
}

} // namespace kaminpar::shm::graph
