/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#include "dkaminpar/algorithm/distributed_local_graph_contraction.h"

#include "kaminpar/datastructure/rating_map.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

namespace dkaminpar::graph {
ContractionResult contract_locally(const DistributedGraph &graph, const scalable_vector<DNodeID> &clustering,
                                   ContractionMemoryContext m_ctx) {
  auto &buckets_index = m_ctx.buckets_index;
  auto &buckets = m_ctx.buckets;
  auto &leader_mapping = m_ctx.leader_mapping;
  auto &all_buffered_nodes = m_ctx.all_buffered_nodes;

  scalable_vector<DNodeID> mapping(graph.total_n());
  if (leader_mapping.size() < graph.n()) { leader_mapping.resize(graph.n()); }
  if (buckets.size() < graph.n()) { buckets.resize(graph.n()); }

  //
  // Compute a mapping from the nodes of the current graph to the nodes of the coarse graph
  // I.e., node_mapping[node u] = coarse node c_u
  //
  // Note that clustering satisfies this invariant (I): if clustering[x] = y for some node x, then clustering[y] = y
  //

  // Set node_mapping[x] = 1 iff. there is a cluster with leader x
  graph.pfor_nodes([&](const DNodeID u) { leader_mapping[u] = 0; });
  graph.pfor_nodes([&](const DNodeID u) { leader_mapping[clustering[u]].store(1, std::memory_order_relaxed); });

  // Compute prefix sum to get coarse node IDs (starting at 1!)
  shm::parallel::prefix_sum(leader_mapping.begin(), leader_mapping.begin() + graph.n(), leader_mapping.begin());
  const DNodeID c_n = leader_mapping[graph.n() - 1]; // number of nodes in the coarse graph

  // Assign coarse node ID to all nodes; this works due to (I)
  graph.pfor_nodes([&](const DNodeID u) { mapping[u] = leader_mapping[clustering[u]]; });
  graph.pfor_nodes([&](const DNodeID u) { --mapping[u]; });

  buckets_index.clear();
  buckets_index.resize(c_n + 1);

  //
  // Sort nodes into buckets: place all nodes belonging to coarse node i into the i-th bucket
  //
  // Count the number of nodes in each bucket, then compute the position of the bucket in the global buckets array
  // using a prefix sum, roughly 2/5-th of time on europe.osm with 2/3-th to 1/3-tel for loop to prefix sum
  graph.pfor_nodes([&](const DNodeID u) { buckets_index[mapping[u]].fetch_add(1, std::memory_order_relaxed); });

  shm::parallel::prefix_sum(buckets_index.begin(), buckets_index.end(), buckets_index.begin());
  ASSERT(buckets_index.back() <= graph.n());

  // Sort nodes into   buckets, roughly 3/5-th of time on europe.osm
  tbb::parallel_for(static_cast<DNodeID>(0), graph.n(), [&](const DNodeID u) {
    const std::size_t pos = buckets_index[mapping[u]].fetch_sub(1, std::memory_order_relaxed) - 1;
    buckets[pos] = u;
  });

  //
  // Build nodes array of the coarse graph
  // - firstly, we count the degree of each coarse node
  // - secondly, we obtain the nodes array using a prefix sum
  //
  shm::StaticArray<DEdgeID> c_nodes{c_n + 1};
  shm::StaticArray<DNodeWeight> c_node_weights{c_n};

  tbb::enumerable_thread_specific<shm::RatingMap<DEdgeWeight>> collector{[&] { return shm::RatingMap<DEdgeWeight>(c_n); }};

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

  tbb::parallel_for(tbb::blocked_range<DNodeID>(0, c_n), [&](const auto &r) {
    auto &local_collector = collector.local();
    auto &local_edge_buffer = shared_edge_buffer.local();
    auto &local_node_buffer = shared_node_buffer.local();

    for (DNodeID c_u = r.begin(); c_u != r.end(); ++c_u) {
      std::size_t edge_buffer_position = local_edge_buffer.get_current_position();
      local_node_buffer.emplace_back(c_u, edge_buffer_position, &local_edge_buffer);

      const std::size_t first = buckets_index[c_u];
      const std::size_t last = buckets_index[c_u + 1];

      // build coarse graph
      auto collect_edges = [&](auto &map) {
        DNodeWeight c_u_weight = 0;
        for (std::size_t i = first; i < last; ++i) {
          const DNodeID u = buckets[i];
          ASSERT(mapping[u] == c_u);

          c_u_weight += graph.node_weight(u); // coarse node weight

          // collect coarse edges
          for (const auto [e, v] : graph.neighbors(u)) {
            const DNodeID c_v = mapping[v];
            if (c_u != c_v) { map[c_v] += graph.edge_weight(e); }
          }
        }

        c_node_weights[c_u] = c_u_weight; // coarse node weights are done now
        c_nodes[c_u + 1] = map.size();    // node degree (used to build c_nodes)

        // since we don't know the value of c_nodes[c_u] yet (so far, it only holds the nodes degree), we can't place the
        // edges of c_u in the c_edges and c_edge_weights arrays; hence, we store them in auxiliary arrays and note their
        // position in the auxiliary arrays
        for (const auto [c_v, weight] : map.entries()) { local_edge_buffer.push(c_v, weight); }
        map.clear();
      };

      // to select the right map, we compute a upper bound on the coarse node degree by summing the degree of all fine
      // nodes
      DEdgeID upper_bound_degree = 0;
      for (std::size_t i = first; i < last; ++i) {
        const DNodeID u = buckets[i];
        upper_bound_degree += graph.degree(u);
      }
      local_collector.update_upper_bound_size(upper_bound_degree);
      local_collector.run_with_map(collect_edges, collect_edges);
    }
  });

  shm::parallel::prefix_sum(c_nodes.begin(), c_nodes.end(), c_nodes.begin());

  ASSERT(c_nodes[0] == 0) << V(c_nodes);
  const DEdgeID c_m = c_nodes.back();

  //
  // Construct rest of the coarse graph: edges, edge weights
  //
  if (all_buffered_nodes.size() < c_n) { all_buffered_nodes.resize(c_n); }

  shm::parallel::IntegralAtomicWrapper<std::size_t> global_pos = 0;

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

  shm::StaticArray<DNodeID> c_edges{c_m};
  shm::StaticArray<DEdgeWeight> c_edge_weights{c_m};

  // build coarse graph
  tbb::parallel_for(static_cast<DNodeID>(0), c_n, [&](const DNodeID i) {
    const auto &buffered_node = all_buffered_nodes[i];
    const auto *chunks = buffered_node.chunks;
    const DNodeID c_u = buffered_node.c_u;

    const DEdgeID c_u_degree = c_nodes[c_u + 1] - c_nodes[c_u];
    const DEdgeID first_target_index = c_nodes[c_u];
    const DEdgeID first_source_index = buffered_node.position;

    for (std::size_t j = 0; j < c_u_degree; ++j) {
      const auto to = first_target_index + j;
      const auto [c_v, weight] = chunks->get(first_source_index + j);
      c_edges[to] = c_v;
      c_edge_weights[to] = weight;
    }
  });

  throw "";
//  return {DistributedGraph{std::move(c_nodes), std::move(c_edges), std::move(c_node_weights), std::move(c_edge_weights)},
//          std::move(mapping), std::move(m_ctx)};
}

} // namespace dkaminpar::graph