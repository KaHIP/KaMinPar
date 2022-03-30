/*******************************************************************************
 * @file:   graph.cc
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Static graph data structure with dynamic partition wrapper.
 ******************************************************************************/
#include "kaminpar/datastructure/graph.h"

#include "kaminpar/parallel/accumulate.h"
#include "kaminpar/parallel/max_element.h"
#include "kaminpar/utility/math.h"
#include "kaminpar/utility/timer.h"

namespace kaminpar {
Degree lowest_degree_in_bucket(const std::size_t bucket) { return (1u << bucket) >> 1u; }
Degree degree_bucket(const Degree degree) { return (degree == 0) ? 0 : math::floor_log2(degree) + 1; }

//
// Graph
//

Graph::Graph(StaticArray<EdgeID> nodes, StaticArray<NodeID> edges, StaticArray<NodeWeight> node_weights,
             StaticArray<EdgeWeight> edge_weights, const bool sorted)
    : _nodes{std::move(nodes)}, _edges{std::move(edges)}, _node_weights{std::move(node_weights)},
      _edge_weights{std::move(edge_weights)}, _sorted{sorted} {
  if (_node_weights.empty()) {
    _total_node_weight = static_cast<NodeWeight>(n());
    _max_node_weight = 1;
  } else {
    _total_node_weight = parallel::accumulate(_node_weights, 0);
    _max_node_weight = parallel::max_element(_node_weights);
  }

  if (_edge_weights.empty()) {
    _total_edge_weight = static_cast<EdgeWeight>(m());
  } else {
    _total_edge_weight = parallel::accumulate(_edge_weights, 0);
  }

  init_degree_buckets();
}

Graph::Graph(tag::Sequential, StaticArray<EdgeID> nodes, StaticArray<NodeID> edges,
             StaticArray<NodeWeight> node_weights, StaticArray<EdgeWeight> edge_weights, const bool sorted)
    : _nodes{std::move(nodes)}, _edges{std::move(edges)}, _node_weights{std::move(node_weights)},
      _edge_weights{std::move(edge_weights)}, _sorted{sorted} {
  if (_node_weights.empty()) {
    _total_node_weight = static_cast<NodeWeight>(n());
    _max_node_weight = 1;
  } else {
    _total_node_weight = std::accumulate(_node_weights.begin(), _node_weights.end(), 0);
    _max_node_weight = *std::max_element(_node_weights.begin(), _node_weights.end());
  }

  if (_edge_weights.empty()) {
    _total_edge_weight = static_cast<EdgeWeight>(m());
  } else {
    _total_edge_weight = std::accumulate(_edge_weights.begin(), _edge_weights.end(), 0);
  }

  init_degree_buckets();
}

void Graph::init_degree_buckets() {
  ASSERT(std::all_of(_buckets.begin(), _buckets.end(), [](const auto n) { return n == 0; }));
  if (_sorted) {
    for (const NodeID u : nodes()) {
      ++_buckets[degree_bucket(degree(u)) + 1];
    }
    auto last_nonempty_bucket = std::find_if(_buckets.rbegin(), _buckets.rend(), [](const auto n) { return n > 0; });
    _number_of_buckets = std::distance(_buckets.begin(), (last_nonempty_bucket + 1).base());
  } else {
    _buckets[1] = n();
    _number_of_buckets = 1;
  }
  std::partial_sum(_buckets.begin(), _buckets.end(), _buckets.begin());
}

void Graph::update_total_node_weight() {
  if (_node_weights.empty()) {
    _total_node_weight = n();
    _max_node_weight = 1;
  } else {
    _total_node_weight = std::accumulate(_node_weights.begin(), _node_weights.end(), 0);
    _max_node_weight = *std::max_element(_node_weights.begin(), _node_weights.end());
  }
}

//
// Utility debug functions
//

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
        if (u != u_prime) {
          continue;
        }
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

//
// PartitionedGraph
//

PartitionedGraph::PartitionedGraph(const Graph &graph, BlockID k, StaticArray<BlockID> partition,
                                   scalable_vector<BlockID> final_k)
    : _graph{&graph}, _k{k}, _partition{std::move(partition)}, _block_weights{k}, _final_k{std::move(final_k)} {
  if (graph.n() > 0 && _partition.empty()) {
    _partition.resize(_graph->n(), kInvalidBlockID);
  }
  if (_final_k.empty()) {
    _final_k.resize(k, 1);
  }
  ASSERT(_partition.size() == graph.n());

  init_block_weights();
}

PartitionedGraph::PartitionedGraph(tag::Sequential, const Graph &graph, BlockID k, StaticArray<BlockID> partition,
                                   scalable_vector<BlockID> final_k)
    : _graph{&graph}, _k{k}, _partition{std::move(partition)}, _block_weights{k}, _final_k{std::move(final_k)} {
  if (graph.n() > 0 && _partition.empty()) {
    _partition.resize(_graph->n(), kInvalidBlockID);
  }
  if (_final_k.empty()) {
    _final_k.resize(k, 1);
  }
  ASSERT(_partition.size() == graph.n());

  init_block_weights_seq();
}

void PartitionedGraph::change_k(const BlockID new_k) {
  _block_weights = StaticArray<parallel::Atomic<BlockWeight>>{new_k};
  _final_k.resize(new_k);
  _k = new_k;
}
} // namespace kaminpar
