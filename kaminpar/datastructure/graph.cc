#include "datastructure/graph.h"

#include "utility/timer.h"

namespace kaminpar {
Degree lowest_degree_in_bucket(const std::size_t bucket) { return (1u << bucket) >> 1u; }
Degree degree_bucket(const Degree degree) { return (degree == 0) ? 0 : math::floor_log2(degree) + 1; }

//
// Graph
//

Graph::Graph(StaticArray<EdgeID> nodes, StaticArray<NodeID> edges, StaticArray<NodeWeight> node_weights,
             StaticArray<EdgeWeight> edge_weights, const bool sorted)
    : _nodes{std::move(nodes)},
      _edges{std::move(edges)},
      _node_weights{std::move(node_weights)},
      _edge_weights{std::move(edge_weights)},
      _sorted{sorted} {
  if (_node_weights.empty()) {
    _total_node_weight = n();
    _max_node_weight = 1;
  } else {
    _total_node_weight = parallel::accumulate(_node_weights);
    _max_node_weight = parallel::max_element(_node_weights);
  }

  if (_edge_weights.empty()) {
    _total_edge_weight = m();
  } else {
    _total_edge_weight = parallel::accumulate(_edge_weights);
  }

  init_degree_buckets();
}

Graph::Graph(tag::Sequential, StaticArray<EdgeID> nodes, StaticArray<NodeID> edges,
             StaticArray<NodeWeight> node_weights, StaticArray<EdgeWeight> edge_weights, const bool sorted)
    : _nodes{std::move(nodes)},
      _edges{std::move(edges)},
      _node_weights{std::move(node_weights)},
      _edge_weights{std::move(edge_weights)},
      _sorted{sorted} {
  if (_node_weights.empty()) {
    _total_node_weight = n();
    _max_node_weight = 1;
  } else {
    _total_node_weight = std::accumulate(_node_weights.begin(), _node_weights.end(), 0);
    _max_node_weight = *std::max_element(_node_weights.begin(), _node_weights.end());
  }

  if (_edge_weights.empty()) {
    _total_edge_weight = m();
  } else {
    _total_edge_weight = std::accumulate(_edge_weights.begin(), _edge_weights.end(), 0);
  }

  init_degree_buckets();
}

void Graph::print() const {
  LOG << "n=" << n() << " m=" << m();
  for (const NodeID u : nodes()) {
    LLOG << u << " ";
    for (const auto [e, v] : neighbors(u)) { LLOG << "->" << v << " "; }
    LOG;
  }
}

void Graph::print_weighted() const {
  LOG << "n=" << n() << " m=" << m();
  for (const NodeID u : nodes()) {
    LLOG << u << "|" << node_weight(u) << " ";
    for (const auto [e, v] : neighbors(u)) {
      LLOG << "-" << e << "|" << edge_weight(e) << "->" << v << "|" << node_weight(v) << " ";
    }
    LOG;
  }
}

void Graph::init_degree_buckets() {
  ASSERT(std::ranges::all_of(_buckets, [](const auto n) { return n == 0; }));
  if (_sorted) {
    for (const NodeID u : nodes()) { ++_buckets[degree_bucket(degree(u)) + 1]; }
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
// PartitionedGraph
//

PartitionedGraph::PartitionedGraph(const Graph &graph, BlockID k, StaticArray<BlockID> partition,
                                   scalable_vector<BlockID> final_k)
    : _graph{&graph},
      _k{k},
      _partition{std::move(partition)},
      _block_weights{k},
      _final_k{std::move(final_k)} {
  if (graph.n() > 0 && _partition.empty()) { _partition.resize(_graph->n(), kInvalidBlockID); }
  if (_final_k.empty()) { _final_k.resize(k, 1); }
  ASSERT(_partition.size() == graph.n());

  init_block_weights();
  GDBG(_block_names.resize(k));
}

PartitionedGraph::PartitionedGraph(tag::Sequential, const Graph &graph, BlockID k, StaticArray<BlockID> partition,
                                   scalable_vector<BlockID> final_k)
    : _graph{&graph},
      _k{k},
      _partition{std::move(partition)},
      _block_weights{k},
      _final_k{std::move(final_k)} {
  if (graph.n() > 0 && _partition.empty()) { _partition.resize(_graph->n(), kInvalidBlockID); }
  if (_final_k.empty()) { _final_k.resize(k, 1); }
  ASSERT(_partition.size() == graph.n());

  init_block_weights_seq();
  GDBG(_block_names.resize(k));
}

void PartitionedGraph::change_k(const BlockID new_k) {
  _block_weights = StaticArray<parallel::IntegralAtomicWrapper<BlockWeight>>{new_k};
  _final_k.resize(new_k);
  _k = new_k;
  GDBG(_block_names.resize(new_k));
}
} // namespace kaminpar