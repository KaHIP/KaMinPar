/*******************************************************************************
 * @file:   initial_coarsener.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Sequential coarsener based on label propagation with leader
 * locking.
 ******************************************************************************/
#include "kaminpar-shm/initial_partitioning/initial_coarsener.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::ip {
InitialCoarsener::InitialCoarsener(
    const Graph *graph, const InitialCoarseningContext &c_ctx, MemoryContext &&m_ctx
)
    : _input_graph(graph),
      _current_graph(graph),
      _hierarchy(graph),
      _c_ctx(c_ctx),
      _clustering(std::move(m_ctx.clustering)),
      _rating_map(std::move(m_ctx.rating_map)),
      _cluster_sizes(std::move(m_ctx.cluster_sizes)),
      _leader_node_mapping(std::move(m_ctx.leader_node_mapping)),
      _edge_weight_collector(std::move(m_ctx.edge_weight_collector)),
      _cluster_nodes(std::move(m_ctx.cluster_nodes)) {
  if (_clustering.size() < _input_graph->n() + 1) {
    _clustering.resize(_input_graph->n() + 1);
  }
  if (_rating_map.capacity() < _input_graph->n()) {
    _rating_map.resize(_input_graph->n());
  }
  if (_cluster_sizes.size() < _input_graph->n()) {
    _cluster_sizes.resize(_input_graph->n());
  }
  if (_leader_node_mapping.size() < _input_graph->n()) {
    _leader_node_mapping.resize(_input_graph->n());
  }
  if (_edge_weight_collector.capacity() < _input_graph->n()) {
    _edge_weight_collector.resize(_input_graph->n());
  } // c_n
  if (_cluster_nodes.size() < _input_graph->n()) {
    _cluster_nodes.resize(_input_graph->n());
  }
}

InitialCoarsener::InitialCoarsener(const Graph *graph, const InitialCoarseningContext &c_ctx)
    : InitialCoarsener(graph, c_ctx, MemoryContext{}) {}

const Graph *
InitialCoarsener::coarsen(const std::function<NodeWeight(NodeID)> &cb_max_cluster_weight) {
  const NodeWeight max_cluster_weight = cb_max_cluster_weight(_current_graph->n());
  if (!_precomputed_clustering) {
    perform_label_propagation(max_cluster_weight);
  }

  const NodeID c_n = _current_graph->n() - _current_num_moves;
  const bool converged = (1.0 - 1.0 * c_n / _current_graph->n()) <= _c_ctx.convergence_threshold;

  if (!converged) {
    _interleaved_max_cluster_weight = cb_max_cluster_weight(c_n);
    auto [c_graph, c_mapping] = contract_current_clustering();
    _hierarchy.take_coarse_graph(std::move(c_graph), std::move(c_mapping));
    _current_graph = &_hierarchy.coarsest_graph();
  }

  return _current_graph;
}

PartitionedGraph InitialCoarsener::uncoarsen(PartitionedGraph &&c_p_graph) {
  PartitionedGraph p_graph = _hierarchy.pop_and_project(std::move(c_p_graph));
  _current_graph = &_hierarchy.coarsest_graph();
  return p_graph;
}

InitialCoarsener::MemoryContext InitialCoarsener::free() {
  return {
      .clustering = std::move(_clustering),
      .cluster_sizes = std::move(_cluster_sizes),
      .leader_node_mapping = std::move(_leader_node_mapping),
      .rating_map = std::move(_rating_map),
      .edge_weight_collector = std::move(_edge_weight_collector),
      .cluster_nodes = std::move(_cluster_nodes),
  };
}

NodeID InitialCoarsener::pick_cluster(
    const NodeID u, const NodeWeight u_weight, const NodeWeight max_cluster_weight
) {
  KASSERT(_rating_map.empty());
  for (const auto [e, v] : _current_graph->neighbors(u)) {
    _rating_map[_clustering[v].leader] += _current_graph->edge_weight(e);
  }

  return pick_cluster_from_rating_map(u, u_weight, max_cluster_weight);
}

NodeID InitialCoarsener::pick_cluster_from_rating_map(
    NodeID u, NodeWeight u_weight, NodeWeight max_cluster_weight
) {
  NodeID best_cluster = u;
  EdgeWeight best_cluster_gain = 0;

  for (const NodeID cluster : _rating_map.used_entry_ids()) {
    const EdgeWeight gain = _rating_map[cluster];
    _rating_map[cluster] = 0;

    const NodeWeight weight = _clustering[cluster].weight;

    if ((gain > best_cluster_gain || (gain == best_cluster_gain && _rand.random_bool())) &&
        (weight + u_weight <= max_cluster_weight)) {
      best_cluster = cluster;
      best_cluster_gain = gain;
    }
  }
  _rating_map.used_entry_ids().clear();

  return best_cluster;
}

void InitialCoarsener::perform_label_propagation(const NodeWeight max_cluster_weight) {
  reset_current_clustering();

  const auto max_bucket = math::floor_log2(_c_ctx.large_degree_threshold);
  for (std::size_t bucket = 0;
       bucket < std::min<std::size_t>(max_bucket, _current_graph->number_of_buckets());
       ++bucket) {
    const NodeID bucket_size = static_cast<NodeID>(_current_graph->bucket_size(bucket));
    if (bucket_size == 0) {
      continue;
    }

    const NodeID first_node = _current_graph->first_node_in_bucket(bucket);
    const auto num_chunks = bucket_size / kChunkSize;

    std::vector<std::size_t> chunks(num_chunks);
    std::iota(chunks.begin(), chunks.end(), 0);
    std::transform(chunks.begin(), chunks.end(), chunks.begin(), [first_node](const std::size_t i) {
      return first_node + i * kChunkSize;
    });
    Random::instance().shuffle(chunks);

    for (const NodeID chunk_offset : chunks) {
      const auto &permutation{_random_permutations.get()};
      for (const NodeID local_u : permutation) {
        handle_node(chunk_offset + local_u, max_cluster_weight);
      }
    }

    const NodeID last_chunk_size{bucket_size % kChunkSize};
    const NodeID last_start{first_node + bucket_size - last_chunk_size};
    for (NodeID local_u = 0; local_u < last_chunk_size; ++local_u) {
      handle_node(last_start + local_u, max_cluster_weight);
    }
  }

  _precomputed_clustering = true;
}

void InitialCoarsener::handle_node(const NodeID u, const NodeWeight max_cluster_weight) {
  if (!_clustering[u].locked) {
    const NodeWeight u_weight{_current_graph->node_weight(u)};
    const NodeID best_cluster{pick_cluster(u, u_weight, max_cluster_weight)};

    if (best_cluster != u) {
      _clustering[u].leader = best_cluster;
      _clustering[best_cluster].locked = true;
      _clustering[best_cluster].weight += u_weight;
      ++_current_num_moves;
    }
  }
}

InitialCoarsener::ContractionResult InitialCoarsener::contract_current_clustering() {
  StaticArray<EdgeID> c_nodes{};
  StaticArray<NodeID> c_edges{};
  StaticArray<NodeWeight> c_node_weights{};
  StaticArray<EdgeWeight> c_edge_weights{};
  std::vector<NodeID> node_mapping{};

  const NodeID n = _current_graph->n();
  NodeID c_n = n - _current_num_moves;

  node_mapping.resize(_current_graph->n());
  c_nodes.resize(c_n + 1);
  c_node_weights.resize(c_n);
  c_edges.resize(_current_graph->m(), StaticArray<NodeID>::no_init{});            // overestimate
  c_edge_weights.resize(_current_graph->m(), StaticArray<EdgeWeight>::no_init{}); // overestimate

  std::fill(_cluster_sizes.begin(), _cluster_sizes.end(), 0);
  std::fill(_leader_node_mapping.begin(), _leader_node_mapping.end(), 0);
  // _clustering does not need to be cleared

  KASSERT(_current_graph->n() <= _cluster_sizes.size());
  KASSERT(_current_graph->n() <= _leader_node_mapping.size());
  KASSERT(_current_graph->n() <= _cluster_nodes.size());

  {
    NodeID current_node = 0;

    // - build node_mapping[u] = node id of u in the coarse graph
    // - build c_node_weights[c_u] = node weight array of the coarse graph
    // - build _cluster_sizes[c_u] = number of nodes that get contracted to form
    // coarse node c_u
    // - find number of coarse nodes c_n
    for (const NodeID u : _current_graph->nodes()) {
      const NodeID leader{_clustering[u].leader};
      if (_leader_node_mapping[leader] == 0) {
        c_node_weights[current_node] = _clustering[leader].weight; // index c_u
        _leader_node_mapping[leader] = ++current_node;             // 1-based
      }

      const NodeID cluster{_leader_node_mapping[leader] - 1}; // leader_node_mapping is 1-based
      node_mapping[u] = cluster;
      ++_cluster_sizes[cluster];
    }

    // turn _cluster_sizes into a "first node of"-array: next, we place all
    // nodes corresponding to coarse node 0 in
    // _cluster_nodes[_cluster_sizes[0]].._cluster_nodes[_cluster_sizes[1] - 1],
    // all nodes corresponding to coarse node 1 in
    // _cluster_nodes[_cluster_sizes[1]].._cluster_nodes[_cluster_sizes[2] - 1]
    // and so on
    NodeID counter = 0;
    for (NodeID &_cluster_size : _cluster_sizes) {
      counter += std::exchange(_cluster_size, counter);
    }

    // build the _cluster_nodes[] array as described above
    for (const NodeID u : _current_graph->nodes()) {
      _cluster_nodes[_cluster_sizes[node_mapping[u]]++] = u;
    }

    // initialize clustering data structures for computing a clustering of the
    // coarse graph (interleaved clustering computation)
    reset_current_clustering(c_n, c_node_weights);
  }

  // Here, we have the following arrays, with n -- number of nodes in current
  // graph, c_n -- number of nodes in coarse
  //   graph:
  //
  // _cluster_nodes[0..n]: first all nodes in cluster 0, then all nodes in
  // cluster 1, ...
  //
  // node_mapping[0..n]: coarse node id of node 0, node 1, node 2, ...
  //
  // _cluster_sizes[0..c_n]: first node in the *following* cluster, i.e.,
  //   _cluster_nodes[_cluster_sizes[0]] = first node in cluster 1
  //   Hence, the size of cluster 0 is `_cluster_sizes[0]`, the size of cluster
  //   1 is
  //   `_cluster_sizes[1] - _cluster_sizes[0]` and so on

  {
    // note: c_node_weights is already set
    NodeID c_u = 0;
    NodeID c_m = 0;
    c_nodes[0] = 0;

    for (std::size_t i = 0; i < n; ++i) { // n might be smaller than _cluster_nodes[]
      const NodeID u = _cluster_nodes[i];

      // node mapping points to the next coarse graph, hence we've seen all
      // nodes in the last cluster we now add it to the coarse graph
      if (node_mapping[u] != c_u) {
        KASSERT(node_mapping[u] == c_u + 1, V(u) << V(node_mapping[u]) << V(c_u));

        interleaved_handle_node(c_u, c_node_weights[c_u]);
        for (const auto c_v : _edge_weight_collector.used_entry_ids()) {
          const EdgeWeight weight = _edge_weight_collector.get(c_v);
          c_edges[c_m] = c_v;
          c_edge_weights[c_m] = weight;
          ++c_m;
        }
        _edge_weight_collector.clear();
        c_nodes[++c_u] = c_m;
      }

      for (const auto [e, v] : _current_graph->neighbors(u)) {
        const NodeID c_v = node_mapping[v];
        if (c_u != c_v) {
          const EdgeWeight weight{_current_graph->edge_weight(e)};
          _edge_weight_collector[c_v] += weight;
          interleaved_visit_neighbor(c_u, c_v, weight);
        }
      }
    }

    // finish last cluster ...
    interleaved_handle_node(c_u, c_node_weights[c_u]);
    for (const auto c_v : _edge_weight_collector.used_entry_ids()) {
      const EdgeWeight weight{_edge_weight_collector.get(c_v)};
      c_edges[c_m] = c_v;
      c_edge_weights[c_m] = weight;
      ++c_m;
    }
    c_nodes[++c_u] = c_m;
    _edge_weight_collector.clear();

    KASSERT(c_u == c_n);
    c_edges.restrict(c_m);
    c_edge_weights.restrict(c_m);
  }

  Graph coarse_graph(
      Graph::seq{},
      std::move(c_nodes),
      std::move(c_edges),
      std::move(c_node_weights),
      std::move(c_edge_weights)
  );

  return {std::move(coarse_graph), std::move(node_mapping)};
}
} // namespace kaminpar::shm::ip
