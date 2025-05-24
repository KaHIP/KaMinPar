/*******************************************************************************
 * Sequential label propagation coarsening used during initial bipartitionign.
 *
 * @file:   initial_coarsener.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/initial_partitioning/coarsening/initial_coarsener.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

namespace {

constexpr static bool kRandomizeNodeOrder = true;

}

InitialCoarsener::InitialCoarsener(const InitialCoarseningContext &c_ctx) : _c_ctx(c_ctx) {}

void InitialCoarsener::init(const CSRGraph &graph) {
  _input_graph = &graph;
  _current_graph = &graph;
  _hierarchy.init(graph);

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
  }
  if (_cluster_nodes.size() < _input_graph->n()) {
    _cluster_nodes.resize(_input_graph->n());
  }

  _precomputed_clustering = false;
}

const CSRGraph *InitialCoarsener::coarsen(const NodeWeight max_cluster_weight) {
  timer::LocalTimer timer;

  timer.reset();
  if (!_precomputed_clustering) {
    perform_label_propagation(max_cluster_weight);
  }
  _timings.lp_ms += timer.elapsed();

  const NodeID c_n = _current_graph->n() - _current_num_moves;
  const bool converged = (1.0 - 1.0 * c_n / _current_graph->n()) <= _c_ctx.convergence_threshold;

  if (!converged) {
    _interleaved_max_cluster_weight = max_cluster_weight;

    auto [c_graph, c_mapping] = contract_current_clustering();
    _hierarchy.push(std::move(c_graph), std::move(c_mapping));

    _current_graph = &_hierarchy.current();
  }

  _timings.total_ms += timer.elapsed();
  return _current_graph;
}

PartitionedCSRGraph InitialCoarsener::uncoarsen(PartitionedCSRGraph &&c_p_graph) {
  PartitionedCSRGraph p_graph = _hierarchy.pop(std::move(c_p_graph));
  _current_graph = &_hierarchy.current();
  return p_graph;
}

NodeID InitialCoarsener::pick_cluster(
    const NodeID u, const NodeWeight u_weight, const NodeWeight max_cluster_weight
) {
  KASSERT(_rating_map.empty());
  _current_graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
    _rating_map[_clustering[v].leader] += w;
  });

  return pick_cluster_from_rating_map(u, u_weight, max_cluster_weight);
}

NodeID InitialCoarsener::pick_cluster_from_rating_map(
    const NodeID u, const NodeWeight u_weight, const NodeWeight max_cluster_weight
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

  if constexpr (kRandomizeNodeOrder) {
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

      _chunks.clear();
      if (_chunks.size() < num_chunks) {
        _chunks.resize(num_chunks);
      }

      std::iota(_chunks.begin(), _chunks.begin() + num_chunks, 0);
      std::transform(
          _chunks.begin(),
          _chunks.begin() + num_chunks,
          _chunks.begin(),
          [first_node](const NodeID i) { return first_node + i * kChunkSize; }
      );
      _rand.shuffle(_chunks.begin(), _chunks.begin() + num_chunks);

      for (NodeID i = 0; i < num_chunks; ++i) {
        const NodeID chunk_offset = _chunks[i];
        const auto &permutation{_random_permutations.get()};
        for (const NodeID local_u : permutation) {
          handle_node(chunk_offset + local_u, max_cluster_weight);
        }
      }

      const NodeID last_chunk_size = bucket_size % kChunkSize;
      const NodeID last_start = first_node + bucket_size - last_chunk_size;

      for (NodeID local_u = 0; local_u < last_chunk_size; ++local_u) {
        handle_node(last_start + local_u, max_cluster_weight);
      }
    }
  } else {
    for (NodeID u : _current_graph->nodes()) {
      handle_node(u, max_cluster_weight);
    }
  }

  _precomputed_clustering = true;
}

void InitialCoarsener::handle_node(const NodeID u, const NodeWeight max_cluster_weight) {
  if (_clustering[u].locked) {
    return;
  }

  const NodeWeight u_weight = _current_graph->node_weight(u);
  const NodeID best_cluster = pick_cluster(u, u_weight, max_cluster_weight);

  if (best_cluster != u) {
    _clustering[u].leader = best_cluster;
    _clustering[best_cluster].locked = true;
    _clustering[best_cluster].weight += u_weight;
    ++_current_num_moves;
  }
}

InitialCoarsener::ContractionResult InitialCoarsener::contract_current_clustering() {
  const NodeID n = _current_graph->n();
  NodeID c_n = n - _current_num_moves;

  timer::LocalTimer timer;
  timer.reset();

  StaticArray<NodeID> node_mapping = _hierarchy.alloc_mapping_memory();
  node_mapping.unrestrict();

  if (node_mapping.size() < _current_graph->n()) {
    node_mapping.resize(_current_graph->n(), static_array::seq);
  }
  node_mapping.restrict(_current_graph->n());

  CSRGraphMemory c_memory = _hierarchy.alloc_graph_memory();
  StaticArray<EdgeID> c_nodes = std::move(c_memory.nodes);
  StaticArray<NodeID> c_edges = std::move(c_memory.edges);
  StaticArray<NodeWeight> c_node_weights = std::move(c_memory.node_weights);
  StaticArray<EdgeWeight> c_edge_weights = std::move(c_memory.edge_weights);

  std::vector<NodeID> buckets = std::move(c_memory.buckets);
  std::fill(buckets.begin(), buckets.end(), 0);

  c_nodes.unrestrict();
  c_node_weights.unrestrict();

  if (c_nodes.size() < c_n + 1) {
    c_nodes.resize(c_n + 1, static_array::seq);
  }
  if (c_node_weights.size() < c_n) {
    c_node_weights.resize(c_n, static_array::seq);
  }

  // CSRGraph determines the number of nodes based on the size of the c_nodes array:
  // thus, we must set the right size, since the array is generally larger than the graph
  c_nodes.restrict(c_n + 1);
  c_node_weights.restrict(c_n);

  const EdgeID prev_c_edges_size = c_edges.size();
  const EdgeID prev_c_edge_weights_size = c_edge_weights.size();
  c_edges.unrestrict();
  c_edge_weights.unrestrict();

  // Overcommit memory for the edge and edge weight array.
  const bool resize_edges = c_edges.size() < _current_graph->m();
  if (resize_edges) {
    c_edges.resize(
        _current_graph->m(), static_array::seq, static_array::noinit, static_array::overcommit
    );
  }
  const bool resize_edge_weights = c_edge_weights.size() < _current_graph->m();
  if (resize_edge_weights) {
    c_edge_weights.resize(
        _current_graph->m(), static_array::seq, static_array::noinit, static_array::overcommit
    );
  }

  // Similarly to the c_nodes array, we must restrict the size of the c_edges array: this is
  // done at the end, when the number of coarse edges is known

  _timings.alloc_ms += timer.elapsed();
  timer.reset();

  std::fill(_cluster_sizes.begin(), _cluster_sizes.begin() + n, 0);
  std::fill(_leader_node_mapping.begin(), _leader_node_mapping.begin() + n, 0);
  // Note: _clustering does not need to be cleared

  _timings.contract_ms += timer.elapsed();

  timer.reset();
  {
    NodeID current_node = 0;

    // - build node_mapping[u] = node id of u in the coarse graph
    // - build c_node_weights[c_u] = node weight array of the coarse graph
    // - build _cluster_sizes[c_u] = number of nodes that get contracted to form
    // coarse node c_u
    // - find number of coarse nodes c_n
    for (const NodeID u : _current_graph->nodes()) {
      const NodeID leader = _clustering[u].leader;
      if (_leader_node_mapping[leader] == 0) {
        c_node_weights[current_node] = _clustering[leader].weight; // index c_u
        _leader_node_mapping[leader] = ++current_node;             // 1-based
      }

      // Note: _leader_node_mapping is 1-based
      const NodeID cluster = _leader_node_mapping[leader] - 1;
      node_mapping[u] = cluster;
      ++_cluster_sizes[cluster];
    }

    // Turn _cluster_sizes into a "first node of"-array: next, we place all
    // nodes corresponding to coarse node 0 in
    // _cluster_nodes[_cluster_sizes[0]].._cluster_nodes[_cluster_sizes[1] - 1],
    // all nodes corresponding to coarse node 1 in
    // _cluster_nodes[_cluster_sizes[1]].._cluster_nodes[_cluster_sizes[2] - 1]
    // and so on
    NodeID counter = 0;
    for (NodeID u : _current_graph->nodes()) {
      counter += std::exchange(_cluster_sizes[u], counter);
    }

    // Build the _cluster_nodes[] array as described above
    for (const NodeID u : _current_graph->nodes()) {
      _cluster_nodes[_cluster_sizes[node_mapping[u]]++] = u;
    }

    // Initialize clustering data structures for computing a clustering of the
    // coarse graph (interleaved clustering computation)
    reset_current_clustering(c_n, c_node_weights);
  }
  _timings.interleaved1_ms += timer.elapsed();

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

  timer.reset();
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
        KASSERT(node_mapping[u] == c_u + 1);

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

      _current_graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
        const NodeID c_v = node_mapping[v];

        if (c_u != c_v) {
          _edge_weight_collector[c_v] += weight;
          interleaved_visit_neighbor(c_u, c_v, weight);
        }
      });
    }

    // Finish last cluster:
    interleaved_handle_node(c_u, c_node_weights[c_u]);
    for (const auto c_v : _edge_weight_collector.used_entry_ids()) {
      const EdgeWeight weight = _edge_weight_collector.get(c_v);

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
  _timings.interleaved2_ms += timer.elapsed();

  timer.reset();
  if constexpr (kHeapProfiling) {
    if (resize_edges) {
      heap_profiler::HeapProfiler::global().record_alloc(
          c_edges.data(), c_edges.size() * sizeof(NodeID)
      );
    } else if (c_edges.size() > prev_c_edges_size) {
      heap_profiler::HeapProfiler::global().record_free(c_edges.data());
      heap_profiler::HeapProfiler::global().record_alloc(
          c_edges.data(), c_edges.size() * sizeof(NodeID)
      );
    }

    if (resize_edge_weights) {
      heap_profiler::HeapProfiler::global().record_alloc(
          c_edge_weights.data(), c_edge_weights.size() * sizeof(EdgeWeight)
      );
    } else if (c_edge_weights.size() > prev_c_edge_weights_size) {
      heap_profiler::HeapProfiler::global().record_free(c_edge_weights.data());
      heap_profiler::HeapProfiler::global().record_alloc(
          c_edge_weights.data(), c_edge_weights.size() * sizeof(EdgeWeight)
      );
    }
  }

  CSRGraph coarse_graph(
      CSRGraph::seq{},
      std::move(c_nodes),
      std::move(c_edges),
      std::move(c_node_weights),
      std::move(c_edge_weights),
      false,
      std::move(buckets)
  );
  _timings.alloc_ms += timer.elapsed();

  return {std::move(coarse_graph), std::move(node_mapping)};
}

void InitialCoarsener::reset_current_clustering() {
  if (_current_graph->is_node_weighted()) {
    reset_current_clustering(_current_graph->n(), _current_graph->raw_node_weights());
  } else {
    // This is robust if _current_graph is empty
    // (in this case, we cannot use node_weight(0))
    reset_current_clustering_unweighted(
        _current_graph->n(), _current_graph->total_node_weight() / _current_graph->n()
    );
  }
}

void InitialCoarsener::reset_current_clustering_unweighted(
    const NodeID n, const NodeWeight unit_node_weight
) {
  _current_num_moves = 0;
  for (NodeID u = 0; u < n; ++u) {
    _clustering[u].locked = false;
    _clustering[u].leader = u;
    _clustering[u].weight = unit_node_weight;
  }
}

void InitialCoarsener::interleaved_handle_node(const NodeID c_u, const NodeWeight c_u_weight) {
  if (!_interleaved_locked) {
    const NodeID best_cluster =
        pick_cluster_from_rating_map(c_u, c_u_weight, _interleaved_max_cluster_weight);
    const bool changed_cluster = (best_cluster != c_u);

    if (changed_cluster) {
      ++_current_num_moves;
      _clustering[c_u].leader = best_cluster;
      _clustering[best_cluster].weight += c_u_weight;
      _clustering[best_cluster].locked = true;
    }
  }

  _interleaved_locked = _clustering[c_u + 1].locked;
}

void InitialCoarsener::interleaved_visit_neighbor(
    const NodeID, const NodeID c_v, const EdgeWeight weight
) {
  if (!_interleaved_locked) {
    _rating_map[_clustering[c_v].leader] += weight;
  }
}

} // namespace kaminpar::shm
