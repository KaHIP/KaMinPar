/*******************************************************************************
 * Public label-propagation post-processing member functions.
 ******************************************************************************/

void match_isolated_nodes(
    const NodeID from = 0, const NodeID to = std::numeric_limits<NodeID>::max()
) {
  handle_isolated_nodes<true>(from, to);
}

void cluster_isolated_nodes(
    const NodeID from = 0, const NodeID to = std::numeric_limits<NodeID>::max()
) {
  handle_isolated_nodes<false>(from, to);
}

void match_two_hop_nodes(
    const NodeID from = 0, const NodeID to = std::numeric_limits<NodeID>::max()
) {
  handle_two_hop_nodes<true, false>(from, to);
}

void cluster_two_hop_nodes(
    const NodeID from = 0, const NodeID to = std::numeric_limits<NodeID>::max()
) {
  handle_two_hop_nodes<false, false>(from, to);
}

void match_two_hop_nodes_threadwise(
    const NodeID from = 0, const NodeID to = std::numeric_limits<NodeID>::max()
) {
  handle_two_hop_nodes<true, true>(from, to);
}

void cluster_two_hop_nodes_threadwise(
    const NodeID from = 0, const NodeID to = std::numeric_limits<NodeID>::max()
) {
  handle_two_hop_nodes<false, true>(from, to);
}

void relabel_clusters() {
  SCOPED_HEAP_PROFILER("Relabel");
  SCOPED_TIMER("Relabel");

  ClusterID num_actual_clusters = _current_num_clusters;
  _initial_num_clusters = num_actual_clusters;
  _relabeled = true;

  if (_workspace.moved.size() < _graph.n()) {
    _workspace.moved.resize(_graph.n());
  }

  StaticArray<ClusterID> mapping(_graph.n());
  tbb::parallel_for(tbb::blocked_range<NodeID>(0, _graph.n()), [&](const auto &r) {
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      const ClusterID c_u = _labels.cluster(u);
      __atomic_store_n(&mapping[c_u], 1, __ATOMIC_RELAXED);

      if (u != c_u) {
        _workspace.moved[u] = 1;
      }
    }
  });

  parallel::prefix_sum(mapping.begin(), mapping.end(), mapping.begin());
  KASSERT(num_actual_clusters == mapping[_graph.n() - 1]);

  tbb::parallel_for(tbb::blocked_range<NodeID>(0, _graph.n()), [&](const auto &r) {
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      _labels.move_node(u, mapping[_labels.cluster(u)] - 1);

      if (u < _workspace.favored_clusters.size()) {
        _workspace.favored_clusters[u] = mapping[_workspace.favored_clusters[u]] - 1;
      }
    }
  });

  _weights.reassign_cluster_weights(mapping, num_actual_clusters);
}
