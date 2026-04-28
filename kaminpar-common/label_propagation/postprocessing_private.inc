/*******************************************************************************
 * Private label-propagation post-processing member functions.
 ******************************************************************************/

void reset_state() {
  tbb::parallel_invoke(
      [&] {
        tbb::parallel_for<NodeID>(0, _num_active_nodes, [&](const NodeID u) {
          _active_set.initialize_node(u);

          const ClusterID initial_cluster = _labels.initial_cluster(u);
          _labels.init_cluster(u, initial_cluster);
          if (_config.selection.track_favored_clusters && u < _workspace.favored_clusters.size()) {
            _workspace.favored_clusters[u] = initial_cluster;
          }
          if (u < _workspace.moved.size()) {
            _workspace.moved[u] = 0;
          }
          if constexpr (requires(LabelStore labels, NodeID node) {
                          labels.reset_node_state(node);
                        }) {
            _labels.reset_node_state(u);
          }
        });
      },
      [&] {
        tbb::parallel_for<ClusterID>(0, _initial_num_clusters, [&](const ClusterID cluster) {
          _weights.init_cluster_weight(cluster, _weights.initial_cluster_weight(cluster));
        });
      }
  );
  _expected_total_gain = 0;
  _current_num_clusters = _initial_num_clusters;
}

template <bool kMatch> void handle_isolated_nodes(const NodeID from, const NodeID to) {
  constexpr ClusterID kInvalidClusterID = std::numeric_limits<ClusterID>::max();
  tbb::enumerable_thread_specific<ClusterID> current_cluster_ets(kInvalidClusterID);

  tbb::parallel_for(
      tbb::blocked_range<NodeID>(from, std::min(_graph.n(), to)),
      [&](tbb::blocked_range<NodeID> r) {
        ClusterID cluster = current_cluster_ets.local();

        for (NodeID u = r.begin(); u != r.end(); ++u) {
          if (_graph.degree(u) == 0) {
            const ClusterID cu = _labels.cluster(u);

            if (cluster != kInvalidClusterID &&
                _weights.move_cluster_weight(
                    cu, cluster, _weights.cluster_weight(cu), _weights.max_cluster_weight(cluster)
                )) {
              _labels.move_node(u, cluster);
              if constexpr (kMatch) {
                cluster = kInvalidClusterID;
              }
            } else {
              cluster = cu;
            }
          }
        }

        current_cluster_ets.local() = cluster;
      }
  );
}

template <bool kMatch, bool kThreadwise>
void handle_two_hop_nodes(const NodeID from, const NodeID to) {
  KASSERT(_config.selection.track_favored_clusters);

  if constexpr (kThreadwise) {
    handle_two_hop_nodes_threadwise_impl<kMatch>(from, to);
  } else {
    handle_two_hop_nodes_impl<kMatch>(from, to);
  }
}

[[nodiscard]] bool is_considered_for_two_hop_clustering(const NodeID u) {
  if (_graph.degree(u) == 0) {
    return false;
  }

  const auto check_cluster_weight = [&](const ClusterID c_u) {
    const ClusterWeight current_weight = _weights.cluster_weight(c_u);

    if (current_weight > _weights.max_cluster_weight(c_u) / 2 ||
        current_weight != _weights.initial_cluster_weight(c_u)) {
      return false;
    }

    return true;
  };

  if (_relabeled) {
    if (u < _workspace.moved.size() && _workspace.moved[u]) {
      return false;
    }

    const ClusterID c_u = _labels.cluster(u);
    return check_cluster_weight(c_u);
  } else {
    if (u != _labels.cluster(u)) {
      return false;
    }

    return check_cluster_weight(u);
  }
}

template <bool kMatch>
void handle_two_hop_nodes_threadwise_impl(const NodeID from, const NodeID to) {
  tbb::enumerable_thread_specific<DynamicFlatMap<ClusterID, NodeID>> matching_map_ets;

  auto handle_node = [&](DynamicFlatMap<ClusterID, NodeID> &matching_map, const NodeID u) {
    const ClusterID c_u = _labels.cluster(u);
    ClusterID &rep_key = matching_map[_workspace.favored_clusters[u]];

    if (rep_key == 0) {
      rep_key = c_u + 1;
    } else {
      const ClusterID rep = rep_key - 1;

      const bool could_move_u_to_rep = _weights.move_cluster_weight(
          c_u, rep, _weights.cluster_weight(c_u), _weights.max_cluster_weight(rep)
      );

      if constexpr (kMatch) {
        KASSERT(could_move_u_to_rep);
        _labels.move_node(u, rep);
        rep_key = 0;
      } else {
        if (could_move_u_to_rep) {
          _labels.move_node(u, rep);
        } else {
          rep_key = c_u + 1;
        }
      }
    }
  };

  tbb::parallel_for(
      tbb::blocked_range<NodeID>(from, std::min(to, _graph.n()), 512),
      [&](const tbb::blocked_range<NodeID> &r) {
        auto &matching_map = matching_map_ets.local();

        for (NodeID u = r.begin(); u != r.end(); ++u) {
          if (is_considered_for_two_hop_clustering(u)) {
            handle_node(matching_map, u);
          }
        }
      }
  );
}

template <bool kMatch> void handle_two_hop_nodes_impl(const NodeID from, const NodeID to) {
  const auto is_considered_for_non_threadwise_two_hop_clustering = [&](const NodeID u) {
    if (_graph.degree(u) == 0 || u != _labels.cluster(u)) {
      return false;
    }

    const ClusterWeight current_weight = _weights.cluster_weight(u);
    return current_weight <= _weights.max_cluster_weight(u) / 2 &&
           current_weight == _weights.initial_cluster_weight(u);
  };

  tbb::parallel_for(from, std::min(to, _graph.n()), [&](const NodeID u) {
    if (is_considered_for_non_threadwise_two_hop_clustering(u)) {
      const NodeID cluster = _workspace.favored_clusters[u];
      if (is_considered_for_non_threadwise_two_hop_clustering(cluster) &&
          _weights.move_cluster_weight(
              u, cluster, _weights.cluster_weight(u), _weights.max_cluster_weight(cluster)
          )) {
        _labels.move_node(u, cluster);
        --_current_num_clusters;
      }
    } else {
      _workspace.favored_clusters[u] = u;
    }
  });

  tbb::parallel_for(from, std::min(to, _graph.n()), [&](const NodeID u) {
    if (should_stop() || !is_considered_for_non_threadwise_two_hop_clustering(u)) {
      return;
    }

    const NodeID C = __atomic_load_n(&_workspace.favored_clusters[u], __ATOMIC_RELAXED);
    auto &sync = _workspace.favored_clusters[C];

    do {
      NodeID cluster = sync;

      if (cluster == C) {
        if (__atomic_compare_exchange_n(
                &sync, &cluster, u, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
            )) {
          break;
        }
        if (cluster == C) {
          continue;
        }
      }

      KASSERT(__atomic_load_n(&_workspace.favored_clusters[cluster], __ATOMIC_RELAXED) == C);

      if constexpr (kMatch) {
        if (__atomic_compare_exchange_n(
                &sync, &cluster, C, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
            )) {
          [[maybe_unused]] const bool success = _weights.move_cluster_weight(
              u, cluster, _weights.cluster_weight(u), _weights.max_cluster_weight(cluster)
          );
          KASSERT(success);

          _labels.move_node(u, cluster);
          break;
        }
      } else {
        if (_weights.move_cluster_weight(
                u, cluster, _weights.cluster_weight(u), _weights.max_cluster_weight(cluster)
            )) {
          _labels.move_node(u, cluster);
          break;
        } else if (
            __atomic_compare_exchange_n(
                &sync, &cluster, u, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
            )
        ) {
          break;
        }
      }
    } while (true);
  });
}
