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
#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/utility/mpi_helper.h"
#include "kaminpar/datastructure/marker.h"
#include "kaminpar/datastructure/rating_map.h"
#include "kaminpar/utility/random.h"

#include <tbb/enumerable_thread_specific.h>

namespace dkaminpar {
template<typename DClusterID = DNodeID, typename DClusterWeight = DNodeWeight>
class DistributedLabelPropagation {
  SET_DEBUG(true);

  constexpr static int TAG_JOIN_MESSAGE = 1;

  struct ClusterSelectionState {
    shm::Randomize &local_rand;
    DNodeID u;
    DNodeWeight u_weight;
    DClusterID initial_cluster;
    DClusterWeight initial_cluster_weight;
    DClusterID best_cluster;
    DEdgeWeight best_gain;
    DClusterWeight best_cluster_weight;
    DClusterID current_cluster;
    DEdgeWeight current_gain;
    DClusterWeight current_cluster_weight;
  };

public:
  explicit DistributedLabelPropagation(DistributedPartitionedGraph *graph, const DClusterID num_clusters,
                                       const DClusterWeight max_cluster_weight)
      : _p_graph{graph},
        _num_clusters{num_clusters},
        _active(_p_graph->n()),
        _current_clustering(_p_graph->total_n()),
        _next_clustering(_p_graph->total_n()),
        _gains(_p_graph->n()),
        _cluster_weights(_p_graph->total_n()),
        _max_cluster_weight{max_cluster_weight} {
    init_clusters();
  }

  void perform_iteration() {
    DBG << "Performing local label propagation iteration with " << std::accumulate(_active.begin(), _active.end(), 0)
        << " active nodes";
    _p_graph->pfor_nodes([&](const DNodeID u) {
      auto &local_rand = shm::Randomize::instance();
      auto &local_map = _rating_map.local();
      handle_node(u, local_rand, local_map);
    });
    DBG << "Iteration completed; " << std::accumulate(_active.begin(), _active.end(), 0) << " actives nodes left";

    scalable_vector<shm::parallel::IntegralAtomicWrapper<DNodeWeight>> weight_to_block(_num_clusters);
    scalable_vector<shm::parallel::IntegralAtomicWrapper<DEdgeWeight>> gain_to_block(_num_clusters);

    _p_graph->pfor_nodes([&](const DNodeID u) {
      if (_current_clustering[u] == _next_clustering[u]) { return; }

      weight_to_block[_next_clustering[u]] += _p_graph->node_weight(u);
      gain_to_block[_next_clustering[u]] += _gains[u];
    });

    DLOG << V(weight_to_block) << V(gain_to_block);

    std::vector<DBlockWeight> residual_cluster_weights;
    std::vector<DEdgeWeight> global_total_gains_to_block;

    // gather statistics
    for (const DBlockID b : _p_graph->blocks()) {
      const DNodeWeight local_weight_to = weight_to_block[b];
      DBG << V(local_weight_to) << V(b);
      DNodeWeight global_weight_to = 0;
      MPI_Allreduce(&local_weight_to, &global_weight_to, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

      const DEdgeWeight local_gain_to = gain_to_block[b];
      DEdgeWeight global_gain_to = 0;
      MPI_Allreduce(&local_gain_to, &global_gain_to, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
      ASSERT(global_gain_to >= 0);

      residual_cluster_weights.push_back(_max_cluster_weight - _cluster_weights[b]);
      global_total_gains_to_block.push_back(global_gain_to);
    }

    // perform probabilistic moves
    perform_moves(residual_cluster_weights, global_total_gains_to_block);

    // and synchronize ghost node state
    synchronize_state();

    // check block weights after migration
    MPI_Barrier(MPI_COMM_WORLD);
    mpi::sequentially([&] {
      std::vector<DBlockWeight> tmp(_p_graph->k());
      for (const DNodeID u : _p_graph->nodes()) { tmp[_current_clustering[u]] += _p_graph->node_weight(u); }
      DLOG << V(tmp) << V(_max_cluster_weight);
    });

    /*
    // send join requests to adjacent PEs
    const auto [size, rank] = mpi::get_comm_info();

    std::vector<shm::parallel::IntegralAtomicWrapper<int>> num_messages(size);
    scalable_vector<shm::parallel::IntegralAtomicWrapper<DClusterWeight>> total_weight(_p_graph->ghost_n());
    scalable_vector<shm::parallel::IntegralAtomicWrapper<DEdgeWeight>> total_gain(_p_graph->ghost_n());

    // collect total weight and gain of nodes that want to join clusters owned by other PEs
    _p_graph->pfor_nodes([&](const DNodeID u) {
      const DClusterID c_u = _next_clustering[u];

      // join cluster owned by adjacent PE
      if (!_p_graph->is_owned_node(c_u)) {
        const DNodeID ghost_id = c_u - _p_graph->n(); // ghost ids start at zero

        if (total_weight[ghost_id] == 0) { ++num_messages[_p_graph->find_owner(c_u)]; }
        total_weight[ghost_id] += _p_graph->node_weight(u);
        total_gain[ghost_id] += _gains[u];
      }
    });

    DBG << "Number of messages for adjacent PEs: " << num_messages;

    // build messages
    std::vector<scalable_vector<int64_t>> messages(size);
    std::vector<shm::parallel::IntegralAtomicWrapper<std::size_t>> next_slot(size);
    for (PEID p = 0; p < size; ++p) {
      DBG << V(num_messages[p]);
      messages[p].resize(3 * num_messages[p]);
    }

    _p_graph->pfor_ghost_nodes([&](const DNodeID ghost_id) {
      if (total_weight[ghost_id] > 0) {
        const DNodeID ghost_u = ghost_id + _p_graph->n();
        const PEID owner = _p_graph->find_owner(ghost_u);

        const std::size_t slot = next_slot[owner] += 3;
        messages[owner][slot - 3] = _p_graph->global_node(ghost_u);
        messages[owner][slot - 2] = total_weight[ghost_id];
        messages[owner][slot - 1] = total_gain[ghost_id];
      }
    });

    std::vector<MPI_Request> messages_req(size);
    for (PEID p = 0; p < size; ++p) {
      if (p != rank) {
        MPI_Isend(messages[p].data(), 3 * num_messages[p], MPI_INT64_T, p, TAG_JOIN_MESSAGE, MPI_COMM_WORLD,
                  &messages_req[p]);
      }
    }

    for (PEID p = 0; p < size; ++p) {
      if (p != rank) {
        MPI_Status status;
        MPI_Probe(p, TAG_JOIN_MESSAGE, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, MPI_INT64_T, &count);
        ASSERT(count % 3 == 0) << "count should be a multiple of 3, is actually " << V(count);

        // receive messages from p
        std::vector<int64_t> recv_messages(count);
        MPI_Recv(recv_messages.data(), count, MPI_INT64_T, p, TAG_JOIN_MESSAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // aggregate per cluster
        for (std::size_t i = 0; i < count; i += 3) {
          const DNodeID global_u = recv_messages[i];
          const DNodeWeight total_weight = recv_messages[i + 1];
          const DEdgeWeight total_gain = recv_messages[i + 2];
        }
      }
    }

    MPI_Waitall(static_cast<int>(messages_req.size()), messages_req.data(), MPI_STATUS_IGNORE);

    DBG << "Done";
     */
  }

  scalable_vector<DClusterID> &&take_clustering() { return std::move(_current_clustering); }

private:
  void perform_moves(const std::vector<DBlockWeight> &residual_block_weights,
                     const std::vector<DEdgeWeight> &total_gains_to_block) {
    _p_graph->pfor_nodes([&](const DNodeID u) {
      if (_next_clustering[u] == _current_clustering[u]) { return; }

      const DBlockID b = _next_clustering[u];
      const double gain_prob = (total_gains_to_block[b] == 0) ? 1.0 : 1.0 * _gains[u] / total_gains_to_block[b];
      const double probability = gain_prob * (1.0 * residual_block_weights[b] / _p_graph->node_weight(u));

      auto &rand = shm::Randomize::instance();
      if (rand.random_bool(probability)) {
        //        DBG << "Move " << u << ": " << _current_clustering[u] << " --> " << _next_clustering[u]
        //            << " (with probability: " << probability << ", gain: " << _gains[u] << ")";
        _current_clustering[u] = _next_clustering[u];
        _p_graph->set_block(u, _current_clustering[u]);
      }
    });
  }

  void synchronize_state() {
    const auto [size, rank] = mpi::get_comm_info();
    tbb::enumerable_thread_specific<shm::Marker<>> created_message_for_adjacent_pe_ets{
        [&] { return shm::Marker<>(mpi::get_comm_size()); }};

    // message: global_id, new label, new cluster weight
    std::vector<shm::parallel::IntegralAtomicWrapper<std::size_t>> num_send_messages(size);
    std::vector<scalable_vector<int64_t>> send_messages;

    // find number of messages that we want to send to each PE
    _p_graph->pfor_nodes([&](const DNodeID u) {
      auto &created_message_for_adjacent_pe = created_message_for_adjacent_pe_ets.local();
      for (const auto [e, v] : _p_graph->neighbors(u)) {
        if (!_p_graph->is_owned_node(v)) {
          const int owner = _p_graph->ghost_owner(v);
          if (!created_message_for_adjacent_pe.get(owner)) {
            num_send_messages[owner] += 2;
            created_message_for_adjacent_pe.set(owner);
          }
        }
      }
      created_message_for_adjacent_pe.reset();
    });

    DLOG << V(num_send_messages);

    // allocate memory for messages
    for (PEID p = 0; p < size; ++p) { send_messages.emplace_back(num_send_messages[p]); }

    // now actually create the messages
    _p_graph->pfor_nodes([&](const DNodeID u) {
      auto &created_message_for_adjacent_pe = created_message_for_adjacent_pe_ets.local();
      for (const auto [e, v] : _p_graph->neighbors(u)) {
        if (!_p_graph->is_owned_node(v)) {
          const int owner = _p_graph->ghost_owner(v);
          if (!created_message_for_adjacent_pe.get(owner)) {
            // allocate memory in send_messages
            const std::size_t pos = num_send_messages[owner].fetch_sub(2);
            ASSERT(pos - 1 < send_messages[owner].size()) << V(owner) << V(pos) << V(send_messages[owner].size());
            send_messages[owner][pos - 2] = _p_graph->global_node(u);
            send_messages[owner][pos - 1] = _current_clustering[u];

            created_message_for_adjacent_pe.set(owner);
          }
        }
      }
      created_message_for_adjacent_pe.reset();
    });

    // exchange messages
    std::vector<MPI_Request> send_requests;
    send_requests.reserve(size);

    for (PEID p = 0; p < size; ++p) {
      if (p != rank) {
        send_requests.emplace_back();
        MPI_Isend(send_messages[p].data(), send_messages[p].size(), MPI_INT64_T, p, 0, MPI_COMM_WORLD,
                  &send_requests.back());
      }
    }

    for (PEID p = 0; p < size; ++p) {
      if (p != rank) {
        // get message count from PE p
        MPI_Status status{};
        MPI_Probe(p, 0, MPI_COMM_WORLD, &status);
        int count = 0;
        MPI_Get_count(&status, MPI_INT64_T, &count);

        scalable_vector<int64_t> recv_messages(count);
        MPI_Recv(recv_messages.data(), count, MPI_INT64_T, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        ASSERT(recv_messages.size() % 2 == 0); // we send pair of int64_t's

        // integrate the changes
        tbb::parallel_for(static_cast<std::size_t>(0), recv_messages.size() / 2, [&](const std::size_t i) {
          const DNodeID global_u = recv_messages[i * 2];
          const DNodeID local_u = _p_graph->local_node(global_u);
          const DClusterID c_global_u = recv_messages[i * 2 + 1];

          if (c_global_u != _current_clustering[local_u]) {
            _current_clustering[local_u] = c_global_u;
            _p_graph->set_block(local_u, c_global_u);
            //            DBG << "Moved " << local_u << " to " << _current_clustering[local_u];
          }
        });
      }
    }

    MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUS_IGNORE);
  }

  void init_clusters() {
    _p_graph->pfor_all_nodes([&](const DNodeID u) {
      ASSERT(u < _current_clustering.size());
      ASSERT(u < _next_clustering.size());
      _current_clustering[u] = _p_graph->block(u);
      _next_clustering[u] = _p_graph->block(u);
    });

    _p_graph->pfor_nodes([&](const DNodeID u) {
      ASSERT(u < _active.size());
      _active[u] = 1;
    });

    _p_graph->pfor_blocks([&](const DBlockID b) {
      ASSERT(b < _cluster_weights.size());
      _cluster_weights[b] = _p_graph->block_weight(b);
    });
  }

  std::pair<bool, bool> handle_node(const DNodeID u, shm::Randomize &local_rand, auto &local_rating_map) {
    const DNodeWeight u_weight = _p_graph->node_weight(u);
    const DClusterID u_cluster = _current_clustering[u];
    const auto [new_cluster, new_gain] = find_best_cluster(u, u_weight, u_cluster, local_rand, local_rating_map);

    bool success = false;
    bool emptied_cluster = false;

    if (u_cluster != new_cluster) {
      //      DClusterWeight new_weight = _cluster_weights[new_cluster];
      //      while (new_weight + u_weight <= _max_cluster_weight) {
      //        if (_cluster_weights[new_cluster].compare_exchange_weak(new_weight, new_weight + u_weight,
      //                                                                std::memory_order_relaxed)) {
      //          success = true;
      //          break;
      //        }
      //      }

      success = true;
      if (success) {
        //        _cluster_weights[u_cluster].fetch_sub(u_weight, std::memory_order_relaxed);
        _next_clustering[u] = new_cluster;
        _gains[u] = new_gain;
        activate_neighbors(u);
      }
    } else {
      _next_clustering[u] = u_cluster;
    }

    _active[u] = 0;
    return {success, emptied_cluster};
  }

  std::pair<DClusterID, DEdgeWeight> find_best_cluster(const DNodeID u, const DNodeWeight u_weight,
                                                       const DClusterID u_cluster, shm::Randomize &local_rand,
                                                       auto &local_rating_map) {
    auto action = [&](auto &map) {
      const DClusterWeight initial_cluster_weight = _cluster_weights[u_cluster];
      ClusterSelectionState state{
          .local_rand = local_rand,
          .u = u,
          .u_weight = u_weight,
          .initial_cluster = u_cluster,
          .initial_cluster_weight = initial_cluster_weight,
          .best_cluster = u_cluster,
          .best_gain = 0,
          .best_cluster_weight = initial_cluster_weight,
          .current_cluster = 0,
          .current_gain = 0,
          .current_cluster_weight = 0,
      };

      for (const auto [e, v] : _p_graph->neighbors(u)) {
        const DClusterID v_cluster = _current_clustering[v];
        const DEdgeWeight rating = _p_graph->edge_weight(e);
        map[v_cluster] += rating;
      }

      for (const auto [cluster, rating] : map.entries()) {
        state.current_cluster = cluster;
        state.current_gain = rating;
        state.current_cluster_weight = _cluster_weights[cluster];

        if (accept_cluster(state)) {
          state.best_cluster = state.current_cluster;
          state.best_cluster_weight = state.current_cluster_weight;
          state.best_gain = state.current_gain;
        }
      }

      map.clear();
      return std::make_pair(state.best_cluster, state.best_gain);
    };

    local_rating_map.update_upper_bound_size(_p_graph->degree(u));
    return local_rating_map.run_with_map(action, action);
  }

  void activate_neighbors(const DNodeID u) {
    for (const DNodeID v : _p_graph->adjacent_nodes(u)) {
      if (_p_graph->is_owned_node(v)) { _active[v].store(1); }
    }
  }

  [[nodiscard]] bool accept_cluster(const ClusterSelectionState &state) const {
    return (state.current_gain > state.best_gain ||
            (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
           (state.current_cluster_weight + state.u_weight < _max_cluster_weight ||
            state.current_cluster == state.initial_cluster);
  }

  DistributedPartitionedGraph *_p_graph;
  DClusterID _num_clusters;
  scalable_vector<shm::parallel::IntegralAtomicWrapper<uint8_t>> _active;
  scalable_vector<DClusterID> _current_clustering;
  scalable_vector<DClusterID> _next_clustering;
  scalable_vector<DEdgeWeight> _gains;
  scalable_vector<shm::parallel::IntegralAtomicWrapper<DClusterWeight>> _cluster_weights;
  tbb::enumerable_thread_specific<shm::RatingMap<DEdgeWeight>> _rating_map{
      [&] { return shm::RatingMap<DEdgeWeight>{_p_graph->total_n()}; }};
  DClusterWeight _max_cluster_weight;
};
} // namespace dkaminpar