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
#include "dkaminpar/mpi_utils.h"
#include "dkaminpar/utility/distributed_math.h"
#include "dkaminpar/utility/vector_ets.h"
#include "kaminpar/datastructure/marker.h"
#include "kaminpar/datastructure/rating_map.h"
#include "kaminpar/utility/random.h"

#include <tbb/enumerable_thread_specific.h>

namespace dkaminpar {
template<typename DClusterID = DNodeID, typename DClusterWeight = DNodeWeight>
class DistributedLabelPropagationRefiner {
  SET_DEBUG(true);

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
  explicit DistributedLabelPropagationRefiner(const LabelPropagationRefinementContext &lp_ctx,
                                              DistributedPartitionedGraph *p_graph, const DClusterID num_clusters,
                                              const DClusterWeight max_cluster_weight)
      : _p_graph{p_graph},
        _num_clusters{num_clusters},
        _active(_p_graph->n()),
        _current_clustering(_p_graph->total_n()),
        _next_clustering(_p_graph->total_n()),
        _gains(_p_graph->n()),
        _cluster_weights(_p_graph->total_n()),
        _cluster_weights_tmp(_p_graph->total_n()),
        _max_cluster_weight{max_cluster_weight},
        _lp_ctx{lp_ctx} {
    init_clusters();
  }

  void perform_iteration() {
    for (std::size_t i = 0; i < _lp_ctx.num_chunks; ++i) {
      const auto [from, to] = math::compute_local_range<DNodeID>(_p_graph->n(), _lp_ctx.num_chunks, i);
      perform_iteration(from, to);
    }
  }

  scalable_vector<DClusterID> &&take_clustering() { return std::move(_current_clustering); }

private:
  void perform_iteration(const DNodeID from, const DNodeID to) {
    MPI_Barrier(MPI_COMM_WORLD);

    DBG << "Performing local label propagation iteration with " << std::accumulate(_active.begin(), _active.end(), 0)
        << " active nodes, from=" << from << " to=" << to << " count=" << to - from;
    _p_graph->pfor_nodes(from, to, [&](const DNodeID u) {
      auto &local_rand = shm::Randomize::instance();
      auto &local_map = _rating_map.local();
      handle_node(u, local_rand, local_map);
    });
    DBG << "Iteration completed; " << std::accumulate(_active.begin(), _active.end(), 0) << " actives nodes left";

    //    const DEdgeWeight local_total_gain = shm::parallel::accumulate(_gains);
    //    DEdgeWeight global_total_gain = 0;
    //    MPI_Allreduce(&local_total_gain, &global_total_gain, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    //    DLOG << V(global_total_gain) << V(local_total_gain);

    // accumulate total weight of nodes moved to each block
    parallel::vector_ets<DNodeWeight> weight_to_block_ets(_num_clusters);
    parallel::vector_ets<DEdgeWeight> gain_to_block_ets(_num_clusters);

    _p_graph->pfor_nodes_range(from, to, [&](const auto r) {
      auto &weight_to_block = weight_to_block_ets.local();
      auto &gain_to_block = gain_to_block_ets.local();

      for (DNodeID u = r.begin(); u < r.end(); ++u) {
        if (_current_clustering[u] != _next_clustering[u]) {
          weight_to_block[_next_clustering[u]] += _p_graph->node_weight(u);
          gain_to_block[_next_clustering[u]] += _gains[u];
        }
      }
    });

    const auto weight_to_block = weight_to_block_ets.combine(std::plus{});
    const auto gain_to_block = gain_to_block_ets.combine(std::plus{});

    // allreduce gain to block

    std::vector<DBlockWeight> residual_cluster_weights;
    std::vector<DEdgeWeight> global_total_gains_to_block;

    // gather statistics
    for (const DBlockID b : _p_graph->blocks()) {
      const DEdgeWeight local_gain_to = gain_to_block[b];
      DEdgeWeight global_gain_to = 0;
      MPI_Allreduce(&local_gain_to, &global_gain_to, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
      ASSERT(global_gain_to >= 0);

      residual_cluster_weights.push_back(_max_cluster_weight - _cluster_weights[b]);
      global_total_gains_to_block.push_back(global_gain_to);
    }

    // perform probabilistic moves
    for (std::size_t i = 0; i < _lp_ctx.num_move_attempts; ++i) {
      if (perform_moves(from, to, residual_cluster_weights, global_total_gains_to_block)) {
        synchronize_state(from, to);
        break;
      }
    }
  }

  bool perform_moves(const DNodeID from, const DNodeID to, const std::vector<DBlockWeight> &residual_block_weights,
                     const std::vector<DEdgeWeight> &total_gains_to_block) {
    MPI_Barrier(MPI_COMM_WORLD);

    struct Move {
      DNodeID u;
      DBlockID from;
      DBlockID to;
    };

    std::vector<Move> moves;

    _p_graph->pfor_nodes(from, to, [&](const DNodeID u) {
      if (_next_clustering[u] == _current_clustering[u]) { return; }

      const DBlockID b = _next_clustering[u];
      const double gain_prob = (total_gains_to_block[b] == 0) ? 1.0 : 1.0 * _gains[u] / total_gains_to_block[b];
      const double probability = gain_prob * (1.0 * residual_block_weights[b] / _p_graph->node_weight(u));

      auto &rand = shm::Randomize::instance();
      if (rand.random_bool(probability)) {
        moves.emplace_back(u, _current_clustering[u], _next_clustering[u]);
        _current_clustering[u] = _next_clustering[u];
        _p_graph->set_block(u, _current_clustering[u]);
      }
    });

    // get global block weights
    std::vector<DBlockWeight> local_block_weights(_p_graph->k());
    for (const DNodeID u : _p_graph->nodes()) { // TODO parallel::accumulate
      local_block_weights[_current_clustering[u]] += _p_graph->node_weight(u);
    }
    std::vector<DBlockWeight> global_block_weights(_p_graph->k());
    for (const DBlockID b : _p_graph->blocks()) {
      MPI_Allreduce(&local_block_weights[b], &global_block_weights[b], 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    }

    // check for balance violations
    shm::parallel::IntegralAtomicWrapper<std::uint8_t> feasible = 1;
    _p_graph->pfor_blocks([&](const DBlockID b) {
      if (global_block_weights[b] > _max_cluster_weight) { feasible = 0; }
    });

    if (feasible) { // apply new weights
      _p_graph->pfor_blocks([&](const DBlockID b) {
        _p_graph->set_block_weight(b, global_block_weights[b]);
        _cluster_weights[b] = global_block_weights[b];
        _cluster_weights_tmp[b] = global_block_weights[b];
      });
    } else { // discard moves
      for (const auto &move : moves) {
        _current_clustering[move.u] = move.from;
        _p_graph->set_block(move.u, move.from);
      }
    }

    return feasible;
  }

  void synchronize_state(const DNodeID from, const DNodeID to) {
    UNUSED(from);
    UNUSED(to);

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
      _cluster_weights_tmp[b] = _p_graph->block_weight(b);
    });
  }

  std::pair<bool, bool> handle_node(const DNodeID u, shm::Randomize &local_rand, auto &local_rating_map) {
    const DNodeWeight u_weight = _p_graph->node_weight(u);
    const DClusterID u_cluster = _current_clustering[u];
    const auto [new_cluster, new_gain] = find_best_cluster(u, u_weight, u_cluster, local_rand, local_rating_map);

    bool success = false;
    bool emptied_cluster = false;

    if (u_cluster != new_cluster) {
      _next_clustering[u] = new_cluster;
      _gains[u] = new_gain;
      activate_neighbors(u);
    } else {
      _next_clustering[u] = u_cluster;
      _gains[u] = 0;
    }

    _active[u] = 0;
    return {success, emptied_cluster};
  }

  std::pair<DClusterID, DEdgeWeight> find_best_cluster(const DNodeID u, const DNodeWeight u_weight,
                                                       const DClusterID u_cluster, shm::Randomize &local_rand,
                                                       auto &local_rating_map) {
    auto action = [&](auto &map) {
      const DClusterWeight initial_cluster_weight = _cluster_weights_tmp[u_cluster];
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
        state.current_cluster_weight = _cluster_weights_tmp[cluster];

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
  scalable_vector<shm::parallel::IntegralAtomicWrapper<DClusterWeight>> _cluster_weights_tmp;
  tbb::enumerable_thread_specific<shm::RatingMap<DEdgeWeight>> _rating_map{
      [&] { return shm::RatingMap<DEdgeWeight>{_p_graph->total_n()}; }};
  DClusterWeight _max_cluster_weight;
  const LabelPropagationRefinementContext &_lp_ctx;
};
} // namespace dkaminpar