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
#include "dkaminpar/distributed_context.h"

#include "mpi_wrapper.h"

#include <tbb/parallel_for.h>

namespace dkaminpar {
using namespace std::string_literals;

DEFINE_ENUM_STRING_CONVERSION(PartitioningMode, partitioning_mode) = {
    {PartitioningMode::KWAY, "kway"}, //
    {PartitioningMode::DEEP, "deep"}, //
    {PartitioningMode::RB, "rb"}      //
};

DEFINE_ENUM_STRING_CONVERSION(CoarseningAlgorithm, coarsening_algorithm) = {
    {CoarseningAlgorithm::NOOP, "noop"},        //
    {CoarseningAlgorithm::LOCAL_LP, "local-lp"} //
};

DEFINE_ENUM_STRING_CONVERSION(InitialPartitioningAlgorithm, initial_partitioning_algorithm) = {
    {InitialPartitioningAlgorithm::KAMINPAR, "kaminpar"} //
};

DEFINE_ENUM_STRING_CONVERSION(KWayRefinementAlgorithm, kway_refinement_algorithm) = {
    {KWayRefinementAlgorithm::NOOP, "noop"}, //
    {KWayRefinementAlgorithm::LP, "lp"}      //
};

void LabelPropagationCoarseningContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "num_iterations=" << num_iterations << " "                                             //
      << prefix << "max_degree=" << large_degree_threshold << " "                                         //
      << prefix << "max_num_neighbors=" << max_num_neighbors << " "                                       //
      << prefix << "merge_singleton_clusters=" << merge_singleton_clusters << " "                         //
      << prefix << "merge_nonadjacent_clusters_threshold=" << merge_nonadjacent_clusters_threshold << " " //
      << prefix << "num_chunks=" << num_chunks << " ";                                                    //
}

void LabelPropagationRefinementContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "num_iterations=" << num_iterations << " "        //
      << prefix << "num_chunks=" << num_chunks << " "                //
      << prefix << "num_move_attempts=" << num_move_attempts << " "; //
}

void CoarseningContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "algorithm=" << algorithm << " "        //
      << "contraction_limit=" << contraction_limit << " "; //
  lp.print(out, prefix + "lp.");
}

void InitialPartitioningContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "algorithm=" << algorithm << " ";
  sequential.print(out, prefix + "sequential.");
}

void RefinementContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "algorithm=" << algorithm << " ";
  lp.print(out, prefix + "lp.");
}

void ParallelContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "num_threads=" << num_threads << " "                                         //
      << prefix << "use_interleaved_numa_allocation=" << use_interleaved_numa_allocation << " " //
      << prefix << "mpi_thread_support=" << mpi_thread_support << " ";                          //
}

void PartitionContext::setup(const DistributedGraph &graph) {
  _global_n = graph.global_n();
  _global_m = graph.global_m();
  _global_total_node_weight = mpi::allreduce(graph.total_node_weight(), MPI_SUM, graph.communicator());
  _local_n = graph.n();
  _local_m = graph.m();
  _total_node_weight = graph.total_node_weight();

  setup_perfectly_balanced_block_weights();
  setup_max_block_weights();
}

void PartitionContext::setup_perfectly_balanced_block_weights() {
  _perfectly_balanced_block_weights.resize(k);

  const BlockWeight perfectly_balanced_block_weight = std::ceil(static_cast<double>(global_total_node_weight()) / k);
  tbb::parallel_for<BlockID>(0, k, [&](const BlockID b) {
    _perfectly_balanced_block_weights[b] = perfectly_balanced_block_weight;
  });
}

void PartitionContext::setup_max_block_weights() {
  _max_block_weights.resize(k);

  tbb::parallel_for<BlockID>(0, k, [&](const BlockID b) {
    _max_block_weights[b] = static_cast<BlockWeight>((1.0 + epsilon) *
                                                     static_cast<double>(perfectly_balanced_block_weight(b)));
  });
}

void PartitionContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "k=" << k << " "             //
      << prefix << "epsilon=" << epsilon << " " //
      << prefix << "mode=" << mode << " ";      //
}

void Context::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "graph_filename=" << graph_filename << " " //
      << prefix << "seed=" << seed << " "                     //
      << prefix << "quiet=" << quiet << " ";                  //
  partition.print(out, prefix + "partition.");
  parallel.print(out, prefix + "parallel.");
  coarsening.print(out, prefix + "coarsening.");
  initial_partitioning.print(out, prefix + "initial_partitioning.");
  refinement.print(out, prefix + "refinement.");
}

std::ostream &operator<<(std::ostream &out, const Context &context) {
  context.print(out);
  return out;
}

Context create_default_context() {
  // clang-format off
  return {
    .graph_filename = "",
    .seed = 0,
    .quiet = false,
    .partition = {
      /* .k = */ 0,
      /* .epsilon = */ 0.03,
      /* .mode = */ PartitioningMode::KWAY,
    },
    .parallel = {
      .num_threads = 1,
      .use_interleaved_numa_allocation = true,
      .mpi_thread_support = MPI_THREAD_FUNNELED,
    },
    .coarsening = {
      .algorithm = CoarseningAlgorithm::LOCAL_LP,
      .contraction_limit = 5000,
      .lp = {
        .num_iterations = 5,
        .large_degree_threshold = 1000000,
        .max_num_neighbors = kInvalidNodeID,
        .merge_singleton_clusters = true,
        .merge_nonadjacent_clusters_threshold = 0.5,
        .num_chunks = 8,
      }
    },
    .initial_partitioning = {
      .algorithm = InitialPartitioningAlgorithm::KAMINPAR,
      .sequential = shm::create_default_context(),
    },
    .refinement = {
      .algorithm = KWayRefinementAlgorithm::LP,
      .lp = {
        .num_iterations = 5,
        .num_chunks = 8,
        .num_move_attempts = 2,
      }
    }
  };
  // clang-format on
}
} // namespace dkaminpar