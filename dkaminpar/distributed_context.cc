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

namespace dkaminpar {
using namespace std::string_literals;

DEFINE_ENUM_STRING_CONVERSION(PartitioningMode, partitioning_mode) = {{PartitioningMode::KWAY, "kway"},
                                                                      {PartitioningMode::DEEP, "deep"},
                                                                      {PartitioningMode::RB, "rb"}};

void DLabelPropagationCoarseningContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "num_iterations=" << num_iterations << " "                                              //
      << prefix << "large_degree_threshold=" << large_degree_threshold << " "                              //
      << prefix << "max_num_neighbors=" << max_num_neighbors << " "                                        //
      << prefix << "merge_singleton_clusters=" << merge_singleton_clusters << " "                          //
      << prefix << "merge_nonadjacent_clusters_threshold=" << merge_nonadjacent_clusters_threshold << " "; //
}

void DLabelPropagationRefinementContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "num_iterations=" << num_iterations << " "        //
      << prefix << "num_chunks=" << num_chunks << " "                //
      << prefix << "num_move_attempts=" << num_move_attempts << " "; //
}

void DCoarseningContext::print(std::ostream &out, const std::string &prefix) const { lp.print(out, prefix + "lp."); }

void DInitialPartitioning::print(std::ostream &out, const std::string &prefix) const {
  sequential.print(out, prefix + "sequential.");
}

void DRefinementContext::print(std::ostream &out, const std::string &prefix) const { lp.print(out, prefix + "lp."); }

void DParallelContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "num_threads=" << num_threads << " "                                          //
      << prefix << "use_interleaved_numa_allocation=" << use_interleaved_numa_allocation << " "; //
}

void DPartitionContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "k=" << k << " "             //
      << prefix << "epsilon=" << epsilon << " " //
      << prefix << "mode=" << mode << " ";      //
}

void DContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "graph_filename=" << graph_filename << " " //
      << prefix << "seed=" << seed << " "                     //
      << prefix << "quiet=" << quiet << " ";                  //
  partition.print(out, prefix + "partition.");
  parallel.print(out, prefix + "parallel.");
  coarsening.print(out, prefix + "coarsening.");
  initial_partitioning.print(out, prefix + "initial_partitioning.");
  refinement.print(out, prefix + "refinement.");
}

std::ostream &operator<<(std::ostream &out, const DContext &context) {
  context.print(out);
  return out;
}

DContext create_default_context() {
  // clang-format off
  return {
    .graph_filename = "",
    .seed = 0,
    .quiet = false,
    .partition = {
      .k = 0,
      .epsilon = 0.03,
      .mode = PartitioningMode::KWAY,
    },
    .parallel = {
      .num_threads = 1,
      .use_interleaved_numa_allocation = true,
    },
    .coarsening = {
      .lp = {
        .num_iterations = 5,
        .large_degree_threshold = 1000000,
        .max_num_neighbors = std::numeric_limits<DNodeID>::max(),
        .merge_singleton_clusters = true,
        .merge_nonadjacent_clusters_threshold = 0.5,
      }
    },
    .initial_partitioning = {
      .sequential = shm::create_default_context(),
    },
    .refinement = {
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