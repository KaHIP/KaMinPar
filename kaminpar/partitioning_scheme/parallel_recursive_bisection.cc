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
#include "partitioning_scheme/parallel_recursive_bisection.h"

#include "partitioning_scheme/parallel_initial_partitioner.h"
#include "partitioning_scheme/parallel_synchronized_initial_partitioner.h"

namespace kaminpar::partitioning {
ParallelRecursiveBisection::ParallelRecursiveBisection(const Graph &input_graph, const Context &input_ctx)
    : _input_graph{input_graph},
      _input_ctx{input_ctx},
      _current_p_ctx{input_ctx.partition}, //
      _coarsener{factory::create_coarsener(input_graph, input_ctx.coarsening)},
      _refiner{factory::create_refiner(input_graph, input_ctx.partition, input_ctx.refinement)},
      _balancer{factory::create_balancer(input_graph, input_ctx.partition, input_ctx.refinement)},
      _subgraph_memory{input_graph.n(), input_ctx.partition.k, input_graph.m(), true, true} {}

PartitionedGraph ParallelRecursiveBisection::partition() {
  cio::print_banner("Coarsening");
  const Graph *c_graph = coarsen();

  cio::print_banner("Initial partitioning");
  PartitionedGraph p_graph = initial_partition(c_graph);

  cio::print_banner("Uncoarsening");
  bool refined;
  p_graph = uncoarsen(std::move(p_graph), refined);
  if (!refined) { refine(p_graph); }
  if (p_graph.k() < _input_ctx.partition.k) {
    extend_partition(p_graph, _input_ctx.partition.k);
    refine(p_graph);
  }

  if constexpr (kStatistics) { print_statistics(); }

  return p_graph;
}

PartitionedGraph ParallelRecursiveBisection::uncoarsen(PartitionedGraph p_graph, bool &refined) {
  while (!_coarsener->empty()) {
    p_graph = uncoarsen_once(std::move(p_graph));
    refine(p_graph);
    refined = true;

    const BlockID desired_k = helper::compute_k_for_n(p_graph.n(), _input_ctx);
    if (p_graph.k() < desired_k) {
      extend_partition(p_graph, desired_k);
      refined = false;
    }
  }

  return p_graph;
}

const Graph *ParallelRecursiveBisection::coarsen() {
  const Graph *c_graph = &_input_graph;
  bool shrunk = true;

  while (shrunk && c_graph->n() > initial_partition_threshold()) {
    shrunk = helper::coarsen_once(_coarsener.get(), c_graph, _input_ctx, _current_p_ctx);
    c_graph = _coarsener->coarsest_graph();
  }

  return c_graph;
}

NodeID ParallelRecursiveBisection::initial_partition_threshold() {
  if (_input_ctx.initial_partitioning.parallelize) {
    return _input_ctx.parallel.num_threads * _input_ctx.coarsening.contraction_limit; // p * C
  } else {
    return 2 * _input_ctx.coarsening.contraction_limit; // 2 * C
  }
}

PartitionedGraph ParallelRecursiveBisection::initial_partition(const Graph *graph) {
  SCOPED_TIMER(TIMER_INITIAL_PARTITIONING_SCHEME);
  PartitionedGraph p_graph = _input_ctx.initial_partitioning.parallelize //
                                 ? parallel_initial_partition(graph)     //
                                 : sequential_initial_partition(graph);  //

  helper::update_partition_context(_current_p_ctx, p_graph);

  LOG << "-> Initial partition on a graph with n=" << graph->n() << " "       //
      << "m=" << graph->m()                                                   //
      << ": k=" << p_graph.k();                                               //
  DBG << "-> "                                                                //
      << "cut=" << metrics::edge_cut(p_graph) << " "                          //
      << "imbalance=" << metrics::imbalance(p_graph) << " "                   //
      << "feasible=" << metrics::is_feasible(p_graph, _current_p_ctx) << " "; //

  return p_graph;
}

PartitionedGraph ParallelRecursiveBisection::parallel_initial_partition(const Graph * /* use _coarsener */) {
  LOG << "Performing parallel initial partitioning";

  // Timers are tricky during parallel initial partitioning
  // Hence, we only record its total time
  DISABLE_TIMERS();
  PartitionedGraph p_graph = [&] {
    if (_input_ctx.initial_partitioning.parallelize_synchronized) {
      ParallelSynchronizedInitialPartitioner initial_partitioner{_input_ctx, _ip_m_ctx_pool, _ip_extraction_pool};
      return initial_partitioner.partition(_coarsener.get(), _current_p_ctx);
    } else {
      ParallelInitialPartitioner initial_partitioner{_input_ctx, _ip_m_ctx_pool, _ip_extraction_pool};
      return initial_partitioner.partition(_coarsener.get(), _current_p_ctx);
    }
  }();
  ENABLE_TIMERS();

  return p_graph;
}

PartitionedGraph ParallelRecursiveBisection::sequential_initial_partition(const Graph *graph) {
  LOG << "Performing sequential initial partitioning";

  // Timers work fine with sequential initial partitioning, but we disable them to get a output similar to the
  // parallel case
  DISABLE_TIMERS();
  PartitionedGraph p_graph = helper::bipartition(graph, _input_ctx.partition.k, _input_ctx, _ip_m_ctx_pool);
  ENABLE_TIMERS();

  return p_graph;
}

void ParallelRecursiveBisection::print_statistics() {
  std::size_t num_ip_m_ctx_objects = 0;
  std::size_t max_ip_m_ctx_objects = 0;
  std::size_t min_ip_m_ctx_objects = std::numeric_limits<std::size_t>::max();
  std::size_t ip_m_ctx_memory_in_kb = 0;
  for (const auto &pool : _ip_m_ctx_pool) {
    num_ip_m_ctx_objects += pool.pool.size();
    max_ip_m_ctx_objects = std::max(max_ip_m_ctx_objects, pool.pool.size());
    min_ip_m_ctx_objects = std::min(min_ip_m_ctx_objects, pool.pool.size());
    ip_m_ctx_memory_in_kb += pool.memory_in_kb();
  }

  LOG << logger::CYAN << "Initial partitioning: Memory pool";
  LOG << logger::CYAN << " * # of pool objects: " << min_ip_m_ctx_objects
      << " <= " << 1.0 * num_ip_m_ctx_objects / _input_ctx.parallel.num_threads << " <= " << max_ip_m_ctx_objects;
  LOG << logger::CYAN << " * total memory: " << ip_m_ctx_memory_in_kb / 1000 << " Mb";

  std::size_t extraction_nodes_reallocs = 0;
  std::size_t extraction_edges_reallocs = 0;
  std::size_t extraction_memory_in_kb = 0;
  for (const auto &buffer : _ip_extraction_pool) {
    extraction_nodes_reallocs += buffer.num_node_reallocs;
    extraction_edges_reallocs += buffer.num_edge_reallocs;
    extraction_memory_in_kb += buffer.memory_in_kb();
  }

  LOG << logger::CYAN << "Extraction buffer pool:";
  LOG << logger::CYAN << " * # of node buffer reallocs: " << extraction_nodes_reallocs
      << ", # of edge buffer reallocs: " << extraction_edges_reallocs;
  LOG << logger::CYAN << " * total memory: " << extraction_memory_in_kb / 1000 << " Mb";
}
} // namespace kaminpar::partitioning
