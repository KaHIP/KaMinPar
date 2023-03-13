/*******************************************************************************
 * @file:   helper.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:
 ******************************************************************************/
#include "kaminpar/partitioning/helper.h"

#include "common/math.h"

namespace kaminpar::shm::partitioning::helper {
namespace {
SET_DEBUG(false);
SET_STATISTICS_FROM_GLOBAL();

bool should_balance(const BalancingTimepoint configured,
                    const BalancingTimepoint current) {
  return configured == current ||
         (configured == BalancingTimepoint::ALWAYS &&
          (current == BalancingTimepoint::BEFORE_KWAY_REFINEMENT ||
           current == BalancingTimepoint::AFTER_KWAY_REFINEMENT));
}

void balance(IBalancer *balancer, PartitionedGraph &p_graph,
             const BalancingTimepoint tp, const PartitionContext &p_ctx,
             const RefinementContext &r_ctx) {
  SCOPED_TIMER("Balancing");

  if (should_balance(r_ctx.balancer.timepoint, tp)) {
    LOG << "-> Balance graph with n=" << p_graph.n() << " m=" << p_graph.m()
        << " k=" << p_graph.k();
    const EdgeWeight cut_before = IFDBG(metrics::edge_cut(p_graph));
    const double imbalance_before = IFDBG(metrics::imbalance(p_graph));
    const bool feasible_before = IFDBG(metrics::is_feasible(p_graph, p_ctx));

    balancer->initialize(p_graph);
    balancer->balance(p_graph, p_ctx);

    DBG << "-> cut=" << C(cut_before, metrics::edge_cut(p_graph))
        << "imbalance=" << C(imbalance_before, metrics::imbalance(p_graph))
        << "feasible="
        << C(feasible_before, metrics::is_feasible(p_graph, p_ctx));
  }
}
} // namespace

void update_partition_context(PartitionContext &current_p_ctx,
                              const PartitionedGraph &p_graph) {
  current_p_ctx.setup(p_graph.graph());
  current_p_ctx.k = p_graph.k();
  current_p_ctx.block_weights.setup(current_p_ctx, p_graph.final_ks());
}

PartitionedGraph uncoarsen_once(ICoarsener *coarsener, PartitionedGraph p_graph,
                                PartitionContext &current_p_ctx) {
  SCOPED_TIMER("Uncoarsening");

  if (!coarsener->empty()) {
    const NodeID n_before = p_graph.n();
    const EdgeID m_before = p_graph.m();
    p_graph = coarsener->uncoarsen(std::move(p_graph));

    LOG << "-> Uncoarsen graph: "           //
        << "n=" << C(n_before, p_graph.n()) //
        << "m=" << C(m_before, p_graph.m()) //
        << "k=" << p_graph.k();             //

    update_partition_context(current_p_ctx, p_graph);
  }

  return p_graph;
}

void refine(IRefiner *refiner, IBalancer *balancer, PartitionedGraph &p_graph,
            const PartitionContext &current_p_ctx,
            const RefinementContext &r_ctx) {
  SCOPED_TIMER("Refinement");

  balance(balancer, p_graph, BalancingTimepoint::BEFORE_KWAY_REFINEMENT,
          current_p_ctx, r_ctx);

  LOG << "-> Refine graph with n=" << p_graph.n() << " m=" << p_graph.m()
      << " k=" << p_graph.k();
  const EdgeWeight cut_before = IFSTATS(metrics::edge_cut(p_graph));
  refiner->initialize(p_graph.graph());
  refiner->refine(p_graph, current_p_ctx);
  const EdgeWeight cut_after = IFSTATS(metrics::edge_cut(p_graph));

  STATS << "Edge cut changed by refinement: " << C(cut_before, cut_after)
        << "= reduced by " << cut_before - cut_after
        << "; expected cut reduction: " << refiner->expected_total_gain();

  balance(balancer, p_graph, BalancingTimepoint::AFTER_KWAY_REFINEMENT,
          current_p_ctx, r_ctx);
}

PartitionedGraph
bipartition(const Graph *graph, const BlockID final_k, const Context &input_ctx,
            GlobalInitialPartitionerMemoryPool &ip_m_ctx_pool) {
  ip::InitialPartitioner partitioner{*graph, input_ctx, final_k,
                                     ip_m_ctx_pool.local().get()};
  PartitionedGraph p_graph = partitioner.partition();
  ip_m_ctx_pool.local().put(partitioner.free());
  //  DBG << "Bipartition result: " << V(p_graph.final_ks()) <<
  //  V(p_graph.block_weights());
  return p_graph;
}

void extend_partition_recursive(
    const Graph &graph, BlockArray &partition, const BlockID b0,
    const BlockID k, const BlockID final_k, const Context &input_ctx,
    graph::SubgraphMemory &subgraph_memory,
    const graph::SubgraphMemoryStartPosition position,
    TemporaryGraphExtractionBufferPool &extraction_pool,
    GlobalInitialPartitionerMemoryPool &ip_m_ctx_pool) {
  KASSERT(k > 1u);

  // obtain bipartition of current graph
  PartitionedGraph p_graph =
      bipartition(&graph, final_k, input_ctx, ip_m_ctx_pool);

  const BlockID final_k1 = p_graph.final_k(0);
  const BlockID final_k2 = p_graph.final_k(1);
  KASSERT(final_k1 > 0u);
  KASSERT(final_k2 > 0u);
  KASSERT(final_k == final_k1 + final_k2);

  std::array<BlockID, 2> ks{0, 0};
  std::tie(ks[0], ks[1]) = math::split_integral(k);
  KASSERT(ks[0] + ks[1] == k);
  KASSERT(ks[0] >= 1u);
  KASSERT(ks[1] >= 1u);
  KASSERT(final_k1 >= ks[0]);
  KASSERT(final_k2 >= ks[1]);

  // copy p_graph to partition -> replace b0 with b0 or b1
  std::array<BlockID, 2> b{b0, b0 + ks[0]};
  KASSERT(b[0] < input_ctx.partition.k);
  KASSERT(b[1] < input_ctx.partition.k);
  NodeID current_node = 0;
  for (std::size_t i = 0; i < partition.size(); ++i) {
    if (partition[i] == b0) {
      partition[i] = b[p_graph.block(current_node++)];
    }
  }
  KASSERT(current_node == p_graph.n());

  if (k > 2) {
    auto extraction = extract_subgraphs_sequential(
        p_graph, position, subgraph_memory, extraction_pool.local());
    const auto &subgraphs = extraction.subgraphs;
    const auto &positions = extraction.positions;

    for (const std::size_t i : {0, 1}) {
      if (ks[i] > 1) {
        extend_partition_recursive(
            subgraphs[i], partition, b[i], ks[i], p_graph.final_k(i), input_ctx,
            subgraph_memory, positions[i], extraction_pool, ip_m_ctx_pool);
      }
    }
  }
}

void extend_partition(PartitionedGraph &p_graph, const BlockID k_prime,
                      const Context &input_ctx, PartitionContext &current_p_ctx,
                      graph::SubgraphMemory &subgraph_memory,
                      TemporaryGraphExtractionBufferPool &extraction_pool,
                      GlobalInitialPartitionerMemoryPool &ip_m_ctx_pool) {
  SCOPED_TIMER("Initial partitioning");

  LOG << "-> Extend from=" << p_graph.k() << " "    //
      << "to=" << k_prime << " "                    //
      << "on a graph with n=" << p_graph.n() << " " //
      << "m=" << p_graph.m();                       //

  auto extraction = TIMED_SCOPE("Extract subgraphs") {
    return extract_subgraphs(p_graph, subgraph_memory);
  };
  const auto &subgraphs = extraction.subgraphs;
  const auto &mapping = extraction.node_mapping;
  const auto &positions = extraction.positions;

  START_TIMER("Allocation");
  scalable_vector<BlockArray> subgraph_partitions;
  for (const auto &subgraph : subgraphs) {
    subgraph_partitions.emplace_back(subgraph.n());
  }
  STOP_TIMER();

  START_TIMER("Bipartitioning");
  tbb::parallel_for(
      static_cast<BlockID>(0), static_cast<BlockID>(subgraphs.size()),
      [&](const BlockID b) {
        const auto &subgraph = subgraphs[b];
        const BlockID subgraph_k = (k_prime == input_ctx.partition.k)
                                       ? p_graph.final_k(b)
                                       : k_prime / p_graph.k();
        if (subgraph_k > 1) {
          KASSERT(subgraph_k <= p_graph.final_k(b));
          extend_partition_recursive(subgraph, subgraph_partitions[b], 0,
                                     subgraph_k, p_graph.final_k(b), input_ctx,
                                     subgraph_memory, positions[b],
                                     extraction_pool, ip_m_ctx_pool);
        }
      });
  STOP_TIMER();

  TIMED_SCOPE("Copy subgraph partitions") {
    graph::copy_subgraph_partitions(p_graph, subgraph_partitions, k_prime,
                                    input_ctx.partition.k, mapping);
  };
  update_partition_context(current_p_ctx, p_graph);

  KASSERT(p_graph.k() == k_prime);
}

// extend_partition with local memory allocation for subgraphs
void extend_partition(PartitionedGraph &p_graph, const BlockID k_prime,
                      const Context &input_ctx, PartitionContext &current_p_ctx,
                      TemporaryGraphExtractionBufferPool &extraction_pool,
                      GlobalInitialPartitionerMemoryPool &ip_m_ctx_pool) {
  START_TIMER("Allocation");
  graph::SubgraphMemory memory{p_graph.n(), input_ctx.partition.k, p_graph.m(),
                               p_graph.graph().is_node_weighted(),
                               p_graph.graph().is_edge_weighted()};
  STOP_TIMER();
  extend_partition(p_graph, k_prime, input_ctx, current_p_ctx, memory,
                   extraction_pool, ip_m_ctx_pool);
}

bool coarsen_once(ICoarsener *coarsener, const Graph *graph,
                  const Context &input_ctx, PartitionContext &current_p_ctx) {
  SCOPED_TIMER("Coarsening");

  const NodeWeight max_cluster_weight = compute_max_cluster_weight(
      *graph, input_ctx.partition, input_ctx.coarsening);
  const auto [c_graph, shrunk] =
      coarsener->compute_coarse_graph(max_cluster_weight, 0);

  LOG << "-> "                                              //
      << "n=" << c_graph->n() << " "                        //
      << "m=" << c_graph->m() << " "                        //
      << "max_cluster_weight=" << max_cluster_weight << " " //
      << ((shrunk) ? "" : "==> converged");                 //

  if (shrunk) {
    current_p_ctx.setup(*c_graph);
  } // update graph stats (max node weight)
  return shrunk;
}

BlockID compute_k_for_n(const NodeID n, const Context &input_ctx) {
  if (n < 2 * input_ctx.coarsening.contraction_limit) {
    return 2;
  } // catch special case where log is negative
  const BlockID k_prime =
      1 << math::ceil_log2(n / input_ctx.coarsening.contraction_limit);
  return std::clamp(k_prime, static_cast<BlockID>(2), input_ctx.partition.k);
}

std::size_t compute_num_copies(const Context &input_ctx, const NodeID n,
                               const bool converged,
                               const std::size_t num_threads) {
  KASSERT(num_threads > 0u);

  // sequential base case?
  const NodeID C = input_ctx.coarsening.contraction_limit;
  if (converged || n <= 2 * C) {
    return num_threads;
  }

  // parallel case
  const std::size_t f =
      1 << static_cast<std::size_t>(std::ceil(std::log2(1.0 * n / C)));

  // continue with coarsening if the graph is still too large
  if (f > num_threads) {
    return 1;
  }

  // split into groups
  return num_threads / f;
}

std::size_t select_best(const scalable_vector<PartitionedGraph> &p_graphs,
                        const PartitionContext &p_ctx) {
  return select_best(p_graphs.begin(), p_graphs.end(), p_ctx);
}

std::size_t compute_num_threads_for_parallel_ip(const Context &input_ctx) {
  return math::floor2(input_ctx.parallel.num_threads) *
         (1 << input_ctx.initial_partitioning.multiplier_exponent);
}
} // namespace kaminpar::shm::partitioning::helper
