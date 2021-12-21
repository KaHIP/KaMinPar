/*******************************************************************************
 * @file:   dkaminpar_graphgen.h
 *
 * @author: Daniel Seemaier
 * @date:   26.11.21
 * @brief:  In-memory graph generator using KaGen.
 ******************************************************************************/
#include "apps/dkaminpar_graphgen.h"

#include "dkaminpar/coarsening/contraction_helper.h"
#include "dkaminpar/datastructure/distributed_graph_builder.h"
#include "dkaminpar/mpi_wrapper.h"
#include "kaminpar/parallel.h"
#include "kaminpar/utility/random.h"
#include "kaminpar/utility/timer.h"

#include <tbb/parallel_sort.h>

#include <kagen_interface.h>

namespace dkaminpar::graphgen {
using namespace std::string_literals;

DEFINE_ENUM_STRING_CONVERSION(GeneratorType, generator_type) = {
    {GeneratorType::NONE, "none"}, {GeneratorType::GNM, "gnm"},     {GeneratorType::RGG2D, "rgg2d"},
    {GeneratorType::RHG, "rhg"},   {GeneratorType::RDG2D, "rdg2d"}, {GeneratorType::KRONECKER, "kronecker"},
    {GeneratorType::BA, "ba"},
};

using namespace kagen::interface;

namespace {
SET_DEBUG(true);

DistributedGraph build_graph_directed(const auto &edge_list, scalable_vector<GlobalNodeID> node_distribution) {
  auto find_node_owner = [&](const GlobalNodeID node) {
    const auto it = std::upper_bound(node_distribution.begin() + 1, node_distribution.end(), node);
    return static_cast<PEID>(std::distance(node_distribution.begin(), it) - 1);
  };

  const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);
  const auto from = node_distribution[rank];
  const auto to = node_distribution[rank + 1];

  // create reverse internal edges and send reverse edges for ghost nodes to other PEs
  scalable_vector<coarsening::helper::LocalToGlobalEdge> my_edges;
  std::vector<std::vector<coarsening::helper::LocalToGlobalEdge>> other_edges(size);
  for (const auto &[u, v] : edge_list) {
    if (u == v) {
      continue;
    } // ignore self-loops

    ASSERT(from <= u && u < to);
    my_edges.push_back({static_cast<NodeID>(u - from), 1, v});

    if (from <= v && v < to) {
      my_edges.push_back({static_cast<NodeID>(v - from), 1, u});
    } else {
      const PEID owner = find_node_owner(v);
      other_edges[owner].push_back({static_cast<NodeID>(v - node_distribution[owner]), 1, u});
    }
  }

  const auto all_recv_edges =
      mpi::sparse_alltoall_get<coarsening::helper::LocalToGlobalEdge>(std::move(other_edges), MPI_COMM_WORLD);
  for (const auto &recv_edges : all_recv_edges) {
    my_edges.insert(my_edges.end(), recv_edges.begin(), recv_edges.end());
  }

  // deduplicate edges and set weight of all edges to 1
  auto [deduplicated_edges, mem] = coarsening::helper::deduplicate_edge_list(std::move(my_edges), to - from, {});
  for (auto &[u, weight, v] : deduplicated_edges) {
    weight = 1;
  }

  // build new graph
  auto graph = coarsening::helper::build_distributed_graph_from_edge_list(
      deduplicated_edges, std::move(node_distribution), MPI_COMM_WORLD, [&](const auto /* node */) { return 1; },
      find_node_owner);
  for (const NodeID ghost : graph.ghost_nodes()) {
    graph.set_ghost_node_weight(ghost, 1);
  }
  graph::debug::validate(graph);
  return graph;
}

DistributedGraph build_graph(const auto &edge_list, scalable_vector<GlobalNodeID> node_distribution) {
  SCOPED_TIMER("Build graph from edge list");

  const auto [size, rank] = mpi::get_comm_info();
  const GlobalNodeID from = node_distribution[rank];
  const GlobalNodeID to = node_distribution[rank + 1];
  ALWAYS_ASSERT(from <= to);

  const auto n = static_cast<NodeID>(to - from);

  // bucket sort nodes
  START_TIMER("Bucket sort");
  scalable_vector<Atomic<NodeID>> buckets(n);
  tbb::parallel_for<EdgeID>(0, edge_list.size(), [&](const EdgeID e) {
    const GlobalNodeID u = edge_list[e].first;
    if (from <= u && u < to) {
      buckets[u - from].fetch_add(1, std::memory_order_relaxed);
    }
  });
  shm::parallel::prefix_sum(buckets.begin(), buckets.end(), buckets.begin());
  STOP_TIMER();

  const EdgeID m = buckets.back();

  // build edges array
  START_TIMER("Build edges array");
  scalable_vector<EdgeID> edges(m);
  graph::GhostNodeMapper ghost_node_mapper(node_distribution);
  tbb::parallel_for<EdgeID>(0, edge_list.size(), [&](const EdgeID e) {
    const GlobalNodeID u = edge_list[e].first;

    if (from <= u && u < to) {
      const GlobalNodeID v = edge_list[e].second;
      const auto pos = buckets[u - from].fetch_sub(1, std::memory_order_relaxed) - 1;
      ASSERT(pos < edges.size()) << V(pos) << V(edges.size());

      if (from <= v && v < to) {
        edges[pos] = static_cast<NodeID>(v - from);
      } else {
        edges[pos] = ghost_node_mapper.new_ghost_node(v);
      }
    }
  });
  STOP_TIMER();

  auto mapped_ghost_nodes = TIMED_SCOPE("Finalize ghost node mapping") { return ghost_node_mapper.finalize(); };

  // build nodes array
  START_TIMER("Build nodes array");
  scalable_vector<NodeID> nodes(n + 1);
  tbb::parallel_for<NodeID>(0, n, [&](const NodeID u) { nodes[u] = buckets[u]; });
  nodes.back() = m;
  STOP_TIMER();

  DistributedGraph graph{std::move(node_distribution),
                         mpi::build_distribution_from_local_count<GlobalEdgeID, scalable_vector>(m, MPI_COMM_WORLD),
                         std::move(nodes),
                         std::move(edges),
                         std::move(mapped_ghost_nodes.ghost_owner),
                         std::move(mapped_ghost_nodes.ghost_to_global),
                         std::move(mapped_ghost_nodes.global_to_ghost),
                         MPI_COMM_WORLD};
  HEAVY_ASSERT(graph::debug::validate(graph));
  return graph;
}

scalable_vector<GlobalNodeID> build_node_distribution(const std::pair<SInt, SInt> range) {
  const auto [size, rank] = mpi::get_comm_info();
  const GlobalNodeID to = range.second + 1;
  //  DLOG << V(range.first) << V(range.second);

  scalable_vector<GlobalNodeID> node_distribution(size + 1);
  mpi::allgather(&to, 1, node_distribution.data() + 1, 1);
  return node_distribution;
}
} // namespace

/*
DistributedGraph create_undirected_gmm(const GlobalNodeID n, const GlobalEdgeID m, const BlockID k, const int seed) {
const auto [edges, range] = TIMED_SCOPE("KaGen") {
  const auto [size, rank] = mpi::get_comm_info();
  return KaGen{rank, size}.GenerateUndirectedGNM(n, m, k, seed);
};
return build_graph(edges, build_node_distribution(range));
}
*/

DistributedGraph create_rgg2d(const GlobalNodeID n, const double r, const BlockID k, const int seed) {
  const auto [edges, range] = TIMED_SCOPE("KaGen") {
    const auto [size, rank] = mpi::get_comm_info();
    return KaGen{rank, size}.Generate2DRGG(n, r, k, seed);
  };

  LOG << "Building graph with " << edges.size() << " edges on PE 0";
  return build_graph(edges, build_node_distribution(range));
}

DistributedGraph create_rhg(const GlobalNodeID n, const double gamma, const NodeID d, const BlockID k, const int seed) {
  const auto [edges, range] = TIMED_SCOPE("KaGen") {
    const auto [size, rank] = mpi::get_comm_info();
    return KaGen{rank, size}.GenerateRHG(n, gamma, d, k, seed);
  };
  return build_graph(edges, build_node_distribution(range));
}

/*
DistributedGraph create_kronecker(const GlobalNodeID n, const GlobalEdgeID m, const BlockID k, const int seed) {
const auto [edges, range] = TIMED_SCOPE("KaGen") {
  const auto [size, rank] = mpi::get_comm_info();
  return KaGen{rank, size}.GenerateKronecker(n, m, k, seed);
};
return build_graph(edges, build_node_distribution(range));
}
*/

/*
DistributedGraph create_rdg2d(const GlobalNodeID n, const BlockID k, const int seed) {
const auto [edges, range] = TIMED_SCOPE("KaGen") {
  const auto [size, rank] = mpi::get_comm_info();
  return KaGen{rank, size}.Generate2DRDG(n, k, seed);
};
return build_graph(edges, build_node_distribution(range));
}
*/

DistributedGraph create_ba(const GlobalNodeID n, const NodeID d, const BlockID k, const int seed) {
  const auto [edges, range] = TIMED_SCOPE("KaGen") {
    const auto [size, rank] = mpi::get_comm_info();
    return KaGen{rank, size}.GenerateBA(n, d, k, seed);
  };
  return build_graph_directed(edges, build_node_distribution(range));
}

DistributedGraph generate(const GeneratorContext ctx) {
  const int seed = static_cast<int>(shm::Randomize::instance().random_index(0, std::numeric_limits<int>::max()));

  switch (ctx.type) {
  case GeneratorType::NONE:
    FATAL_ERROR << "no graph generator configured";
    break;

    /*
  case GeneratorType::GNM: {
    ALWAYS_ASSERT(ctx.n > 0 && ctx.m > 0) << "Must specify bost number of nodes and number of edges";
    GlobalNodeID n = 1;
    GlobalEdgeID m = 1;
    n <<= ctx.n;
    m <<= ctx.m;
    return create_undirected_gmm(n, m, ctx.k, seed);
  }
    */

  case GeneratorType::RGG2D: {
    ALWAYS_ASSERT(ctx.r > 0) << "Radius cannot be zero";
    ALWAYS_ASSERT(ctx.n > 0 || ctx.m > 0) << "Number of nodes or number of edges must be specified";
    ALWAYS_ASSERT(ctx.n == 0 || ctx.m == 0) << "Cannot specify both number of nodes and number of edges";

    GlobalNodeID n = 1;
    if (ctx.m > 0) {
      GlobalEdgeID m = 1;
      m <<= ctx.m;
      n = static_cast<GlobalNodeID>(std::sqrt(1.0 * static_cast<double>(m) * M_PI) / ctx.r);
    } else {
      n <<= ctx.n;
    }

    LOG << "Generate 2D RGG graph with n=" << n << ", r=" << ctx.r << ", k=" << ctx.k << ", seed=" << seed;
    return create_rgg2d(n, ctx.r, ctx.k, seed);
  }

  case GeneratorType::RHG: {
    ALWAYS_ASSERT(ctx.gamma > 0) << "Must specify gamma";
    ALWAYS_ASSERT(ctx.d > 0) << "Must specify average degree";
    ALWAYS_ASSERT(ctx.n > 0 || ctx.m > 0) << "Must specify number of nodes or number of edges";
    ALWAYS_ASSERT(ctx.n == 0 || ctx.m == 0) << "Cannot specify both number of nodes and number of edges";

    GlobalNodeID n = 1;
    if (ctx.m > 0) {
      GlobalEdgeID m = 1;
      m <<= ctx.m;
      n = m / ctx.d;
    } else {
      n <<= ctx.n;
    }

    LOG << "Generate 2D RHG graph with n=" << n << ", gamma=" << ctx.gamma << ", d=" << ctx.d << ", k=" << ctx.k
        << ", seed=" << seed;
    return create_rhg(n, ctx.gamma, ctx.d, ctx.k, seed);
  }

  case GeneratorType::BA: {
    ALWAYS_ASSERT(ctx.d > 0) << "Must specify minimum degree";
    ALWAYS_ASSERT(ctx.n > 0 || ctx.m > 0) << "Must specify number of nodes or number of edges";
    ALWAYS_ASSERT(ctx.n == 0 || ctx.m == 0) << "Cannot specify both number of nodes and number of edges";

    GlobalNodeID n = 1;
    if (ctx.m > 0) {
      GlobalEdgeID m = 1;
      m <<= ctx.m;
      n = m / ctx.d / 2;
    } else {
      n <<= ctx.n;
    }

    LOG << "Generate BA graph with n=" << n << ", d=" << ctx.d << ", k=" << ctx.k << ", seed=" << seed;
    return create_ba(n, ctx.d, ctx.k, seed);
  }

    /*
  case GeneratorType::KRONECKER: {
    ALWAYS_ASSERT(false) << "broken -- does not generate any edges";
    ALWAYS_ASSERT(ctx.n > 0 && ctx.m > 0) << "Must specify number of nodes and number of edges";
    GlobalNodeID n = 1;
    GlobalEdgeID m = 1;
    n <<= ctx.n;
    m <<= ctx.m;

    return create_kronecker(n, m, ctx.k, seed);
  }
    */

    /*
  case GeneratorType::RDG2D: {
    ALWAYS_ASSERT(false) << "broken";
    ALWAYS_ASSERT(ctx.n > 0 || ctx.m > 0) << "Must specify number of nodes or number of edges";
    ALWAYS_ASSERT(ctx.n == 0 || ctx.m == 0) << "Cannot specify both number of nodes and number of edges";
    GlobalNodeID n = 1;

    if (ctx.m > 0) {
      GlobalEdgeID m = 1;
      m <<= ctx.m;
      n = m / 6;
    } else {
      n <<= ctx.n;
    }
    FATAL_ERROR << "disabled";
    //  return create_rdg2d(n, ctx.k, seed);
  }
    */

  default:
    FATAL_ERROR << "graph generator is deactivated";
  }

  __builtin_unreachable();
}
} // namespace dkaminpar::graphgen
