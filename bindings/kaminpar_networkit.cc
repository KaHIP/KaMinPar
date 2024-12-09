/*******************************************************************************
 * NetworKit bindings for shared-memory KaMinPar.
 *
 * @file:   kaminpar_networkit.cc
 * @author: Daniel Seemaier
 * @date:   09.12.2024
 ******************************************************************************/
#include "kaminpar_networkit.h"

#include <networkit/graph/Graph.hpp>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/parallel/algorithm.h"

namespace kaminpar {

KaMinParNetworKit::KaMinParNetworKit(const int num_threads, const shm::Context &ctx)
    : KaMinPar(num_threads, ctx) {}

void KaMinParNetworKit::copy_graph(const NetworKit::Graph &graph) {
  using namespace kaminpar::shm;

  if (graph.isDirected()) {
    throw std::invalid_argument("KaMinParNetworKit only supports undirected graphs.");
  }

  const NodeID n = graph.numberOfNodes();
  const EdgeID m = graph.numberOfEdges();

  StaticArray<EdgeID> xadj(n + 1);
  graph.parallelForNodes([&](const NodeID u) { xadj[u] = graph.degree(u); });
  parallel::prefix_sum(xadj.begin(), xadj.end(), xadj.begin());

  const bool has_edge_weights = graph.isWeighted();
  StaticArray<EdgeWeight> adjwgt(has_edge_weights ? m : 0);

  StaticArray<NodeID> adjncy(2 * m);
  graph.parallelForEdges(
      [&](const NetworKit::node u, const NetworKit::node v, const NetworKit::edgeweight weight) {
        const std::size_t u_pos = __atomic_sub_fetch(&xadj[u], 1, __ATOMIC_RELAXED);
        const std::size_t v_pos = __atomic_sub_fetch(&xadj[v], 1, __ATOMIC_RELAXED);
        adjncy[u_pos] = v;
        adjncy[v_pos] = u;
        if (has_edge_weights) {
          adjwgt[u_pos] = weight;
          adjwgt[v_pos] = weight;
        }
      }
  );
  xadj.back() = 2 * m;

  this->set_graph({std::make_unique<CSRGraph>(
      std::move(xadj), std::move(adjncy), StaticArray<NodeWeight>{}, std::move(adjwgt)
  )});
}

namespace {

template <typename Lambda>
NetworKit::Partition compute_partition_generic(KaMinParNetworKit &shm, Lambda &&lambda) {
  using namespace kaminpar::shm;

  NetworKit::Partition partition(shm.graph()->n());
  StaticArray<BlockID> partition_vec(shm.graph()->n());

  lambda(partition_vec);
  shm.graph()->pfor_nodes([&](const NodeID u) { partition[u] = partition_vec[u]; });

  return partition;
}

} // namespace

NetworKit::Partition KaMinParNetworKit::compute_partition(shm::BlockID k) {
  return compute_partition_generic(*this, [&](StaticArray<shm::BlockID> &vec) {
    this->compute_partition(k, vec);
  });
}

NetworKit::Partition KaMinParNetworKit::compute_partition(shm::BlockID k, double epsilon) {
  return compute_partition_generic(*this, [&](StaticArray<shm::BlockID> &vec) {
    this->compute_partition(k, epsilon, vec);
  });
}

NetworKit::Partition
KaMinParNetworKit::compute_partition(std::vector<double> max_block_weight_factors) {
  return compute_partition_generic(*this, [&](StaticArray<shm::BlockID> &vec) {
    this->compute_partition(std::move(max_block_weight_factors), vec);
  });
}

NetworKit::Partition
KaMinParNetworKit::compute_partition(std::vector<shm::BlockWeight> max_block_weights) {
  return compute_partition_generic(*this, [&](StaticArray<shm::BlockID> &vec) {
    this->compute_partition(std::move(max_block_weights), vec);
  });
}

} // namespace kaminpar
