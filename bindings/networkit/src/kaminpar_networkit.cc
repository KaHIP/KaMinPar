/*******************************************************************************
 * NetworKit bindings for shared-memory KaMinPar.
 *
 * @file:   kaminpar_networkit.cc
 * @author: Daniel Seemaier
 * @date:   09.12.2024
 ******************************************************************************/
#include "kaminpar_networkit.h"

#include <kaminpar-common/datastructures/static_array.h>
#include <kaminpar-common/parallel/algorithm.h>
#include <kaminpar-shm/datastructures/csr_graph.h>
#include <kaminpar-shm/datastructures/graph.h>
#include <tbb/task_arena.h>

namespace kaminpar {

KaMinParNetworKit::KaMinParNetworKit(const NetworKit::Graph &G)
    : KaMinPar(tbb::this_task_arena::max_concurrency(), shm::create_default_context()) {
  KaMinPar::set_output_level(kaminpar::OutputLevel::QUIET);
  copyGraph(G);
}

void KaMinParNetworKit::copyGraph(const NetworKit::Graph &G) {
  using namespace kaminpar::shm;

  if (G.isDirected()) {
    throw std::invalid_argument("KaMinParNetworKit only supports undirected graphs.");
  }

  const NodeID n = G.numberOfNodes();
  const EdgeID m = G.numberOfEdges();

  StaticArray<EdgeID> xadj(n + 1);
  G.parallelForNodes([&](const NodeID u) { xadj[u] = G.degree(u); });
  parallel::prefix_sum(xadj.begin(), xadj.end(), xadj.begin());

  const bool hasEdgeWeights = G.isWeighted();
  StaticArray<EdgeWeight> adjwgt(hasEdgeWeights ? m : 0);

  StaticArray<NodeID> adjncy(2 * m);
  G.parallelForEdges(
      [&](const NetworKit::node u, const NetworKit::node v, const NetworKit::edgeweight weight) {
        const std::size_t uPos = __atomic_sub_fetch(&xadj[u], 1, __ATOMIC_RELAXED);
        const std::size_t vPos = __atomic_sub_fetch(&xadj[v], 1, __ATOMIC_RELAXED);
        adjncy[uPos] = v;
        adjncy[vPos] = u;
        if (hasEdgeWeights) {
          adjwgt[uPos] = weight;
          adjwgt[vPos] = weight;
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
NetworKit::Partition computePartitionGeneric(KaMinParNetworKit &shm, Lambda &&lambda) {
  using namespace kaminpar::shm;

  NetworKit::Partition partition(shm.graph()->n());
  StaticArray<BlockID> partitionVec(shm.graph()->n());

  lambda(partitionVec);
  shm.graph()->csr_graph().pfor_nodes([&](const NodeID u) { partition[u] = partitionVec[u]; });

  return partition;
}

} // namespace

NetworKit::Partition KaMinParNetworKit::computePartition(shm::BlockID k) {
  return computePartitionGeneric(*this, [&](StaticArray<shm::BlockID> &vec) {
    KaMinPar::compute_partition(k, vec);
  });
}

NetworKit::Partition
KaMinParNetworKit::computePartitionWithEpsilon(shm::BlockID k, double epsilon) {
  return computePartitionGeneric(*this, [&](StaticArray<shm::BlockID> &vec) {
    KaMinPar::compute_partition(k, epsilon, vec);
  });
}

NetworKit::Partition
KaMinParNetworKit::computePartitionWithFactors(std::vector<double> maxBlockWeightFactors) {
  return computePartitionGeneric(*this, [&](StaticArray<shm::BlockID> &vec) {
    KaMinPar::compute_partition(std::move(maxBlockWeightFactors), vec);
  });
}

NetworKit::Partition
KaMinParNetworKit::computePartitionWithWeights(std::vector<shm::BlockWeight> maxBlockWeights) {
  return computePartitionGeneric(*this, [&](StaticArray<shm::BlockID> &vec) {
    KaMinPar::compute_partition(std::move(maxBlockWeights), vec);
  });
}

} // namespace kaminpar
