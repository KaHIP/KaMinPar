
#include "sparsification_utils.h"
namespace kaminpar::shm::sparsification::utils {

StaticArray<EdgeID> sort_by_traget(const CSRGraph &g) {
  auto permutation = StaticArray<EdgeID>(g.m());
  for (auto e : g.edges())
    permutation[e] = e;
  tbb::parallel_for (static_cast<NodeID>(0), g.n()-1, [&](const NodeID u) {
    std::sort(
        permutation.begin() + g.raw_nodes()[u],
        permutation.begin() + g.raw_nodes()[u + 1]
    );
  });
  return permutation;
}

} // namespace kaminpar::shm::sparsification::utils
