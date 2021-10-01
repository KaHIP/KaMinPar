/*******************************************************************************
 * @file:   locking_lp_clustering.cc
 *
 * @author: Daniel Seemaier
 * @date:   01.10.21
 * @brief:
 ******************************************************************************/
#include "dkaminpar/coarsening/locking_lp_clustering.h"

namespace dkaminpar {
class LockingLpClusteringImpl {
public:
  LockingLpClusteringImpl(const NodeID max_n, const CoarseningContext &c_ctx) {}
};

LockingLpClustering::LockingLpClustering(const NodeID max_n, const CoarseningContext &c_ctx)
    : _impl{std::make_unique<LockingLpClusteringImpl>(max_n, c_ctx)} {}

LockingLpClustering::~LockingLpClustering() = default;

const LockingLpClustering::AtomicClusterArray &LockingLpClustering::compute_clustering(const DistributedGraph &graph,
                                                                                       NodeWeight max_cluster_weight) {}
} // namespace dkaminpar