/*******************************************************************************
* @file:   dkaminpar.cc
*
* @author: Daniel Seemaier
* @date:   21.09.21
* @brief:  Distributed KaMinPar binary.
******************************************************************************/
// This must come first since it redefines output macros (LOG DBG etc)
// clang-format off
#include "dkaminpar/distributed_definitions.h"
// clang-format on

#include "apps.h"
#include "dkaminpar/algorithm/distributed_graph_contraction.h"
#include "dkaminpar/application/arguments.h"
#include "dkaminpar/distributed_context.h"
#include "dkaminpar/distributed_io.h"
#include "dkaminpar/growt.h"
#include "dkaminpar/partitioning_scheme/partitioning.h"
#include "dkaminpar/utility/distributed_metrics.h"
#include "dkaminpar/utility/distributed_timer.h"
#include "kaminpar/definitions.h"
#include "kaminpar/utility/logger.h"
#include "kaminpar/utility/random.h"
#include "kaminpar/utility/timer.h"

#include <fstream>
#include <mpi.h>

namespace dist = dkaminpar;
namespace shm = kaminpar;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  using hasher_type = utils_tm::hash_tm::murmur2_hash;
  using allocator_type = growt::AlignedAllocator<>;
  using table_type = typename growt::table_config<dist::GlobalNodeID, dist::NodeWeight, hasher_type, allocator_type,
                                                  hmod::growable, hmod::deletion>::table_type;

  table_type _cluster_weights{7};

  //  tbb::enumerable_thread_specific<typename table_type::handle_type> _cluster_weights_handles_ets{
  //      [&] { return table_type::handle_type{_cluster_weights}; }};
    tbb::enumerable_thread_specific<typename table_type::handle_type> _cluster_weights_handles_ets{
        [&] { LOG << "create"; return _cluster_weights.get_handle(); }};

    auto &handle = _cluster_weights_handles_ets.local();

  //  auto org_handle = _cluster_weights.get_handle();

  {
//    auto handle = _cluster_weights.get_handle();
    handle.insert(1, 1);
  }
  {
//    auto handle = _cluster_weights.get_handle();
    handle.insert(2, 1);
  }
  {
//    auto handle = _cluster_weights.get_handle();
    handle.insert(3, 1);
  }
  {
//    auto handle = _cluster_weights.get_handle();
    handle.insert(8, 1);
  }
  {
//    auto handle = _cluster_weights.get_handle();
    handle.insert(4, 1);
  }
  {
//    auto handle = _cluster_weights.get_handle();
    handle.insert(6, 1);
  }
  {
//    auto handle = _cluster_weights.get_handle();
    handle.insert(9, 1);
  }

  {
//    auto handle = _cluster_weights.get_handle();
    ASSERT(handle.find(1) != handle.end());
  }
  {
//    auto handle = _cluster_weights.get_handle();
    ASSERT(handle.find(2) != handle.end());
  }
  {
//    auto handle = _cluster_weights.get_handle();
    ASSERT(handle.find(3) != handle.end());
  }
  {
//    auto handle = _cluster_weights.get_handle();
    ASSERT(handle.find(8) != handle.end());
  }
  {
//    auto handle = _cluster_weights.get_handle();
    ASSERT(handle.find(1) != handle.end());
  }

  const dist::NodeWeight delta = 1;

  LOG << "jetzt das update:";
//  auto handle = _cluster_weights.get_handle();
  [[maybe_unused]] const auto [old_it, old_found] = handle.update(
      1, [](auto &lhs, const auto rhs) { return lhs -= rhs; }, delta);
  ASSERT(old_it != handle.end() && old_found);

  MPI_Finalize();
  return 0;
}