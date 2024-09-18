/*******************************************************************************
 * Statistics for the distributed label propagation refiner.
 *
 * @file:   lp_stats.cc
 * @author: Daniel Seemaier
 * @date:   18.09.2024
 ******************************************************************************/
#include "kaminpar-dist/refinement/lp/lp_stats.h"

#include <mpi.h>

#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/logger.h"

namespace kaminpar::dist::lp {

void RefinerStatistics::reset() {
  cut_before = 0;
  cut_after = 0;
  num_successful_moves = 0;
  num_rollbacks = 0;
  num_tentatively_moved_nodes = 0;
  num_tentatively_rejected_nodes = 0;
  max_balance_violation = 0.0;
  total_balance_violation = 0.0;
  expected_gain = 0;
  realized_gain = 0;
  rejected_gain = 0;
  rollback_gain = 0;
  expected_imbalance = 0;
}

void RefinerStatistics::print(MPI_Comm comm) {
  auto expected_gain_reduced = mpi::reduce_single<EdgeWeight>(expected_gain, MPI_SUM, 0, comm);
  auto realized_gain_reduced = mpi::reduce_single<EdgeWeight>(realized_gain, MPI_SUM, 0, comm);
  auto rejected_gain_reduced = mpi::reduce_single<EdgeWeight>(rejected_gain, MPI_SUM, 0, comm);
  auto rollback_gain_reduced = mpi::reduce_single<EdgeWeight>(rollback_gain, MPI_SUM, 0, comm);
  auto expected_imbalance_str = mpi::gather_statistics_str(expected_imbalance, comm);
  auto num_tentatively_moved_nodes_str =
      mpi::gather_statistics_str(num_tentatively_moved_nodes.load(), comm);
  auto num_tentatively_rejected_nodes_str =
      mpi::gather_statistics_str(num_tentatively_rejected_nodes.load(), comm);

  LOG_STATS << "Label Propagation Refinement:";
  LOG_STATS << "- Iterations: " << num_successful_moves << " ok, " << num_rollbacks << " failed";
  LOG_STATS << "- Expected gain: " << expected_gain_reduced
            << " (total expectation value of move gains)";
  LOG_STATS << "- Realized gain: " << realized_gain_reduced
            << " (total value of realized move gains)";
  LOG_STATS << "- Rejected gain: " << rejected_gain_reduced;
  LOG_STATS << "- Rollback gain: " << rollback_gain_reduced
            << " (gain of moves affected by rollback)";
  LOG_STATS << "- Actual gain: " << cut_before - cut_after << " (from " << cut_before << " to "
            << cut_after << ")";
  LOG_STATS << "- Balance violations: " << total_balance_violation / num_rollbacks << " / "
            << max_balance_violation;
  LOG_STATS << "- Expected imbalance: [" << expected_imbalance_str << "]";
  LOG_STATS << "- Num tentatively moved nodes: [" << num_tentatively_moved_nodes_str << "]";
  LOG_STATS << "- Num tentatively rejected nodes: [" << num_tentatively_rejected_nodes_str << "]";
}

} // namespace kaminpar::dist::lp
