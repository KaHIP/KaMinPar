/*******************************************************************************
 * Statistics for the distributed label propagation refiner.
 *
 * @file:   lp_stats.h
 * @author: Daniel Seemaier
 * @date:   18.09.2024
 ******************************************************************************/
#pragma once

#include <atomic>

#include <mpi.h>

#include "kaminpar-dist/dkaminpar.h"

namespace kaminpar::dist::lp {

struct RefinerStatistics {
  EdgeWeight cut_before = 0;
  EdgeWeight cut_after = 0;

  // Global:
  int num_successful_moves = 0;

  // Global:
  int num_rollbacks = 0;

  std::atomic<int> num_tentatively_moved_nodes = 0;
  std::atomic<int> num_tentatively_rejected_nodes = 0;

  double max_balance_violation = 0.0;   // global, only if rollback occurred
  double total_balance_violation = 0.0; // global, only if rollback occurred

  // Local, expectation value of probabilistic gain values
  std::atomic<EdgeWeight> expected_gain = 0;

  // Local, gain values of moves that were accept / rejected
  std::atomic<EdgeWeight> realized_gain = 0;
  std::atomic<EdgeWeight> rejected_gain = 0;

  // Local, gain values that were rollbacked
  std::atomic<EdgeWeight> rollback_gain = 0;

  // Local, expected imbalance
  double expected_imbalance = 0;

  void reset();
  void print(MPI_Comm comm);
};

} // namespace kaminpar::dist::lp
