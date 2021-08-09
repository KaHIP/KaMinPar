/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2020 Daniel Seemaier <daniel.seemaier@kit.edu>
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
#include "dkaminpar/utility/distributed_timer.h"

#include "dkaminpar/mpi_wrapper.h"

#include <numeric>
#include <cmath>

namespace dkaminpar::timer {
namespace {
using shm::Timer;

void annotate_subtree(Timer::TimerTreeNode &node, MPI_Comm comm) {
  const auto times = mpi::gather(node.seconds(), 0, comm);
  const auto [min, max] = std::ranges::minmax(times);
  const double mean = static_cast<double>(std::accumulate(times.begin(), times.end(), 0.0)) /
                      static_cast<double>(times.size());

  double sd = 0;
  for (const auto &time : times) { sd += (time - mean) * (time - mean); }
  sd = std::sqrt(1.0 / static_cast<double>(times.size()) * sd);

  std::stringstream ss;
  ss << "[" << min << " s/" << mean << " s/" << max << " s :: sd=" << sd << " s]";
  node.annotation = ss.str();

  for (const auto &child : node.children) { annotate_subtree(*child, comm); }
}
} // namespace

void finalize_distributed_timer(shm::Timer &timer, MPI_Comm comm) { annotate_subtree(timer.tree(), comm); }
} // namespace dkaminpar::timer