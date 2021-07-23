/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
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
#include "dkaminpar/partitioning_scheme/partitioning.h"

#include "dkaminpar/partitioning_scheme/kway.h"

namespace dkaminpar {
DistributedPartitionedGraph partition(const DistributedGraph &graph, const DContext &ctx) {
  switch (ctx.partition.mode) {
    case PartitioningMode::KWAY: return KWayPartitioningScheme{graph, ctx}.partition();
    case PartitioningMode::DEEP: FATAL_ERROR << "not implemented"; break;
    case PartitioningMode::RB: FATAL_ERROR << "not implemented"; break;
  }

  __builtin_unreachable();
}
} // namespace dkaminpar