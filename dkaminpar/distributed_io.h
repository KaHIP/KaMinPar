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
#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/mpi_wrapper.h"

#include <fstream>
#include <string>

namespace dkaminpar::io {
namespace metis {
DistributedGraph read_node_balanced(const std::string &filename);
DistributedGraph read_edge_balanced(const std::string &filename);
void write(const std::string &filename, const DistributedGraph &graph, bool write_node_weights = true,
           bool write_edge_weights = true);
} // namespace metis

namespace partition {
template<typename Container>
void write(const std::string &filename, const Container &partition) {
  mpi::sequentially([&] {
    std::ofstream out(filename, std::ios_base::out | std::ios_base::app);
    for (const BlockID &b : partition) { out << b << "\n"; }
  });
}
} // namespace partition
} // namespace dkaminpar::io