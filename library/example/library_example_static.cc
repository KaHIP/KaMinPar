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

#include <iostream>
#include <libkaminpar.h>
#include <vector>

int main(int, char *[]) {
  const int k = 2;

  // graph from the manual
  std::vector<libkaminpar::EdgeID> nodes{0, 2, 5, 7, 9, 12};
  std::vector<libkaminpar::NodeID> edges{1, 4, 0, 2, 4, 1, 3, 2, 4, 0, 1, 3};

  libkaminpar::Partitioner partitioner = libkaminpar::PartitionerBuilder                                      //
                                         ::from_adjacency_array(nodes.size() - 1, nodes.data(), edges.data()) //
                                             .create();                                                       //

  partitioner.set_option("--threads", "6");    // use 6 cores
  partitioner.set_option("--epsilon", "0.04"); // allow 4% imbalance

  auto partition = partitioner.partition(k); // compute 16-way partition

  std::vector<int> block_sizes(k);
  for (std::size_t i = 0; i < partitioner.partition_size(); ++i) { ++block_sizes[partition[i]]; }

  std::cout << "Block sizes:" << std::endl;
  for (int b = 0; b < k; ++b) { std::cout << " block " << b << ": " << block_sizes[b] << std::endl; }

  return 0;
}