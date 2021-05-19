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
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <graph> <k>" << std::endl;
    std::exit(1);
  }

  const auto filename = std::string(argv[1]);
  const auto k = static_cast<int>(std::strtol(argv[2], nullptr, 10));

  std::cout << "Partitioning " << filename << " into " << k << " blocks ..." << std::endl;

  auto partitioner = libkaminpar::PartitionerBuilder::from_graph_file(filename).create();
  const auto partition_size = partitioner.partition_size();
  auto partition = partitioner.partition(k);

  std::cout << "And we are done" << std::endl;

  std::vector<int> block_sizes(k);
  for (std::size_t i = 0; i < partition_size; ++i) { ++block_sizes[partition[i]]; }

  std::cout << "Block sizes:" << std::endl;
  for (int b = 0; b < k; ++b) { std::cout << " block " << b << ": " << block_sizes[b] << std::endl; }

  return 0;
}