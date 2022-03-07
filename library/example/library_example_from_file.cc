#include <iostream>
#include <kaminpar.h>
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

  auto partitioner = libkaminpar::PartitionerBuilder::from_graph_file(filename).rearrange_and_create();
  partitioner.set_option("--threads", "6").set_option("--seed", "123");

  const auto partition_size = partitioner.partition_size();
  auto partition = partitioner.partition(k);

  std::cout << "Finished!" << std::endl;

  std::vector<int> block_sizes(k);
  for (std::size_t i = 0; i < partition_size; ++i) {
    ++block_sizes[partition[i]];
  }

  std::cout << "Block sizes:" << std::endl;
  for (int b = 0; b < k; ++b) {
    std::cout << " block " << b << ": " << block_sizes[b] << std::endl;
  }

  return 0;
}