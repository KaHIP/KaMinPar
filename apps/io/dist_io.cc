/*******************************************************************************
 * IO functions for the distributed partitioner.
 *
 * @file:   dist_io.cc
 * @author: Daniel Seemaier
 * @date:   13.06.2023
 ******************************************************************************/
#include "apps/io/dist_io.h"

#include <fstream>

#include "kaminpar-mpi/wrapper.h"

namespace kaminpar::dist::io::partition {
void write(const std::string &filename, const std::vector<BlockID> &partition) {
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);

  if (rank == 0) {
    std::ofstream out(filename, std::ios::trunc);
  }

  for (PEID pe = 0; pe < size; ++pe) {
    if (pe == rank) {
      std::ofstream out(filename, std::ios::app);
      for (const auto &block : partition) {
        out << block << "\n";
      }
    }
    mpi::barrier(MPI_COMM_WORLD);
  }
}
} // namespace kaminpar::dist::io::partition
