/*******************************************************************************
 * Computes a mapping of an arbitrary number of PEs to a 2D grid.
 *
 * @file:   grid_topology.cc
 * @author: Daniel Seemaier
 * @date:   27.03.2023
 ******************************************************************************/
#include "kaminpar-mpi/grid_topology.h"

#include <unordered_map>

#include <mpi.h>

namespace kaminpar::mpi {
GridCommunicator &get_grid_communicator(MPI_Comm comm) {
  static std::unordered_map<MPI_Comm, GridCommunicator> grid_communicators;
  auto [grid_comm_it, ignored] = grid_communicators.try_emplace(comm, comm);
  return grid_comm_it->second;
}
} // namespace kaminpar::mpi
