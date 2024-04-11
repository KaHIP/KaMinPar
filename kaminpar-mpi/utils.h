/*******************************************************************************
 * Utility functions for MPI.
 *
 * @file:   utils.h
 * @author: Daniel Seemaier
 * @date:   08.08.20202
 ******************************************************************************/
#pragma once

#include <vector>

#include <mpi.h>

#include "kaminpar-mpi/definitions.h"

namespace kaminpar::mpi {
inline std::pair<PEID, PEID> get_comm_info(MPI_Comm comm) {
  int size;
  MPI_Comm_size(comm, &size);
  int rank;
  MPI_Comm_rank(comm, &rank);
  return {size, rank};
}

inline PEID get_comm_size(MPI_Comm comm) {
  int size;
  MPI_Comm_size(comm, &size);
  return size;
}

inline PEID get_comm_rank(MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

template <typename Lambda> inline void sequentially(Lambda &&lambda, MPI_Comm comm) {
  const auto [size, rank] = get_comm_info(comm);
  for (int p = 0; p < size; ++p) {
    if (p == rank) {
      lambda(p);
    }
    MPI_Barrier(comm);
  }
}
} // namespace kaminpar::mpi
