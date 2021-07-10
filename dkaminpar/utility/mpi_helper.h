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

#include "dkaminpar/distributed_definitions.h"

#include <concepts>
#include <mpi.h>
#include <utility>
#include <vector>

namespace dkaminpar::mpi {
inline std::pair<int, int> get_comm_info(MPI_Comm comm = MPI_COMM_WORLD) {
  int size;
  MPI_Comm_size(comm, &size);
  int rank;
  MPI_Comm_rank(comm, &rank);
  return {size, rank};
}

inline int get_comm_size(MPI_Comm comm = MPI_COMM_WORLD) {
  int size;
  MPI_Comm_size(comm, &size);
  return size;
}

inline int get_comm_rank(MPI_Comm comm = MPI_COMM_WORLD) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

template<typename Lambda>
inline void sequentially(Lambda &&lambda, MPI_Comm comm = MPI_COMM_WORLD) {
  constexpr bool use_rank_argument = requires { lambda(int()); };
  const auto [size, rank] = get_comm_info();
  for (int p = 0; p < size; ++p) {
    if (p == rank) {
      if constexpr (use_rank_argument) {
        lambda(p);
      } else {
        lambda();
      }
    }
    MPI_Barrier(comm);
  }
}

template<std::ranges::range Distribution>
inline std::vector<int> build_distribution_recvcounts(Distribution &&dist) {
  ASSERT(!dist.empty());
  std::vector<int> recvcounts(dist.size() - 1);
  for (std::size_t i = 0; i + 1 < dist.size(); ++i) { recvcounts[i] = dist[i + 1] - dist[i]; }
  return recvcounts;
}

template<std::ranges::range Distribution>
inline std::vector<int> build_distribution_displs(Distribution &&dist) {
  ASSERT(!dist.empty());
  std::vector<int> displs(dist.size() - 1);
  for (std::size_t i = 0; i + 1 < dist.size(); ++i) { displs[i] = static_cast<int>(dist[i]); }
  return displs;
}

template<typename T>
constexpr MPI_Datatype get_datatype() {
  switch (std::numeric_limits<T>::digits) {
    case 7: return MPI_INT8_T;
    case 8: return MPI_UINT8_T;
    case 15: return MPI_INT16_T;
    case 16: return MPI_UINT64_T;
    case 31: return MPI_INT32_T;
    case 32: return MPI_UINT32_T;
    case 63: return MPI_INT64_T;
    case 64: return MPI_UINT64_T;
  }
}

template<typename T>
inline int reduce(const T *sendbuf, T *recvbuf, const int count, MPI_Op op, int root, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Reduce(sendbuf, recvbuf, count, get_datatype<T>(), op, root, comm);
}

template<typename T>
inline int gather(const T *sendbuf, const int sendcount, T *recvbuf, int recvcount, int root, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Gather(sendbuf, sendcount, get_datatype<T>(), recvbuf, recvcount, get_datatype<T>(), root, comm);
}

template<typename T>
inline int send(const T *buf, int count, int dest, int tag, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Send(buf, count, get_datatype<T>(), dest, tag, comm);
}

template<typename T>
inline int isend(const T *buf, int count, int dest, int tag, MPI_Request *request, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Isend(buf, count, get_datatype<T>(), dest, tag, comm, request);
}

inline MPI_Status probe(int source, int tag, MPI_Comm comm = MPI_COMM_WORLD) {
  MPI_Status status;
  MPI_Probe(source, tag, comm, &status);
  return status;
}

template<typename T>
inline int get_count(MPI_Status *status) {
  int count;
  MPI_Get_count(status, get_datatype<T>(), &count);
  return count;
}

inline int waitall(int count, MPI_Request *array_of_requests, MPI_Status *array_of_statuses = MPI_STATUS_IGNORE) {
  return MPI_Waitall(count, array_of_requests, array_of_statuses);
}

template<typename T>
inline int recv(T *buf, int count, int source, int tag, MPI_Status *status = MPI_STATUS_IGNORE, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Recv(buf, count, get_datatype<T>(), source, tag, comm, status);
}
} // namespace dkaminpar::mpi