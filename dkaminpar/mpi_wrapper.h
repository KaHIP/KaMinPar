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

#include <concepts>
#include <mpi.h>
#include <type_traits>

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

//
// Map MPI datatypes
//

template<std::size_t N>
MPI_Datatype custom_datatype() {
  static MPI_Datatype type = nullptr;
  if (type == nullptr) {
    MPI_Type_contiguous(N, MPI_CHAR, &type);
    MPI_Type_commit(&type);
  }
  return type;
}

template<typename Raw>
MPI_Datatype datatype() {
  using T = std::decay_t<Raw>;
  if (std::is_same_v<T, std::uint8_t>) {
    return MPI_UINT8_T;
  } else if (std::is_same_v<T, std::int8_t>) {
    return MPI_INT8_T;
  } else if (std::is_same_v<T, std::uint16_t>) {
    return MPI_UINT16_T;
  } else if (std::is_same_v<T, std::int16_t>) {
    return MPI_INT16_T;
  } else if (std::is_same_v<T, std::uint32_t>) {
    return MPI_UINT32_T;
  } else if (std::is_same_v<T, std::int32_t>) {
    return MPI_INT32_T;
  } else if (std::is_same_v<T, std::uint64_t>) {
    return MPI_UINT64_T;
  } else if (std::is_same_v<T, std::int64_t>) {
    return MPI_INT64_T;
  } else if (std::is_same_v<T, float>) {
    return MPI_FLOAT;
  } else if (std::is_same_v<T, double>) {
    return MPI_DOUBLE;
  } else if (std::is_same_v<T, long double>) {
    return MPI_LONG_DOUBLE;
  }

  return custom_datatype<sizeof(T)>();
}

//
// Pointer interface
//

inline int barrier(MPI_Comm comm = MPI_COMM_WORLD) { return MPI_Barrier(comm); }

template<typename T>
inline int bcast(T *buffer, const int count, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Bcast(buffer, count, datatype<T>(), root, comm);
}

template<typename T>
inline int reduce(const T *sendbuf, T *recvbuf, const int count, MPI_Op op, const int root = 0,
                  MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Reduce(sendbuf, recvbuf, count, datatype<T>(), op, root, comm);
}

template<typename T>
inline int allreduce(const T *sendbuf, T *recvbuf, const int count, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Allreduce(sendbuf, recvbuf, count, datatype<T>(), op, comm);
}

template<typename Ts, typename Tr>
inline int scatter(const Ts *sendbuf, const int sendcount, Tr *recvbuf, const int recvcount, const int root,
                   MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Scatter(sendbuf, sendcount, datatype<Ts>(), recvbuf, recvcount, datatype<Tr>(), root, comm);
}

template<typename Ts, typename Tr>
inline int gather(const Ts *sendbuf, const int sendcount, Tr *recvbuf, const int recvcount, const int root = 0,
                  MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Gather(sendbuf, sendcount, datatype<Ts>(), recvbuf, recvcount, datatype<Tr>(), root, comm);
}

template<typename Ts, typename Tr>
inline int allgather(const Ts *sendbuf, const int sendcount, Tr *recvbuf, const int recvcount,
                     MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Allgather(sendbuf, sendcount, datatype<Ts>(), recvbuf, recvcount, datatype<Tr>(), comm);
}

template<typename Ts, typename Tr>
inline int alltoall(const Ts *sendbuf, const int sendcount, Tr *recvbuf, const int recvcount,
                    MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Alltoall(sendbuf, sendcount, datatype<Ts>(), recvbuf, recvcount, datatype<Tr>(), comm);
}

template<typename T>
inline int scan(const T *sendbuf, T *recvbuf, const int count, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Scan(sendbuf, recvbuf, count, datatype<T>(), op, comm);
}

template<typename T>
inline int exscan(const T *sendbuf, T *recvbuf, const int count, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Exscan(sendbuf, recvbuf, count, datatype<T>(), op, comm);
}

template<typename T>
inline int reduce_scatter(const T *sendbuf, T *recvbuf, int *recvcounts, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, datatype<T>(), op, comm);
}

//
// Ranges interface
//

template<std::ranges::contiguous_range R>
inline int reduce(const R &sendbuf, R &recvbuf, MPI_Op op, int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  ASSERT(mpi::get_comm_rank(comm) != root || std::ranges::size(sendbuf) == std::ranges::size(recvbuf))
      << "recvbuf(" << std::ranges::size(recvbuf) << ") has not the same size as sendbuf(" << std::ranges::size(sendbuf)
      << ") on root " << root;
  return reduce<std::ranges::range_value_t<R>>(std::ranges::cdata(sendbuf), std::ranges::data(recvbuf),
                                               std::ranges::ssize(sendbuf), op, root, comm);
}

template<std::ranges::contiguous_range Rs, std::ranges::contiguous_range Rr>
inline int gather(const Rs &sendbuf, Rr &recvbuf, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  using rs_value_t = std::ranges::range_value_t<Rs>;
  using rr_value_t = std::ranges::range_value_t<Rr>;
  ASSERT(mpi::get_comm_rank(comm) != root ||
         sizeof(rs_value_t) * std::ranges::size(sendbuf) ==
             mpi::get_comm_size(comm) * sizeof(rr_value_t) * std::ranges::size(recvbuf))
      << "recvbuf is not large enough to receive all sendbufs";
  return gather<rs_value_t, rr_value_t>(std::ranges::cdata(sendbuf), std::ranges::ssize(sendbuf),
                                        std::ranges::data(recvbuf), std::ranges::ssize(recvbuf), root, comm);
}

template<typename T, template<typename> typename Container = scalable_vector>
Container<T> gather(const T element, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  Container<T> result;
  if (mpi::get_comm_rank(comm) == root) { result.resize(mpi::get_comm_size(comm)); }
  gather(&element, 1, std::ranges::data(result), 1, root, comm);
  return result;
}

//
// Misc
//

template<typename T>
inline int send(const T *buf, int count, int dest, int tag, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Send(buf, count, datatype<T>(), dest, tag, comm);
}

template<typename T>
inline int isend(const T *buf, int count, int dest, int tag, MPI_Request *request, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Isend(buf, count, datatype<T>(), dest, tag, comm, request);
}

inline MPI_Status probe(int source, int tag, MPI_Comm comm = MPI_COMM_WORLD) {
  MPI_Status status;
  MPI_Probe(source, tag, comm, &status);
  return status;
}

template<typename T>
inline int get_count(MPI_Status *status) {
  int count;
  MPI_Get_count(status, datatype<T>(), &count);
  return count;
}

inline int waitall(int count, MPI_Request *array_of_requests, MPI_Status *array_of_statuses = MPI_STATUS_IGNORE) {
  return MPI_Waitall(count, array_of_requests, array_of_statuses);
}

template<typename T>
inline int recv(T *buf, int count, int source, int tag, MPI_Status *status = MPI_STATUS_IGNORE,
                MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Recv(buf, count, datatype<T>(), source, tag, comm, status);
}

} // namespace dkaminpar::mpi