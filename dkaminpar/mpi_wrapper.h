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
namespace type {
template<std::size_t N>
inline MPI_Datatype custom() {
  static MPI_Datatype type = nullptr;
  if (type == nullptr) {
    MPI_Type_contiguous(N, MPI_CHAR, &type);
    MPI_Type_commit(&type);
  }
  return type;
}

// Map to default MPI type
#define MAP_DATATYPE(CPP_DATATYPE, MPI_DATATYPE)                                                                       \
  template<std::same_as<CPP_DATATYPE>>                                                                                 \
  inline MPI_Datatype get() {                                                                                          \
    return MPI_DATATYPE;                                                                                               \
  }

MAP_DATATYPE(std::uint8_t, MPI_UINT8_T)
MAP_DATATYPE(std::int8_t, MPI_INT8_T)
MAP_DATATYPE(std::uint16_t, MPI_UINT16_T)
MAP_DATATYPE(std::int16_t, MPI_INT16_T)
MAP_DATATYPE(std::uint32_t, MPI_UINT32_T)
MAP_DATATYPE(std::int32_t, MPI_INT32_T)
MAP_DATATYPE(std::uint64_t, MPI_UINT64_T)
MAP_DATATYPE(std::int64_t, MPI_INT64_T)
MAP_DATATYPE(float, MPI_FLOAT)
MAP_DATATYPE(double, MPI_DOUBLE)
MAP_DATATYPE(long double, MPI_LONG_DOUBLE)
//MAP_DATATYPE(std::size_t, MPI_UNSIGNED_LONG)

#define COMMA ,
MAP_DATATYPE(std::pair<float COMMA int>, MPI_FLOAT_INT)
MAP_DATATYPE(std::pair<double COMMA int>, MPI_DOUBLE_INT)
MAP_DATATYPE(std::pair<long double COMMA int>, MPI_LONG_DOUBLE_INT)
#undef COMMA

#undef MAP_DATATYPE

// Fallback to custom MPI type
template<typename T>
inline MPI_Datatype get() {
  return custom<sizeof(T)>();
}
} // namespace type

//
// Pointer interface for collective operations
//

inline int barrier(MPI_Comm comm = MPI_COMM_WORLD) { return MPI_Barrier(comm); }

template<typename T>
inline int bcast(T *buffer, const int count, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Bcast(buffer, count, type::get<T>(), root, comm);
}

template<typename T>
inline int reduce(const T *sendbuf, T *recvbuf, const int count, MPI_Op op, const int root = 0,
                  MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Reduce(sendbuf, recvbuf, count, type::get<T>(), op, root, comm);
}

template<typename T>
inline int allreduce(const T *sendbuf, T *recvbuf, const int count, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Allreduce(sendbuf, recvbuf, count, type::get<T>(), op, comm);
}

template<typename Ts, typename Tr>
inline int scatter(const Ts *sendbuf, const int sendcount, Tr *recvbuf, const int recvcount, const int root,
                   MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Scatter(sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcount, type::get<Tr>(), root, comm);
}

template<typename Ts, typename Tr>
inline int gather(const Ts *sendbuf, const int sendcount, Tr *recvbuf, const int recvcount, const int root = 0,
                  MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Gather(sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcount, type::get<Tr>(), root, comm);
}

template<typename Ts, typename Tr>
inline int allgather(const Ts *sendbuf, const int sendcount, Tr *recvbuf, const int recvcount,
                     MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Allgather(sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcount, type::get<Tr>(), comm);
}

template<typename Ts, typename Tr>
inline int alltoall(const Ts *sendbuf, const int sendcount, Tr *recvbuf, const int recvcount,
                    MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Alltoall(sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcount, type::get<Tr>(), comm);
}

template<typename T>
inline int scan(const T *sendbuf, T *recvbuf, const int count, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Scan(sendbuf, recvbuf, count, type::get<T>(), op, comm);
}

template<typename T>
inline int exscan(const T *sendbuf, T *recvbuf, const int count, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Exscan(sendbuf, recvbuf, count, type::get<T>(), op, comm);
}

template<typename T>
inline int reduce_scatter(const T *sendbuf, T *recvbuf, int *recvcounts, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, type::get<T>(), op, comm);
}

template<typename Ts, typename Tr>
int gatherv(const Ts *sendbuf, const int sendcount, Tr *recvbuf, const int *recvcounts, const int *displs,
            const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Gatherv(sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcounts, displs, type::get<Tr>(), root, comm);
}

template<typename Ts, typename Tr>
int allgatherv(const Ts *sendbuf, const int sendcount, Tr *recvbuf, const int *recvcounts, const int *displs,
               MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Allgatherv(sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcounts, displs, type::get<Tr>(), comm);
}

//
// Pointer interface for point-to-point operations
//

template<typename T>
inline int send(const T *buf, const int count, const int dest, const int tag, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Send(buf, count, type::get<T>(), dest, tag, comm);
}

template<typename T>
inline int isend(const T *buf, const int count, const int dest, const int tag, MPI_Request *request,
                 MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Isend(buf, count, type::get<T>(), dest, tag, comm, request);
}

template<typename T>
inline int recv(T *buf, int count, int source, int tag, MPI_Status *status = MPI_STATUS_IGNORE,
                MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Recv(buf, count, type::get<T>(), source, tag, comm, status);
}

inline MPI_Status probe(const int source, const int tag, MPI_Comm comm = MPI_COMM_WORLD) {
  MPI_Status status;
  MPI_Probe(source, tag, comm, &status);
  return status;
}

template<typename T>
inline int get_count(const MPI_Status &status) {
  int count;
  MPI_Get_count(&status, type::get<T>(), &count);
  return count;
}

//
// Ranges interface for point-to-point operations
//

template<std::ranges::contiguous_range R>
int send(const R &buf, const int dest, const int tag, MPI_Comm comm = MPI_COMM_WORLD) {
  return send(std::ranges::data(buf), std::ranges::ssize(buf), dest, tag, comm);
}

template<std::ranges::contiguous_range R>
int isend(const R &buf, const int dest, const int tag, MPI_Request &request, MPI_Comm comm = MPI_COMM_WORLD) {
  return isend(std::ranges::data(buf), std::ranges::ssize(buf), dest, tag, &request, comm);
}

template<typename T, template<typename> typename Container = scalable_vector>
Container<T> probe_recv(const int source, const int tag, MPI_Status *status = MPI_STATUS_IGNORE,
                        MPI_Comm comm = MPI_COMM_WORLD) {
  const auto count = mpi::get_count<T>(mpi::probe(source, tag, comm));
  Container<T> buf(count);
  mpi::recv(buf.data(), count, source, tag, status, comm);
  return buf;
}

//
// Other MPI functions
//

inline int waitall(int count, MPI_Request *array_of_requests, MPI_Status *array_of_statuses = MPI_STATUS_IGNORE) {
  return MPI_Waitall(count, array_of_requests, array_of_statuses);
}

template<std::ranges::contiguous_range R>
int waitall(R &requests, MPI_Status *array_of_statuses = MPI_STATUS_IGNORE) {
  return MPI_Waitall(std::ranges::size(requests), std::ranges::data(requests), array_of_statuses);
}

//
// Single element interface for collective operations
//

template<typename T>
inline T bcast(T ans, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  bcast(&ans, 1, root, comm);
  return ans;
}

template<typename T>
inline T reduce(const T &element, MPI_Op op, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  T ans;
  reduce(&element, &ans, 1, op, root, comm);
  return ans;
}

template<typename T>
inline T reduce(const T &element, T &ans, MPI_Op op, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  return reduce(&element, &ans, 1, op, root, comm);
}

template<typename T>
inline T allreduce(const T &element, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
  T ans;
  allreduce(&element, &ans, 1, op, comm);
  return ans;
}

template<typename T>
int allreduce(const T &element, T &ans, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
  return allreduce(&element, &ans, 1, op, comm);
}

template<typename T, template<typename> typename Container = scalable_vector>
Container<T> gather(const T &element, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  Container<T> result;
  if (mpi::get_comm_rank(comm) == root) { result.resize(mpi::get_comm_size(comm)); }
  gather(&element, 1, std::ranges::data(result), 1, root, comm);
  return result;
}

template<std::ranges::contiguous_range R>
int gather(const std::ranges::range_value_t<R> &element, R &ans, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  LIGHT_ASSERT(mpi::get_comm_rank(comm) != root || std::ranges::size(ans) == mpi::get_comm_size(comm));

  return gather(&element, 1, std::ranges::data(ans), 1, comm);
}

template<typename T, template<typename> typename Container = scalable_vector>
Container<T> allgather(const T &element, MPI_Comm comm = MPI_COMM_WORLD) {
  Container<T> result(mpi::get_comm_size(comm));
  allgather(&element, 1, std::ranges::data(result), 1, comm);
  return result;
}

template<std::ranges::contiguous_range R>
inline int allgather(const std::ranges::range_value_t<R> &element, R &ans, MPI_Comm comm = MPI_COMM_WORLD) {
  LIGHT_ASSERT(std::ranges::size(ans) == mpi::get_comm_size(comm));

  return allgather(&element, 1, std::ranges::data(ans), 1, comm);
}

template<std::ranges::contiguous_range Rs, std::ranges::contiguous_range Rr, std::ranges::contiguous_range Rcounts,
         std::ranges::contiguous_range Displs>
int allgatherv(const Rs &sendbuf, Rr &recvbuf, const Rcounts &recvcounts, const Displs &displs,
               MPI_Comm comm = MPI_COMM_WORLD) {
  static_assert(std::is_same_v<std::ranges::range_value_t<Rcounts>, int>);
  static_assert(std::is_same_v<std::ranges::range_value_t<Displs>, int>);
  return allgatherv(std::ranges::data(sendbuf), std::ranges::ssize(sendbuf), std::ranges::data(recvbuf),
                    std::ranges::data(recvcounts), std::ranges::data(displs), comm);
}

template<typename T>
T scan(const T &sendbuf, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
  T recvbuf = T{};
  scan(&sendbuf, &recvbuf, 1, op, comm);
  return recvbuf;
}

template<typename T>
T exscan(const T &sendbuf, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
  T recvbuf = T{};
  exscan(&sendbuf, &recvbuf, 1, op, comm);
  return recvbuf;
}

//
// Ranges interface for collective operations
//

template<std::ranges::contiguous_range R>
inline int reduce(const R &sendbuf, R &recvbuf, MPI_Op op, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  LIGHT_ASSERT(mpi::get_comm_rank(comm) != root || std::ranges::size(sendbuf) == std::ranges::size(recvbuf));

  return reduce<std::ranges::range_value_t<R>>(std::ranges::cdata(sendbuf), std::ranges::data(recvbuf),
                                               std::ranges::ssize(sendbuf), op, root, comm);
}

template<std::ranges::contiguous_range R, template<typename> typename Container = scalable_vector>
inline auto reduce(const R &sendbuf, MPI_Op op, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  Container<std::ranges::range_value_t<R>> recvbuf;
  if (mpi::get_comm_rank(comm) == root) { recvbuf.resize(std::ranges::size(sendbuf)); }
  reduce(std::ranges::cdata(sendbuf), recvbuf.data(), std::ranges::ssize(sendbuf), op, root, comm);
  return recvbuf;
}

template<std::ranges::contiguous_range Rs, std::ranges::contiguous_range Rr>
inline int gather(const Rs &sendbuf, Rr &recvbuf, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  using rs_value_t = std::ranges::range_value_t<Rs>;
  using rr_value_t = std::ranges::range_value_t<Rr>;

  LIGHT_ASSERT([&] {
    const std::size_t expected = sizeof(rs_value_t) * std::ranges::size(sendbuf) * mpi::get_comm_size(comm);
    const std::size_t actual = sizeof(rr_value_t) * std::ranges::size(recvbuf);
    return mpi::get_comm_rank(comm) != root || expected >= actual;
  });

  return gather<rs_value_t, rr_value_t>(std::ranges::cdata(sendbuf), std::ranges::ssize(sendbuf),
                                        std::ranges::data(recvbuf), std::ranges::ssize(recvbuf), root, comm);
}
} // namespace dkaminpar::mpi