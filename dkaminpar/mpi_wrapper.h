/*******************************************************************************
 * @file:   mpi_wrapper.h
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  C++ wrapper for MPI calls.
 ******************************************************************************/
#pragma once

#include "dkaminpar/distributed_definitions.h"
#include "kaminpar/utility/timer.h"

#include <concepts>
#include <mpi.h>
#include <ranges>
#include <type_traits>
#include <utility>

#define SPARSE_ALLTOALL_NOFILTER [](NodeID) { return true; }

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
template <std::size_t N> inline MPI_Datatype custom() {
  static MPI_Datatype type = MPI_DATATYPE_NULL;
  if (type == MPI_DATATYPE_NULL) {
    MPI_Type_contiguous(N, MPI_CHAR, &type);
    MPI_Type_commit(&type);
  }
  return type;
}

// Map to default MPI type
#define MAP_DATATYPE(CPP_DATATYPE, MPI_DATATYPE)                                                                       \
  template <std::same_as<CPP_DATATYPE>> inline MPI_Datatype get() { return MPI_DATATYPE; }

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
// MAP_DATATYPE(std::size_t, MPI_UNSIGNED_LONG)

#define COMMA ,
MAP_DATATYPE(std::pair<float COMMA int>, MPI_FLOAT_INT)
MAP_DATATYPE(std::pair<double COMMA int>, MPI_DOUBLE_INT)
MAP_DATATYPE(std::pair<long double COMMA int>, MPI_LONG_DOUBLE_INT)
#undef COMMA

#undef MAP_DATATYPE

// Fallback to custom MPI type
template <typename T> inline MPI_Datatype get() { return custom<sizeof(T)>(); }
} // namespace type

//
// Pointer interface for collective operations
//

inline int barrier(MPI_Comm comm = MPI_COMM_WORLD) { return MPI_Barrier(comm); }

template <typename T> inline int bcast(T *buffer, const int count, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Bcast(buffer, count, type::get<T>(), root, comm);
}

template <typename T>
inline int reduce(const T *sendbuf, T *recvbuf, const int count, MPI_Op op, const int root = 0,
                  MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Reduce(sendbuf, recvbuf, count, type::get<T>(), op, root, comm);
}

template <typename T>
inline int allreduce(const T *sendbuf, T *recvbuf, const int count, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Allreduce(sendbuf, recvbuf, count, type::get<T>(), op, comm);
}

template <typename Ts, typename Tr>
inline int scatter(const Ts *sendbuf, const int sendcount, Tr *recvbuf, const int recvcount, const int root,
                   MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Scatter(sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcount, type::get<Tr>(), root, comm);
}

template <typename Ts, typename Tr>
inline int gather(const Ts *sendbuf, const int sendcount, Tr *recvbuf, const int recvcount, const int root = 0,
                  MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Gather(sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcount, type::get<Tr>(), root, comm);
}

template <typename Ts, typename Tr>
inline int allgather(const Ts *sendbuf, const int sendcount, Tr *recvbuf, const int recvcount,
                     MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Allgather(sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcount, type::get<Tr>(), comm);
}

template <typename Ts, typename Tr>
inline int alltoall(const Ts *sendbuf, const int sendcount, Tr *recvbuf, const int recvcount,
                    MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Alltoall(sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcount, type::get<Tr>(), comm);
}

template <typename T>
inline int scan(const T *sendbuf, T *recvbuf, const int count, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Scan(sendbuf, recvbuf, count, type::get<T>(), op, comm);
}

template <typename T>
inline int exscan(const T *sendbuf, T *recvbuf, const int count, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Exscan(sendbuf, recvbuf, count, type::get<T>(), op, comm);
}

template <typename T>
inline int reduce_scatter(const T *sendbuf, T *recvbuf, int *recvcounts, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, type::get<T>(), op, comm);
}

template <typename Ts, typename Tr>
int gatherv(const Ts *sendbuf, const int sendcount, Tr *recvbuf, const int *recvcounts, const int *displs,
            const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Gatherv(sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcounts, displs, type::get<Tr>(), root, comm);
}

template <typename Ts, typename Tr>
int allgatherv(const Ts *sendbuf, const int sendcount, Tr *recvbuf, const int *recvcounts, const int *displs,
               MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Allgatherv(sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcounts, displs, type::get<Tr>(), comm);
}

//
// Pointer interface for point-to-point operations
//

template <typename T>
inline int send(const T *buf, const int count, const int dest, const int tag, MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Send(buf, count, type::get<T>(), dest, tag, comm);
}

template <typename T>
inline int isend(const T *buf, const int count, const int dest, const int tag, MPI_Request *request,
                 MPI_Comm comm = MPI_COMM_WORLD) {
  return MPI_Isend(buf, count, type::get<T>(), dest, tag, comm, request);
}

template <typename T>
inline int recv(T *buf, int count, int source, int tag, MPI_Status *status = MPI_STATUS_IGNORE,
                MPI_Comm const comm = MPI_COMM_WORLD) {
  return MPI_Recv(buf, count, type::get<T>(), source, tag, comm, status);
}

inline MPI_Status probe(const int source, const int tag, MPI_Comm comm = MPI_COMM_WORLD) {
  MPI_Status status;
  [[maybe_unused]] auto result = MPI_Probe(source, tag, comm, &status);
  ASSERT(result != MPI_UNDEFINED) << V(source) << V(tag);
  return status;
}

template <typename T> inline int get_count(const MPI_Status &status) {
  int count;
  [[maybe_unused]] auto result = MPI_Get_count(&status, type::get<T>(), &count);
  return count;
}

//
// Ranges interface for point-to-point operations
//

template <std::ranges::contiguous_range R>
int send(const R &buf, const int dest, const int tag, MPI_Comm comm = MPI_COMM_WORLD) {
  return send(std::ranges::data(buf), std::ranges::ssize(buf), dest, tag, comm);
}

template <std::ranges::contiguous_range R>
int isend(const R &buf, const int dest, const int tag, MPI_Request &request, MPI_Comm comm = MPI_COMM_WORLD) {
  return isend(std::ranges::data(buf), std::ranges::ssize(buf), dest, tag, &request, comm);
}

template <typename T, typename Buffer = scalable_noinit_vector<T>>
Buffer probe_recv(const int source, const int tag, MPI_Status *status = MPI_STATUS_IGNORE,
                  MPI_Comm comm = MPI_COMM_WORLD) {
  const auto count = mpi::get_count<T>(mpi::probe(source, MPI_ANY_TAG, comm));
  ASSERT(count >= 0) << V(source) << V(tag);
  Buffer buf(count);
  mpi::recv(buf.data(), count, source, tag, status, comm);
  return buf;
}

//
// Other MPI functions
//

inline int waitall(int count, MPI_Request *array_of_requests, MPI_Status *array_of_statuses = MPI_STATUS_IGNORE) {
  return MPI_Waitall(count, array_of_requests, array_of_statuses);
}

template <std::ranges::contiguous_range R> int waitall(R &requests, MPI_Status *array_of_statuses = MPI_STATUS_IGNORE) {
  return MPI_Waitall(std::ranges::size(requests), std::ranges::data(requests), array_of_statuses);
}

//
// Single element interface for collective operations
//

template <typename T> inline T bcast(T ans, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  bcast(&ans, 1, root, comm);
  return ans;
}

template <typename T> inline T reduce(const T &element, MPI_Op op, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  T ans = T{};
  reduce(&element, &ans, 1, op, root, comm);
  return ans;
}

template <typename T>
inline T reduce(const T &element, T &ans, MPI_Op op, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  return reduce(&element, &ans, 1, op, root, comm);
}

template <typename T> inline T allreduce(const T &element, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
  T ans = T{};
  allreduce(&element, &ans, 1, op, comm);
  return ans;
}

template <typename T> int allreduce(const T &element, T &ans, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
  return allreduce(&element, &ans, 1, op, comm);
}

template <typename T, template <typename> typename Container = scalable_vector>
Container<T> gather(const T &element, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  Container<T> result;
  if (mpi::get_comm_rank(comm) == root) {
    result.resize(mpi::get_comm_size(comm));
  }
  gather(&element, 1, std::ranges::data(result), 1, root, comm);
  return result;
}

template <std::ranges::contiguous_range R>
int gather(const std::ranges::range_value_t<R> &element, R &ans, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  LIGHT_ASSERT(mpi::get_comm_rank(comm) != root || std::ranges::size(ans) == mpi::get_comm_size(comm));

  return gather(&element, 1, std::ranges::data(ans), 1, comm);
}

template <typename T, template <typename> typename Container = scalable_vector>
Container<T> allgather(const T &element, MPI_Comm comm = MPI_COMM_WORLD) {
  Container<T> result(mpi::get_comm_size(comm));
  allgather(&element, 1, std::ranges::data(result), 1, comm);
  return result;
}

template <std::ranges::contiguous_range R>
inline int allgather(const std::ranges::range_value_t<R> &element, R &ans, MPI_Comm comm = MPI_COMM_WORLD) {
  LIGHT_ASSERT(std::ranges::size(ans) >= static_cast<std::size_t>(mpi::get_comm_size(comm)));

  return allgather(&element, 1, std::ranges::data(ans), 1, comm);
}

template <std::ranges::contiguous_range Rs, std::ranges::contiguous_range Rr, std::ranges::contiguous_range Rcounts,
          std::ranges::contiguous_range Displs>
int allgatherv(const Rs &sendbuf, Rr &recvbuf, const Rcounts &recvcounts, const Displs &displs,
               MPI_Comm comm = MPI_COMM_WORLD) {
  static_assert(std::is_same_v<std::ranges::range_value_t<Rcounts>, int>);
  static_assert(std::is_same_v<std::ranges::range_value_t<Displs>, int>);
  return allgatherv(std::ranges::data(sendbuf), std::ranges::ssize(sendbuf), std::ranges::data(recvbuf),
                    std::ranges::data(recvcounts), std::ranges::data(displs), comm);
}

template <typename T> T scan(const T &sendbuf, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
  T recvbuf = T{};
  scan(&sendbuf, &recvbuf, 1, op, comm);
  return recvbuf;
}

template <typename T> T exscan(const T &sendbuf, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
  T recvbuf = T{};
  exscan(&sendbuf, &recvbuf, 1, op, comm);
  return recvbuf;
}

//
// Ranges interface for collective operations
//

template <std::ranges::contiguous_range R>
inline int reduce(const R &sendbuf, R &recvbuf, MPI_Op op, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  LIGHT_ASSERT(mpi::get_comm_rank(comm) != root || std::ranges::size(sendbuf) == std::ranges::size(recvbuf));

  return reduce<std::ranges::range_value_t<R>>(std::ranges::cdata(sendbuf), std::ranges::data(recvbuf),
                                               std::ranges::ssize(sendbuf), op, root, comm);
}

template <std::ranges::contiguous_range R, template <typename> typename Container = scalable_vector>
inline auto reduce(const R &sendbuf, MPI_Op op, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
  Container<std::ranges::range_value_t<R>> recvbuf;
  if (mpi::get_comm_rank(comm) == root) {
    recvbuf.resize(std::ranges::size(sendbuf));
  }
  reduce(std::ranges::cdata(sendbuf), recvbuf.data(), std::ranges::ssize(sendbuf), op, root, comm);
  return recvbuf;
}

template <std::ranges::contiguous_range Rs, std::ranges::contiguous_range Rr>
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

//
// Misc utility functions
//

template <typename Lambda> inline void sequentially(Lambda &&lambda, MPI_Comm comm = MPI_COMM_WORLD) {
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

template <std::ranges::range Distribution> inline std::vector<int> build_distribution_recvcounts(Distribution &&dist) {
  ASSERT(!dist.empty());
  std::vector<int> recvcounts(dist.size() - 1);
  for (std::size_t i = 0; i + 1 < dist.size(); ++i) {
    recvcounts[i] = dist[i + 1] - dist[i];
  }
  return recvcounts;
}

template <std::ranges::range Distribution> inline std::vector<int> build_distribution_displs(Distribution &&dist) {
  ASSERT(!dist.empty());
  std::vector<int> displs(dist.size() - 1);
  for (std::size_t i = 0; i + 1 < dist.size(); ++i) {
    displs[i] = static_cast<int>(dist[i]);
  }
  return displs;
}

template <typename T, template <typename> typename Container>
inline Container<T> build_distribution_from_local_count(const T value, MPI_Comm comm) {
  const auto [size, rank] = mpi::get_comm_info(comm);

  Container<T> distribution(size + 1);
  mpi::allgather(&value, 1, distribution.data() + 1, 1, comm);
  shm::parallel::prefix_sum(distribution.begin(), distribution.end(), distribution.begin());
  distribution.front() = 0;

  return distribution;
}

template <typename Message, typename Buffer = scalable_noinit_vector<Message>>
void sparse_alltoall(std::vector<Buffer> &&send_buffers, auto &&receiver, MPI_Comm comm) {
  SCOPED_TIMER("Sparse AllToAll Move", TIMER_FINE);

  using Receiver = decltype(receiver);
  constexpr bool receiver_invocable_with_pe = std::is_invocable_r_v<void, Receiver, Buffer, PEID>;
  constexpr bool receiver_invocable_without_pe = std::is_invocable_r_v<void, Receiver, Buffer>;
  static_assert(receiver_invocable_with_pe || receiver_invocable_without_pe, "bad receiver type");

  const auto [size, rank] = mpi::get_comm_info(comm);

  std::vector<MPI_Request> requests(size - 1);

  std::size_t next_req_index = 0;
  for (PEID pe = 0; pe < size; ++pe) {
    if (pe != rank) {
      ASSERT(static_cast<std::size_t>(pe) < send_buffers.size()) << V(pe) << V(send_buffers.size());
      ASSERT(next_req_index < requests.size());
      mpi::isend(send_buffers[pe], pe, 0, requests[next_req_index++], comm);
    }
  }
  ASSERT(next_req_index == requests.size());

  for (PEID pe = 0; pe < size; ++pe) {
    if (pe == rank) {
      if constexpr (receiver_invocable_with_pe) {
        receiver(std::move(send_buffers[rank]), pe);
      } else {
        receiver(std::move(send_buffers[rank]));
      }
    } else {
      const auto recv_buffer = mpi::probe_recv<Message, Buffer>(pe, 0, MPI_STATUS_IGNORE, comm);
      if constexpr (receiver_invocable_with_pe) {
        receiver(std::move(recv_buffer), pe);
      } else /* if (receiver_invocable_without_pe) */ {
        receiver(std::move(recv_buffer));
      }
    }
  }

  mpi::waitall(requests);
}

template <typename Message, typename Buffer = scalable_noinit_vector<Message>>
void sparse_alltoall(const std::vector<Buffer> &send_buffers, auto &&receiver, MPI_Comm comm, const bool self) {
  SCOPED_TIMER("Sparse AllToAll", TIMER_FINE);

  using Receiver = decltype(receiver);
  constexpr bool receiver_invocable_with_pe = std::is_invocable_r_v<void, Receiver, Buffer, PEID>;
  constexpr bool receiver_invocable_without_pe = std::is_invocable_r_v<void, Receiver, Buffer>;
  static_assert(receiver_invocable_with_pe || receiver_invocable_without_pe, "bad receiver type");

  const auto [size, rank] = mpi::get_comm_info(comm);

  std::vector<MPI_Request> requests(size - 1 + self);

  std::size_t next_req_index = 0;
  for (PEID pe = 0; pe < size; ++pe) {
    if (self || pe != rank) {
      ASSERT(static_cast<std::size_t>(pe) < send_buffers.size()) << V(pe) << V(send_buffers.size());
      ASSERT(next_req_index < requests.size());
      mpi::isend(send_buffers[pe], pe, 0, requests[next_req_index++], comm);
    }
  }
  ASSERT(next_req_index == requests.size());

  for (PEID pe = 0; pe < size; ++pe) {
    if (self || pe != rank) {
      const auto recv_buffer = mpi::probe_recv<Message, Buffer>(pe, 0, MPI_STATUS_IGNORE, comm);
      if constexpr (receiver_invocable_with_pe) {
        receiver(std::move(recv_buffer), pe);
      } else /* if (receiver_invocable_without_pe) */ {
        receiver(std::move(recv_buffer));
      }
    }
  }

  mpi::waitall(requests);
}

template <typename Message, typename Buffer = scalable_noinit_vector<Message>>
std::vector<Buffer> sparse_alltoall_get(std::vector<Buffer> &&send_buffers, MPI_Comm comm) {
  std::vector<Buffer> recv_buffers(mpi::get_comm_size(comm));
  sparse_alltoall<Message, Buffer>(
      std::move(send_buffers),
      [&](const auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); }, comm);
  return recv_buffers;
}

template <typename Message, typename Buffer = scalable_noinit_vector<Message>>
std::vector<Buffer> sparse_alltoall_get(const std::vector<Buffer> &send_buffers, MPI_Comm comm, const bool self) {
  std::vector<Buffer> recv_buffers(mpi::get_comm_size(comm));
  sparse_alltoall<Message, Buffer>(
      send_buffers, [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); }, comm, self);
  return recv_buffers;
}

template <typename T> std::tuple<T, double, T, T> gather_statistics(const T value, MPI_Comm comm = MPI_COMM_WORLD) {
  const T min = mpi::allreduce(value, MPI_MIN, comm);
  const T max = mpi::allreduce(value, MPI_MAX, comm);
  const T sum = mpi::allreduce(value, MPI_SUM, comm);
  const double avg = 1.0 * sum / mpi::get_comm_size(comm);
  return {min, avg, max, sum};
}

template <typename T> std::string gather_statistics_str(const T value, MPI_Comm comm = MPI_COMM_WORLD) {
  std::ostringstream os;
  const auto [min, avg, max, sum] = gather_statistics(value, comm);
  os << "min=" << min << "|avg=" << std::setw(3) << avg << "|max=" << max << "|sum=" << sum;
  return os.str();
}
} // namespace dkaminpar::mpi
