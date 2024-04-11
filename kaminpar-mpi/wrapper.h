/*******************************************************************************
 * Basic C++ wrapper for MPI calls.
 *
 * @file:   wrapper.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 ******************************************************************************/
#pragma once

#include <utility>

#include <mpi.h>

#include "kaminpar-mpi/datatype.h"
#include "kaminpar-mpi/definitions.h"
#include "kaminpar-mpi/utils.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/asserting_cast.h"
#include "kaminpar-common/datastructures/noinit_vector.h"
#include "kaminpar-common/parallel/algorithm.h"

namespace kaminpar::mpi {
//
// Pointer interface for collective operations
//
inline int barrier(MPI_Comm comm) {
  return MPI_Barrier(comm);
}

template <typename T> inline int bcast(T *buffer, const int count, const int root, MPI_Comm comm) {
  return MPI_Bcast(buffer, count, type::get<T>(), root, comm);
}

template <typename T>
inline int
reduce(const T *sendbuf, T *recvbuf, const int count, MPI_Op op, const int root, MPI_Comm comm) {
  return MPI_Reduce(sendbuf, recvbuf, count, type::get<T>(), op, root, comm);
}

template <typename T>
inline int allreduce(const T *sendbuf, T *recvbuf, const int count, MPI_Op op, MPI_Comm comm) {
  return MPI_Allreduce(sendbuf, recvbuf, count, type::get<T>(), op, comm);
}

template <typename Ts, typename Tr>
inline int scatter(
    const Ts *sendbuf,
    const int sendcount,
    Tr *recvbuf,
    const int recvcount,
    const int root,
    MPI_Comm comm
) {
  return MPI_Scatter(
      sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcount, type::get<Tr>(), root, comm
  );
}

template <typename Ts, typename Tr>
inline int gather(
    const Ts *sendbuf,
    const int sendcount,
    Tr *recvbuf,
    const int recvcount,
    const int root,
    MPI_Comm comm
) {
  return MPI_Gather(
      sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcount, type::get<Tr>(), root, comm
  );
}

template <typename Ts, typename Tr>
inline int
allgather(const Ts *sendbuf, const int sendcount, Tr *recvbuf, const int recvcount, MPI_Comm comm) {
  return MPI_Allgather(
      sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcount, type::get<Tr>(), comm
  );
}

template <typename Ts, typename Tr>
inline int
alltoall(const Ts *sendbuf, const int sendcount, Tr *recvbuf, const int recvcount, MPI_Comm comm) {
  return MPI_Alltoall(
      sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcount, type::get<Tr>(), comm
  );
}

template <typename Ts, typename Tr>
inline int alltoallv(
    const Ts *sendbuf,
    const int *sendcounts,
    const int *sdispls,
    Tr *recvbuf,
    const int *recvcounts,
    const int *rdispls,
    MPI_Comm comm
) {
  return MPI_Alltoallv(
      sendbuf,
      sendcounts,
      sdispls,
      type::get<Ts>(),
      recvbuf,
      recvcounts,
      rdispls,
      type::get<Tr>(),
      comm
  );
}

template <typename T>
inline int scan(const T *sendbuf, T *recvbuf, const int count, MPI_Op op, MPI_Comm comm) {
  return MPI_Scan(sendbuf, recvbuf, count, type::get<T>(), op, comm);
}

template <typename T>
inline int exscan(const T *sendbuf, T *recvbuf, const int count, MPI_Op op, MPI_Comm comm) {
  return MPI_Exscan(sendbuf, recvbuf, count, type::get<T>(), op, comm);
}

template <typename T>
inline int reduce_scatter(const T *sendbuf, T *recvbuf, int *recvcounts, MPI_Op op, MPI_Comm comm) {
  return MPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, type::get<T>(), op, comm);
}

template <typename Ts, typename Tr>
int gatherv(
    const Ts *sendbuf,
    const int sendcount,
    Tr *recvbuf,
    const int *recvcounts,
    const int *displs,
    const int root,
    MPI_Comm comm
) {
  return MPI_Gatherv(
      sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcounts, displs, type::get<Tr>(), root, comm
  );
}

template <typename Ts, typename Tr>
int allgatherv(
    const Ts *sendbuf,
    const int sendcount,
    Tr *recvbuf,
    const int *recvcounts,
    const int *displs,
    MPI_Comm comm
) {
  return MPI_Allgatherv(
      sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcounts, displs, type::get<Tr>(), comm
  );
}

//
// Pointer interface for point-to-point operations
//

template <typename T>
inline int send(const T *buf, const int count, const int dest, const int tag, MPI_Comm comm) {
  return MPI_Send(buf, count, type::get<T>(), dest, tag, comm);
}

template <typename T>
inline int isend(
    const T *buf,
    const int count,
    const int dest,
    const int tag,
    MPI_Request *request,
    MPI_Comm comm
) {
  return MPI_Isend(buf, count, type::get<T>(), dest, tag, comm, request);
}

template <typename T>
inline int recv(
    T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status = MPI_STATUS_IGNORE
) {
  return MPI_Recv(buf, count, type::get<T>(), source, tag, comm, status);
}

inline MPI_Status probe(const int source, const int tag, MPI_Comm comm) {
  MPI_Status status{};
  [[maybe_unused]] auto result = MPI_Probe(source, tag, comm, &status);
  KASSERT(result != MPI_UNDEFINED);
  return status;
}

template <typename T> inline int get_count(const MPI_Status &status) {
  int count = 0;
  [[maybe_unused]] auto result = MPI_Get_count(&status, type::get<T>(), &count);
  return count;
}

//
// Ranges interface for point-to-point operations
//

template <typename Container, std::enable_if_t<!std::is_pointer_v<Container>, bool> = true>
int send(const Container &buf, const int dest, const int tag, MPI_Comm comm) {
  return send(std::data(buf), asserting_cast<int>(std::size(buf)), dest, tag, comm);
}

template <typename Container, std::enable_if_t<!std::is_pointer_v<Container>, bool> = true>
int isend(
    const Container &buf, const int dest, const int tag, MPI_Request &request, MPI_Comm comm
) {
  return isend(std::data(buf), asserting_cast<int>(std::size(buf)), dest, tag, &request, comm);
}

template <typename T, typename Buffer = NoinitVector<T>>
Buffer
probe_recv(const int source, const int tag, MPI_Comm comm, MPI_Status *status = MPI_STATUS_IGNORE) {
  const auto count = mpi::get_count<T>(mpi::probe(source, MPI_ANY_TAG, comm));
  KASSERT(count >= 0);
  Buffer buf(count);
  mpi::recv(buf.data(), count, source, tag, comm, status);
  return buf;
}

//
// Other MPI functions
//

inline int waitall(
    int count, MPI_Request *array_of_requests, MPI_Status *array_of_statuses = MPI_STATUS_IGNORE
) {
  return MPI_Waitall(count, array_of_requests, array_of_statuses);
}

template <typename Container>
int waitall(Container &requests, MPI_Status *array_of_statuses = MPI_STATUS_IGNORE) {
  return MPI_Waitall(std::size(requests), std::data(requests), array_of_statuses);
}

//
// Single element interface for collective operations
//

template <typename T> inline T bcast(T ans, const int root, MPI_Comm comm) {
  bcast(&ans, 1, root, comm);
  return ans;
}

template <typename T, std::enable_if_t<!std::is_pointer_v<T>, bool> = true>
inline T reduce_single(const T &element, MPI_Op op, const int root, MPI_Comm comm) {
  T ans = T{};
  reduce(&element, &ans, 1, op, root, comm);
  return ans;
}

template <typename T, std::enable_if_t<!std::is_pointer_v<T>, bool> = true>
inline T reduce_single(const T &element, T &ans, MPI_Op op, const int root, MPI_Comm comm) {
  return reduce(&element, &ans, 1, op, root, comm);
}

template <typename T> inline T allreduce(const T &element, MPI_Op op, MPI_Comm comm) {
  T ans = T{};
  allreduce(&element, &ans, 1, op, comm);
  return ans;
}

template <typename T> int allreduce(const T &element, T &ans, MPI_Op op, MPI_Comm comm) {
  return allreduce(&element, &ans, 1, op, comm);
}

template <typename T, typename Container = NoinitVector<T>>
Container gather(const T &element, const int root, MPI_Comm comm) {
  Container result{};
  if (mpi::get_comm_rank(comm) == root) {
    result.resize(mpi::get_comm_size(comm));
  }
  gather(&element, 1, std::data(result), 1, root, comm);
  return result;
}

template <typename Container>
int gather(
    const typename Container::value_type &element, Container &ans, const int root, MPI_Comm comm
) {
  KASSERT(
      mpi::get_comm_rank(comm) != root || std::size(ans) == mpi::get_comm_size(comm),
      "",
      assert::light
  );
  return gather(&element, 1, std::data(ans), 1, comm);
}

template <typename T, template <typename> typename Container = NoinitVector>
Container<T> allgather(const T &element, MPI_Comm comm) {
  Container<T> result(mpi::get_comm_size(comm));
  allgather(&element, 1, std::data(result), 1, comm);
  return result;
}

template <typename Container>
inline int allgather(const typename Container::value_type &element, Container &ans, MPI_Comm comm) {
  KASSERT(std::size(ans) >= static_cast<std::size_t>(mpi::get_comm_size(comm)), "", assert::light);
  return allgather(&element, 1, std::data(ans), 1, comm);
}

template <typename Rs, typename Rr, typename Rcounts, typename Displs>
int allgatherv(
    const Rs &sendbuf, Rr &recvbuf, const Rcounts &recvcounts, const Displs &displs, MPI_Comm comm
) {
  static_assert(std::is_same_v<typename Rcounts::value_type, int>);
  static_assert(std::is_same_v<typename Displs::value_type, int>);
  return allgatherv(
      std::data(sendbuf),
      asserting_cast<int>(std::size(sendbuf)),
      std::data(recvbuf),
      std::data(recvcounts),
      std::data(displs),
      comm
  );
}

template <typename T> T scan(const T &sendbuf, MPI_Op op, MPI_Comm comm) {
  T recvbuf = T{};
  scan(&sendbuf, &recvbuf, 1, op, comm);
  return recvbuf;
}

template <typename T> T exscan(const T &sendbuf, MPI_Op op, MPI_Comm comm) {
  T recvbuf = T{};
  exscan(&sendbuf, &recvbuf, 1, op, comm);
  return recvbuf;
}

//
// Ranges interface for collective operations
//

template <typename R, std::enable_if_t<!std::is_pointer_v<R>, bool> = true>
inline int reduce(const R &sendbuf, R &recvbuf, MPI_Op op, const int root, MPI_Comm comm) {
  KASSERT(
      mpi::get_comm_rank(comm) != root || std::size(sendbuf) == std::size(recvbuf),
      "",
      assert::light
  );
  return reduce<typename R::value_type>(
      sendbuf.cdata(), std::data(recvbuf), asserting_cast<int>(std::size(sendbuf)), op, root, comm
  );
}

template <
    typename R,
    template <typename> typename Container = NoinitVector,
    std::enable_if_t<!std::is_pointer_v<R>, bool> = true>
inline auto reduce(const R &sendbuf, MPI_Op op, const int root, MPI_Comm comm) {
  Container<typename std::remove_reference_t<R>::value_type> recvbuf;
  if (mpi::get_comm_rank(comm) == root) {
    recvbuf.resize(std::size(sendbuf));
  }
  reduce(sendbuf.cdata(), recvbuf.data(), asserting_cast<int>(std::size(sendbuf)), op, root, comm);
  return recvbuf;
}

template <typename Rs, typename Rr>
inline int gather(const Rs &sendbuf, Rr &recvbuf, const int root, MPI_Comm comm) {
  using rs_value_t = typename Rs::value_type;
  using rr_value_t = typename Rr::value_type;

  KASSERT(
      [&] {
        const std::size_t expected =
            sizeof(rs_value_t) * std::size(sendbuf) * mpi::get_comm_size(comm);
        const std::size_t actual = sizeof(rr_value_t) * std::size(recvbuf);
        return mpi::get_comm_rank(comm) != root || expected >= actual;
      }(),
      "",
      assert::light
  );

  return gather<rs_value_t, rr_value_t>(
      sendbuf.cdata(),
      asserting_cast<int>(std::size(sendbuf)),
      std::data(recvbuf),
      asserting_cast<int>(std::size(recvbuf)),
      root,
      comm
  );
}

//
// Misc utility functions
//

template <typename Distribution>
inline std::vector<int> build_distribution_recvcounts(Distribution &&dist) {
  KASSERT(!dist.empty());
  std::vector<int> recvcounts(dist.size() - 1);
  for (std::size_t i = 0; i + 1 < dist.size(); ++i) {
    recvcounts[i] = dist[i + 1] - dist[i];
  }
  return recvcounts;
}

template <typename Distribution>
inline std::vector<int> build_distribution_displs(Distribution &&dist) {
  KASSERT(!dist.empty());
  std::vector<int> displs(dist.size() - 1);
  for (std::size_t i = 0; i + 1 < dist.size(); ++i) {
    displs[i] = asserting_cast<int>(dist[i]);
  }
  return displs;
}

template <typename T, template <typename> typename Container>
inline Container<T> build_distribution_from_local_count(const T value, MPI_Comm comm) {
  const auto [size, rank] = get_comm_info(comm);

  Container<T> distribution(size + 1);
  allgather(&value, 1, distribution.data() + 1, 1, comm);
  parallel::prefix_sum(distribution.begin(), distribution.end(), distribution.begin());
  distribution.front() = 0;

  return distribution;
}

template <typename T>
inline NoinitVector<int> build_counts_from_value(const T original_value, MPI_Comm comm) {
  const int value = asserting_cast<int>(original_value);

  NoinitVector<int> counts(get_comm_size(comm));
  allgather(&value, 1, counts.data(), 1, comm);
  return counts;
}

template <typename T>
inline NoinitVector<int> build_displs_from_value(const T original_value, MPI_Comm comm) {
  const int value = asserting_cast<int>(original_value);

  NoinitVector<int> displs(get_comm_size(comm) + 1);
  allgather(&value, 1, displs.data() + 1, 1, comm);
  parallel::prefix_sum(displs.begin(), displs.end(), displs.begin());
  displs.front() = 0;
  return displs;
}

inline NoinitVector<int> build_displs_from_counts(const NoinitVector<int> &counts) {
  NoinitVector<int> displs(counts.size() + 1);
  parallel::prefix_sum(counts.begin(), counts.end(), displs.begin() + 1);
  displs.front() = 0;
  return displs;
}

template <typename T>
std::tuple<T, double, T, std::int64_t> gather_statistics(const T value, MPI_Comm comm) {
  const T min = allreduce(value, MPI_MIN, comm);
  const T max = allreduce(value, MPI_MAX, comm);
  const auto sum = allreduce<std::int64_t>(value, MPI_SUM, comm);
  const double avg = 1.0 * sum / get_comm_size(comm);
  return {min, avg, max, sum};
}

template <typename T> std::string gather_statistics_str(const T value, MPI_Comm comm) {
  std::ostringstream os;
  const auto [min, avg, max, sum] = gather_statistics(value, comm);
  os << "min=" << min << "|avg=" << avg << "|max=" << max << "|sum=" << sum;
  return os.str();
}
} // namespace kaminpar::mpi
