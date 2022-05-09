/*******************************************************************************
 * @file:   mpi_wrapper.h
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  C++ wrapper for MPI calls.
 ******************************************************************************/
#pragma once

#include <type_traits>
#include <utility>

#include <mpi.h>

#include <kassert/kassert.hpp>

#include "dkaminpar/definitions.h"
#include "kaminpar/parallel/algorithm.h"
#include "kaminpar/utils/timer.h"

#define SPARSE_ALLTOALL_NOFILTER \
    [](NodeID) {                 \
        return true;             \
    }

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
template <std::size_t N>
inline MPI_Datatype custom() {
    static MPI_Datatype type = MPI_DATATYPE_NULL;
    if (type == MPI_DATATYPE_NULL) {
        MPI_Type_contiguous(N, MPI_CHAR, &type);
        MPI_Type_commit(&type);
    }
    return type;
}

// Map to default MPI type
template <typename T>
inline MPI_Datatype get() {
    if constexpr (std::is_same_v<T, std::uint8_t>) {
        return MPI_UINT8_T;
    } else if constexpr (std::is_same_v<T, std::int8_t>) {
        return MPI_INT8_T;
    } else if constexpr (std::is_same_v<T, std::uint16_t>) {
        return MPI_UINT16_T;
    } else if constexpr (std::is_same_v<T, std::int16_t>) {
        return MPI_INT16_T;
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        return MPI_UINT32_T;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        return MPI_INT32_T;
    } else if constexpr (std::is_same_v<T, std::uint64_t>) {
        return MPI_UINT64_T;
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
        return MPI_INT64_T;
    } else if constexpr (std::is_same_v<T, float>) {
        return MPI_FLOAT;
    } else if constexpr (std::is_same_v<T, double>) {
        return MPI_DOUBLE;
    } else if constexpr (std::is_same_v<T, long double>) {
        return MPI_LONG_DOUBLE;
    } else if constexpr (std::is_same_v<T, std::pair<float, int>>) {
        return MPI_FLOAT_INT;
    } else if constexpr (std::is_same_v<T, std::pair<double, int>>) {
        return MPI_DOUBLE_INT;
    } else if constexpr (std::is_same_v<T, std::pair<long double, int>>) {
        return MPI_LONG_DOUBLE_INT;
    } else {
        return custom<sizeof(T)>();
    }
}
} // namespace type

//
// Pointer interface for collective operations
//

inline int barrier(MPI_Comm comm = MPI_COMM_WORLD) {
    return MPI_Barrier(comm);
}

template <typename T>
inline int bcast(T* buffer, const int count, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
    return MPI_Bcast(buffer, count, type::get<T>(), root, comm);
}

template <typename T>
inline int
reduce(const T* sendbuf, T* recvbuf, const int count, MPI_Op op, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
    return MPI_Reduce(sendbuf, recvbuf, count, type::get<T>(), op, root, comm);
}

template <typename T>
inline int allreduce(const T* sendbuf, T* recvbuf, const int count, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
    return MPI_Allreduce(sendbuf, recvbuf, count, type::get<T>(), op, comm);
}

template <typename Ts, typename Tr>
inline int scatter(
    const Ts* sendbuf, const int sendcount, Tr* recvbuf, const int recvcount, const int root,
    MPI_Comm comm = MPI_COMM_WORLD) {
    return MPI_Scatter(sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcount, type::get<Tr>(), root, comm);
}

template <typename Ts, typename Tr>
inline int gather(
    const Ts* sendbuf, const int sendcount, Tr* recvbuf, const int recvcount, const int root = 0,
    MPI_Comm comm = MPI_COMM_WORLD) {
    return MPI_Gather(sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcount, type::get<Tr>(), root, comm);
}

template <typename Ts, typename Tr>
inline int
allgather(const Ts* sendbuf, const int sendcount, Tr* recvbuf, const int recvcount, MPI_Comm comm = MPI_COMM_WORLD) {
    return MPI_Allgather(sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcount, type::get<Tr>(), comm);
}

template <typename Ts, typename Tr>
inline int
alltoall(const Ts* sendbuf, const int sendcount, Tr* recvbuf, const int recvcount, MPI_Comm comm = MPI_COMM_WORLD) {
    return MPI_Alltoall(sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcount, type::get<Tr>(), comm);
}

template <typename Ts, typename Tr>
inline int alltoallv(
    const Ts* sendbuf, const int* sendcounts, const int* sdispls, Tr* recvbuf, const int* recvcounts,
    const int* rdispls, MPI_Comm comm = MPI_COMM_WORLD) {
    return MPI_Alltoallv(
        sendbuf, sendcounts, sdispls, type::get<Ts>(), recvbuf, recvcounts, rdispls, type::get<Tr>(), comm);
}

template <typename T>
inline int scan(const T* sendbuf, T* recvbuf, const int count, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
    return MPI_Scan(sendbuf, recvbuf, count, type::get<T>(), op, comm);
}

template <typename T>
inline int exscan(const T* sendbuf, T* recvbuf, const int count, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
    return MPI_Exscan(sendbuf, recvbuf, count, type::get<T>(), op, comm);
}

template <typename T>
inline int reduce_scatter(const T* sendbuf, T* recvbuf, int* recvcounts, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
    return MPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, type::get<T>(), op, comm);
}

template <typename Ts, typename Tr>
int gatherv(
    const Ts* sendbuf, const int sendcount, Tr* recvbuf, const int* recvcounts, const int* displs, const int root = 0,
    MPI_Comm comm = MPI_COMM_WORLD) {
    return MPI_Gatherv(sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcounts, displs, type::get<Tr>(), root, comm);
}

template <typename Ts, typename Tr>
int allgatherv(
    const Ts* sendbuf, const int sendcount, Tr* recvbuf, const int* recvcounts, const int* displs,
    MPI_Comm comm = MPI_COMM_WORLD) {
    return MPI_Allgatherv(sendbuf, sendcount, type::get<Ts>(), recvbuf, recvcounts, displs, type::get<Tr>(), comm);
}

//
// Pointer interface for point-to-point operations
//

template <typename T>
inline int send(const T* buf, const int count, const int dest, const int tag, MPI_Comm comm = MPI_COMM_WORLD) {
    return MPI_Send(buf, count, type::get<T>(), dest, tag, comm);
}

template <typename T>
inline int isend(
    const T* buf, const int count, const int dest, const int tag, MPI_Request* request,
    MPI_Comm comm = MPI_COMM_WORLD) {
    return MPI_Isend(buf, count, type::get<T>(), dest, tag, comm, request);
}

template <typename T>
inline int recv(
    T* buf, int count, int source, int tag, MPI_Status* status = MPI_STATUS_IGNORE,
    MPI_Comm const comm = MPI_COMM_WORLD) {
    return MPI_Recv(buf, count, type::get<T>(), source, tag, comm, status);
}

inline MPI_Status probe(const int source, const int tag, MPI_Comm comm = MPI_COMM_WORLD) {
    MPI_Status            status;
    [[maybe_unused]] auto result = MPI_Probe(source, tag, comm, &status);
    KASSERT(result != MPI_UNDEFINED);
    return status;
}

template <typename T>
inline int get_count(const MPI_Status& status) {
    int                   count;
    [[maybe_unused]] auto result = MPI_Get_count(&status, type::get<T>(), &count);
    return count;
}

//
// Ranges interface for point-to-point operations
//

template <typename Container, std::enable_if_t<!std::is_pointer_v<Container>, bool> = true>
int send(const Container& buf, const int dest, const int tag, MPI_Comm comm = MPI_COMM_WORLD) {
    return send(std::data(buf), static_cast<int>(std::size(buf)), dest, tag, comm);
}

template <typename Container, std::enable_if_t<!std::is_pointer_v<Container>, bool> = true>
int isend(const Container& buf, const int dest, const int tag, MPI_Request& request, MPI_Comm comm = MPI_COMM_WORLD) {
    return isend(std::data(buf), static_cast<int>(std::size(buf)), dest, tag, &request, comm);
}

template <typename T, typename Buffer = scalable_noinit_vector<T>>
Buffer
probe_recv(const int source, const int tag, MPI_Status* status = MPI_STATUS_IGNORE, MPI_Comm comm = MPI_COMM_WORLD) {
    const auto count = mpi::get_count<T>(mpi::probe(source, MPI_ANY_TAG, comm));
    KASSERT(count >= 0);
    Buffer buf(count);
    mpi::recv(buf.data(), count, source, tag, status, comm);
    return buf;
}

//
// Other MPI functions
//

inline int waitall(int count, MPI_Request* array_of_requests, MPI_Status* array_of_statuses = MPI_STATUS_IGNORE) {
    return MPI_Waitall(count, array_of_requests, array_of_statuses);
}

template <typename Container>
int waitall(Container& requests, MPI_Status* array_of_statuses = MPI_STATUS_IGNORE) {
    return MPI_Waitall(std::size(requests), std::data(requests), array_of_statuses);
}

//
// Single element interface for collective operations
//

template <typename T>
inline T bcast(T ans, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
    bcast(&ans, 1, root, comm);
    return ans;
}

template <typename T, std::enable_if_t<!std::is_pointer_v<T>, bool> = true>
inline T reduce_single(const T& element, MPI_Op op, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
    T ans = T{};
    reduce(&element, &ans, 1, op, root, comm);
    return ans;
}

template <typename T, std::enable_if_t<!std::is_pointer_v<T>, bool> = true>
inline T reduce_single(const T& element, T& ans, MPI_Op op, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
    return reduce(&element, &ans, 1, op, root, comm);
}

template <typename T>
inline T allreduce(const T& element, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
    T ans = T{};
    allreduce(&element, &ans, 1, op, comm);
    return ans;
}

template <typename T>
int allreduce(const T& element, T& ans, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
    return allreduce(&element, &ans, 1, op, comm);
}

template <typename T, typename Container = scalable_vector<T>>
Container gather(const T& element, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
    Container result;
    if (mpi::get_comm_rank(comm) == root) {
        result.resize(mpi::get_comm_size(comm));
    }
    gather(&element, 1, std::data(result), 1, root, comm);
    return result;
}

template <typename Container>
int gather(
    const typename Container::value_type& element, Container& ans, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
    KASSERT((mpi::get_comm_rank(comm) != root || std::size(ans) == mpi::get_comm_size(comm)), "", assert::light);

    return gather(&element, 1, std::data(ans), 1, comm);
}

template <typename T, template <typename> typename Container = scalable_vector>
Container<T> allgather(const T& element, MPI_Comm comm = MPI_COMM_WORLD) {
    Container<T> result(mpi::get_comm_size(comm));
    allgather(&element, 1, std::data(result), 1, comm);
    return result;
}

template <typename Container>
inline int allgather(const typename Container::value_type& element, Container& ans, MPI_Comm comm = MPI_COMM_WORLD) {
    KASSERT(std::size(ans) >= static_cast<std::size_t>(mpi::get_comm_size(comm)), "", assert::light);

    return allgather(&element, 1, std::data(ans), 1, comm);
}

template <typename Rs, typename Rr, typename Rcounts, typename Displs>
int allgatherv(
    const Rs& sendbuf, Rr& recvbuf, const Rcounts& recvcounts, const Displs& displs, MPI_Comm comm = MPI_COMM_WORLD) {
    static_assert(std::is_same_v<typename Rcounts::value_type, int>);
    static_assert(std::is_same_v<typename Displs::value_type, int>);
    return allgatherv(
        std::data(sendbuf), static_cast<int>(std::size(sendbuf)), std::data(recvbuf), std::data(recvcounts),
        std::data(displs), comm);
}

template <typename T>
T scan(const T& sendbuf, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
    T recvbuf = T{};
    scan(&sendbuf, &recvbuf, 1, op, comm);
    return recvbuf;
}

template <typename T>
T exscan(const T& sendbuf, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
    T recvbuf = T{};
    exscan(&sendbuf, &recvbuf, 1, op, comm);
    return recvbuf;
}

//
// Ranges interface for collective operations
//

template <typename R, std::enable_if_t<!std::is_pointer_v<R>, bool> = true>
inline int reduce(const R& sendbuf, R& recvbuf, MPI_Op op, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
    KASSERT((mpi::get_comm_rank(comm) != root || std::size(sendbuf) == std::size(recvbuf)), "", assert::light);

    return reduce<typename R::value_type>(
        sendbuf.cdata(), std::data(recvbuf), static_cast<int>(std::size(sendbuf)), op, root, comm);
}

template <
    typename R, template <typename> typename Container = scalable_vector,
    std::enable_if_t<!std::is_pointer_v<R>, bool> = true>
inline auto reduce(const R& sendbuf, MPI_Op op, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
    Container<typename std::remove_reference_t<R>::value_type> recvbuf;
    if (mpi::get_comm_rank(comm) == root) {
        recvbuf.resize(std::size(sendbuf));
    }
    reduce(sendbuf.cdata(), recvbuf.data(), static_cast<int>(std::size(sendbuf)), op, root, comm);
    return recvbuf;
}

template <typename Rs, typename Rr>
inline int gather(const Rs& sendbuf, Rr& recvbuf, const int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
    using rs_value_t = typename Rs::value_type;
    using rr_value_t = typename Rr::value_type;

    KASSERT(
        [&] {
            const std::size_t expected = sizeof(rs_value_t) * std::size(sendbuf) * mpi::get_comm_size(comm);
            const std::size_t actual   = sizeof(rr_value_t) * std::size(recvbuf);
            return mpi::get_comm_rank(comm) != root || expected >= actual;
        }(),
        "", assert::light);

    return gather<rs_value_t, rr_value_t>(
        sendbuf.cdata(), static_cast<int>(std::size(sendbuf)), std::data(recvbuf), static_cast<int>(std::size(recvbuf)),
        root, comm);
}

//
// Misc utility functions
//

template <typename Lambda>
inline void sequentially(Lambda&& lambda, MPI_Comm comm = MPI_COMM_WORLD) {
    const auto [size, rank] = get_comm_info();
    for (int p = 0; p < size; ++p) {
        if (p == rank) {
            lambda(p);
        }
        MPI_Barrier(comm);
    }
}

template <typename Distribution>
inline std::vector<int> build_distribution_recvcounts(Distribution&& dist) {
    KASSERT(!dist.empty());
    std::vector<int> recvcounts(dist.size() - 1);
    for (std::size_t i = 0; i + 1 < dist.size(); ++i) {
        recvcounts[i] = dist[i + 1] - dist[i];
    }
    return recvcounts;
}

template <typename Distribution>
inline std::vector<int> build_distribution_displs(Distribution&& dist) {
    KASSERT(!dist.empty());
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

template <typename Message, typename Buffer = scalable_noinit_vector<Message>, typename Receiver>
void sparse_alltoall(std::vector<Buffer>&& send_buffers, Receiver&& receiver, MPI_Comm comm) {
    SCOPED_TIMER("Sparse AllToAll", TIMER_DETAIL);

    constexpr bool receiver_invocable_with_pe    = std::is_invocable_r_v<void, Receiver, Buffer, PEID>;
    constexpr bool receiver_invocable_without_pe = std::is_invocable_r_v<void, Receiver, Buffer>;
    static_assert(receiver_invocable_with_pe || receiver_invocable_without_pe, "bad receiver type");

    const auto [size, rank] = mpi::get_comm_info(comm);

    std::vector<MPI_Request> requests(size - 1);

    std::size_t next_req_index = 0;
    for (PEID pe = 0; pe < size; ++pe) {
        if (pe != rank) {
            KASSERT(static_cast<std::size_t>(pe) < send_buffers.size());
            KASSERT(next_req_index < requests.size());
            mpi::isend(send_buffers[pe], pe, 0, requests[next_req_index++], comm);
        }
    }
    KASSERT(next_req_index == requests.size());

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

template <typename Message, typename Buffer = scalable_noinit_vector<Message>, typename Receiver>
void sparse_alltoall(const std::vector<Buffer>& send_buffers, Receiver&& receiver, MPI_Comm comm, const bool self) {
    SCOPED_TIMER("Sparse AllToAll", TIMER_DETAIL);

    constexpr bool receiver_invocable_with_pe    = std::is_invocable_r_v<void, Receiver, Buffer, PEID>;
    constexpr bool receiver_invocable_without_pe = std::is_invocable_r_v<void, Receiver, Buffer>;
    static_assert(receiver_invocable_with_pe || receiver_invocable_without_pe, "bad receiver type");

    const auto [size, rank] = mpi::get_comm_info(comm);

    std::vector<MPI_Request> requests(size - 1 + self);

    std::size_t next_req_index = 0;
    for (PEID pe = 0; pe < size; ++pe) {
        if (self || pe != rank) {
            KASSERT(static_cast<std::size_t>(pe) < send_buffers.size());
            KASSERT(next_req_index < requests.size());
            mpi::isend(send_buffers[pe], pe, 0, requests[next_req_index++], comm);
        }
    }
    KASSERT(next_req_index == requests.size());

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
std::vector<Buffer> sparse_alltoall_get(std::vector<Buffer>&& send_buffers, MPI_Comm comm) {
    std::vector<Buffer> recv_buffers(mpi::get_comm_size(comm));
    sparse_alltoall<Message, Buffer>(
        std::move(send_buffers),
        [&](const auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); }, comm);
    return recv_buffers;
}

template <typename Message, typename Buffer = scalable_noinit_vector<Message>>
std::vector<Buffer> sparse_alltoall_get(const std::vector<Buffer>& send_buffers, MPI_Comm comm, const bool self) {
    std::vector<Buffer> recv_buffers(mpi::get_comm_size(comm));
    sparse_alltoall<Message, Buffer>(
        send_buffers, [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); }, comm, self);
    return recv_buffers;
}

template <typename T>
std::tuple<T, double, T, T> gather_statistics(const T value, MPI_Comm comm = MPI_COMM_WORLD) {
    const T      min = mpi::allreduce(value, MPI_MIN, comm);
    const T      max = mpi::allreduce(value, MPI_MAX, comm);
    const T      sum = mpi::allreduce(value, MPI_SUM, comm);
    const double avg = 1.0 * sum / mpi::get_comm_size(comm);
    return {min, avg, max, sum};
}

template <typename T>
std::string gather_statistics_str(const T value, MPI_Comm comm = MPI_COMM_WORLD) {
    std::ostringstream os;
    const auto [min, avg, max, sum] = gather_statistics(value, comm);
    os << "min=" << min << "|avg=" << avg << "|max=" << max << "|sum=" << sum;
    return os.str();
}
} // namespace dkaminpar::mpi
