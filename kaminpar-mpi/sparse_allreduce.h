/*******************************************************************************
 * Allreduce for sparse key-value-pairs.
 *
 * @file:   sparse_allreduce.h
 * @author: Daniel Seemaier
 * @date:   27.03.2023
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <utility>

#include <mpi.h>
#include <tbb/parallel_for.h>

#include "kaminpar-mpi/definitions.h"
#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/noinit_vector.h"
#include "kaminpar-common/math.h"

namespace kaminpar::mpi {
namespace tag {
struct mpi_allreduce_tag {};
struct doubling_allreduce_tag {};

constexpr static mpi_allreduce_tag mpi_allreduce;
constexpr static doubling_allreduce_tag doubling_allreduce;

// Used if no other implementation has priority
constexpr static auto default_sparse_allreduce = doubling_allreduce;
} // namespace tag

template <typename Buffer>
void inplace_sparse_allreduce(
    tag::mpi_allreduce_tag, Buffer &buffer, const std::size_t buffer_size, MPI_Op op, MPI_Comm comm
) {
  using Value = typename Buffer::value_type;
  MPI_Allreduce(
      MPI_IN_PLACE, buffer.data(), asserting_cast<int>(buffer_size), type::get<Value>(), op, comm
  );
}

template <typename Buffer>
void inplace_sparse_allreduce(
    tag::doubling_allreduce_tag,
    Buffer &buffer,
    const std::size_t buffer_size,
    MPI_Op op,
    MPI_Comm comm
) {
  const PEID size = mpi::get_comm_size(comm);
  const PEID rank = mpi::get_comm_rank(comm);

  KASSERT(op == MPI_SUM);
  KASSERT(math::is_power_of_2(size));

  using Value = typename Buffer::value_type;
  using Dense = std::pair<std::size_t, Value>;

  std::vector<Dense> sendbuf;
  std::vector<Dense> recvbuf;

  for (PEID iteration = 0; (2 << iteration) < size + 1; ++iteration) {
    const PEID distance = 1 << iteration;
    const PEID subtree_size = 2 << iteration;
    PEID neighbor = rank;
    if (rank % subtree_size < subtree_size / 2) {
      neighbor += distance;
    } else {
      neighbor -= distance;
    }

    sendbuf.clear();
    recvbuf.clear();

    for (std::size_t i = 0; i < buffer_size; ++i) {
      if (buffer[i] != 0) {
        sendbuf.emplace_back(i, buffer[i]);
      }
    }
    MPI_Datatype mpi_type = type::get<std::pair<std::size_t, Value>>();

    MPI_Request send_req;
    MPI_Isend(
        sendbuf.data(), asserting_cast<int>(sendbuf.size()), mpi_type, neighbor, 0, comm, &send_req
    );

    MPI_Status probe;
    MPI_Probe(neighbor, 0, comm, &probe);

    int recv_size;
    MPI_Get_count(&probe, mpi_type, &recv_size);
    recvbuf.resize(recv_size);

    MPI_Recv(recvbuf.data(), recv_size, mpi_type, neighbor, 0, comm, MPI_STATUS_IGNORE);

    MPI_Wait(&send_req, MPI_STATUS_IGNORE);

    for (const auto &[i, v] : recvbuf) {
      buffer[i] += v;
    }
  }
}

template <typename Buffer>
void inplace_sparse_allreduce(
    Buffer &buffer, const std::size_t buffer_size, MPI_Op op, MPI_Comm comm
) {
  inplace_sparse_allreduce<Buffer>(tag::default_sparse_allreduce, buffer, buffer_size, op, comm);
}
} // namespace kaminpar::mpi
