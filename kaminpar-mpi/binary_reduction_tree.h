/*******************************************************************************
 * Simple implementation of a binary reduction tree.
 *
 * @file:   binary_reduction_tree.h
 * @author: Daniel Seemaier
 * @date:   21.08.2023
 ******************************************************************************/
#pragma once

#include <cmath>

#include <mpi.h>

#include "kaminpar-mpi/wrapper.h"

namespace kaminpar::mpi {
template <typename Buffer, typename Combiner>
Buffer perform_binary_reduction(Buffer sendbuf, Buffer empty, Combiner &&combiner, MPI_Comm comm) {
  enum class Role {
    SENDER,
    RECEIVER,
    NOOP
  };

  // Special case: if we only have one PE, combine with an empty buffer -- this ensures that we
  // remove moves that would overload blocks in the sequential case
  const PEID size = mpi::get_comm_size(comm);
  if (size == 1) {
    return combiner(std::move(sendbuf), std::move(empty));
  }

  const PEID rank = mpi::get_comm_rank(comm);
  PEID active = size;

  while (active > 1) {
    if (rank >= active) {
      continue;
    }

    const Role role = [&] {
      if (rank == 0 && active % 2 == 1) {
        return Role::NOOP;
      } else if (rank < std::ceil(active / 2.0)) {
        return Role::RECEIVER;
      } else {
        return Role::SENDER;
      }
    }();

    if (role == Role::SENDER) {
      const PEID to = rank - active / 2;
      mpi::send(sendbuf.data(), sendbuf.size(), to, 0, comm);
      return {};
    } else if (role == Role::RECEIVER) {
      const PEID from = rank + active / 2;
      Buffer recvbuf = mpi::probe_recv<typename Buffer::value_type, Buffer>(from, 0, comm);
      sendbuf = combiner(std::move(sendbuf), std::move(recvbuf));
    }

    active = active / 2 + active % 2;
  }

  return sendbuf;
}
} // namespace kaminpar::mpi
