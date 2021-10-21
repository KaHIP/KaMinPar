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
#include "dkaminpar/mpi_wrapper.h"

#include <concepts>
#include <mpi.h>
#include <ranges>
#include <utility>
#include <vector>

namespace dkaminpar::mpi {
template<template<typename> typename RecvContainer, std::ranges::contiguous_range SendBuffer,
         template<typename> typename SendBufferContainer>
std::vector<RecvContainer<std::ranges::range_value_t<SendBuffer>>>
sparse_all_to_all_get(const SendBufferContainer<SendBuffer> &send_buffers, const int tag,
                      MPI_Comm comm = MPI_COMM_WORLD, const bool self = false) {
  const auto [size, rank] = mpi::get_comm_info(comm);
  std::vector<MPI_Request> requests;
  requests.reserve(size);

  for (PEID pe = 0; pe < size; ++pe) {
    if (self || pe != rank) {
      requests.emplace_back();
      mpi::isend(send_buffers[pe], pe, tag, requests.back(), comm);
    }
  }

  std::vector<RecvContainer<std::ranges::range_value_t<SendBuffer>>> recv_messages(size);
  for (PEID pe = 0; pe < size; ++pe) {
    if (self || pe != rank) {
      using T = std::ranges::range_value_t<SendBuffer>;
      recv_messages[pe] = mpi::probe_recv<T, RecvContainer>(pe, tag, MPI_STATUS_IGNORE, comm);
    }
  }

  mpi::waitall(requests);

  return recv_messages;
}

template<template<typename> typename RecvContainer, std::ranges::contiguous_range SendBuffer,
         template<typename> typename SendBufferContainer,
         std::invocable<PEID, const RecvContainer<std::ranges::range_value_t<SendBuffer>> &> RecvLambda>
void sparse_all_to_all(const SendBufferContainer<SendBuffer> &send_buffers, const int tag, RecvLambda &&recv_lambda,
                       MPI_Comm comm = MPI_COMM_WORLD, const bool self = false) {
  const auto [size, rank] = mpi::get_comm_info(comm);
  std::vector<MPI_Request> requests;
  requests.reserve(size);

  for (PEID pe = 0; pe < size; ++pe) {
    if (self || pe != rank) {
      requests.emplace_back();
      mpi::isend(send_buffers[pe], pe, tag, requests.back(), comm);
    }
  }

  for (PEID pe = 0; pe < size; ++pe) {
    if (self || pe != rank) {
      using T = std::ranges::range_value_t<SendBuffer>;
      auto recvbuf = mpi::probe_recv<T, RecvContainer>(pe, tag, MPI_STATUS_IGNORE, comm);
      recv_lambda(pe, recvbuf);
    }
  }

  mpi::waitall(requests);
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

template<typename Message, template<typename> typename Buffer = scalable_vector>
void sparse_alltoall(const std::vector<Buffer<Message>> &send_buffers, auto &&receiver, MPI_Comm comm) {
  using Receiver = decltype(receiver);
  constexpr bool receiver_invocable_with_pe = std::is_invocable_r_v<void, Receiver, Buffer<Message>, PEID>;
  constexpr bool receiver_invocable_without_pe = std::is_invocable_r_v<void, Receiver, Buffer<Message>>;
  static_assert(receiver_invocable_with_pe || receiver_invocable_without_pe, "bad receiver type");

  const auto [size, rank] = mpi::get_comm_info(comm);

  std::vector<MPI_Request> requests;
  requests.reserve(size);

  for (PEID pe = 0; pe < size; ++pe) {
    if (pe != rank) {
      requests.emplace_back();
      ASSERT(static_cast<std::size_t>(pe) < send_buffers.size());
      mpi::isend(send_buffers[pe], pe, 0, requests.back(), comm);
    }
  }

  for (PEID pe = 0; pe < size; ++pe) {
    if (pe != rank) {
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
} // namespace dkaminpar::mpi