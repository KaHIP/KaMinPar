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

#include "kaminpar/definitions.h"

#include <cstdint>
#include <mpi.h>

namespace dkaminpar {
namespace shm = kaminpar;

using DNodeID = uint64_t;
using DEdgeID = uint64_t;
using DNodeWeight = int64_t;
using DEdgeWeight = int64_t;
using DBlockID = uint32_t;
using DBlockWeight = int64_t;

constexpr DBlockWeight kInvalidBlockWeight = std::numeric_limits<DBlockWeight>::max();

using PEID = int;

namespace internal {
inline int get_rank(MPI_Comm comm = MPI_COMM_WORLD) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}
} // namespace internal

template<typename T>
using scalable_vector = shm::scalable_vector<T>;

#define ROOT(x) ((x) == 0)

// clang-format off
#define LOG_RANK "[PE" << dkaminpar::internal::get_rank() << "]"

#undef ALWAYS_ASSERT
#define ALWAYS_ASSERT(x) kaminpar::debug::evaluate_assertion((x)) || kaminpar::debug::DisposableLogger<true>(std::cout) \
  << kaminpar::logger::MAGENTA << POSITION << LOG_RANK << " "                                                           \
  << kaminpar::logger::RED << "Assertion failed: `" << #x << "`\n"

// Assertions that are only evaluated on root (rank 0)
#define ALWAYS_ASSERT_ROOT(x) ALWAYS_ASSERT(dkaminpar::mpi::get_comm_rank() != 0 || (x))
#define ASSERT_ROOT(x) ASSERT(dkaminpar::mpi::get_comm_rank() != 0 || (x))

#undef DBGC
#define DBGC(cond) (kDebug && (cond)) && kaminpar::debug::DisposableLogger<false>(std::cout) << kaminpar::logger::MAGENTA << POSITION << LOG_RANK << " " << kaminpar::logger::DEFAULT_TEXT

#undef LOG
#undef LLOG
#define LOG (dkaminpar::internal::get_rank() == 0) && kaminpar::debug::DisposableLogger<false>(std::cout)
#define LLOG (dkaminpar::internal::get_rank() == 0) && kaminpar::debug::DisposableLogger<false>(std::cout, "")

#undef LOG_ERROR
#define LOG_ERROR (kaminpar::Logger(std::cout) << LOG_RANK << kaminpar::logger::RED << "[Error] ")

#undef FATAL_ERROR
#undef FATAL_PERROR
#define FATAL_ERROR (kaminpar::debug::DisposableLogger<true>(std::cout) << LOG_RANK << " " << kaminpar::logger::RED << "[Fatal] ")
#define FATAL_PERROR (kaminpar::debug::DisposableLogger<true>(std::cout, std::string(": ") + std::strerror(errno) + "\n") << LOG_RANK << " " << kaminpar::logger::RED << "[Fatal] ")

#define DLOG (kaminpar::Logger() << LOG_RANK << " ")
// clang-format on
} // namespace dkaminpar