/*******************************************************************************
 * @file:   distributed_definitions.h
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:
 ******************************************************************************/
#pragma once

#include "kaminpar/definitions.h"
#include "kaminpar/parallel.h"

#include <cstdint>
#include <iomanip>
#include <mpi.h>

namespace dkaminpar {
namespace shm = kaminpar;

using shm::NodeID;
using GlobalNodeID = uint64_t;
using shm::NodeWeight;
using GlobalNodeWeight = int64_t;
using shm::EdgeID;
using GlobalEdgeID = uint64_t;
using shm::EdgeWeight;
using GlobalEdgeWeight = int64_t;
using shm::BlockID;
using BlockWeight = int64_t;

using shm::kInvalidNodeID;
constexpr GlobalNodeID kInvalidGlobalNodeID = std::numeric_limits<GlobalNodeID>::max();
using shm::kInvalidNodeWeight;
constexpr GlobalNodeWeight kInvalidGlobalNodeWeight = std::numeric_limits<GlobalNodeWeight>::max();
using shm::kInvalidEdgeID;
constexpr GlobalEdgeID kInvalidGlobalEdgeID = std::numeric_limits<GlobalEdgeID>::max();
using shm::kInvalidEdgeWeight;
constexpr GlobalEdgeWeight kInvalidGlobalEdgeWeight = std::numeric_limits<GlobalEdgeWeight>::max();
using shm::kInvalidBlockID;
using shm::kInvalidBlockWeight;

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

template<typename T>
using cache_aligned_vector = std::vector<T, tbb::cache_aligned_allocator<T>>;

template<typename T>
using Atomic = shm::parallel::IntegralAtomicWrapper<T>;

class SynchronizedLogger {
public:
  SynchronizedLogger(int root = 0, MPI_Comm comm = MPI_COMM_WORLD) : _buf{}, _logger{_buf}, _root{root}, _comm{comm} {}

  ~SynchronizedLogger() {
    _logger.flush();

    int size, rank;
    MPI_Comm_size(_comm, &size);
    MPI_Comm_rank(_comm, &rank);

    if (rank != _root) {
      std::string str = _buf.str();
      MPI_Send(str.data(), static_cast<int>(str.length()), MPI_CHAR, _root, 0, MPI_COMM_WORLD);
    } else {
      kaminpar::Logger logger;

      for (PEID pe = 0; pe < size; ++pe) {
        logger << "-------------------- " << pe << " --------------------\n";

        if (pe == rank) {
          logger << _buf.str();
        } else {
          MPI_Status status;
          MPI_Probe(pe, 0, MPI_COMM_WORLD, &status);

          int cnt;
          MPI_Get_count(&status, MPI_CHAR, &cnt);

          char *str = new char[cnt];
          MPI_Recv(str, cnt, MPI_CHAR, pe, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          logger << std::string(str, cnt);
        }
      }

      logger << "-------------------------------------------";
    }
  }

  template<typename Arg>
  SynchronizedLogger &operator<<(Arg &&arg) {
    _logger << std::forward<Arg>(arg);
    return *this;
  }

private:
  std::ostringstream _buf;
  shm::Logger _logger;
  int _root;
  MPI_Comm _comm;
};

#define ROOT(x) ((x) == 0)

// clang-format off
#define LOG_RANK "[PE" << dkaminpar::internal::get_rank() << "]"

#undef ALWAYS_ASSERT
#define ALWAYS_ASSERT(x) kaminpar::debug::evaluate_assertion((x)) || kaminpar::debug::DisposableLogger<true>(std::cout) \
  << kaminpar::logger::MAGENTA << POSITION << LOG_RANK << CPU << " "                                                    \
  << kaminpar::logger::RED << "Assertion failed: `" << #x << "`\n"

// Assertions that are only evaluated on root (rank 0)
#define ALWAYS_ASSERT_ROOT(x) ALWAYS_ASSERT(dkaminpar::mpi::get_comm_rank() != 0 || (x))
#define ASSERT_ROOT(x) ASSERT(dkaminpar::mpi::get_comm_rank() != 0 || (x))

#undef DBGC
#define DBGC(cond) (kDebug && (cond)) && kaminpar::debug::DisposableLogger<false>(std::cout)                            \
  << kaminpar::logger::MAGENTA << POSITION << LOG_RANK << CPU << " " << kaminpar::logger::DEFAULT_TEXT

#undef LOG
#undef LLOG
#define LOG (dkaminpar::internal::get_rank() == 0) && kaminpar::debug::DisposableLogger<false>(std::cout)
#define LLOG (dkaminpar::internal::get_rank() == 0) && kaminpar::debug::DisposableLogger<false>(std::cout, "")

#undef STATS
#define STATS kStatistics && (dkaminpar::internal::get_rank() == 0) && kaminpar::debug::DisposableLogger<false>(std::cout) << kaminpar::logger::CYAN

#undef LOG_ERROR
#define LOG_ERROR (kaminpar::Logger(std::cout) << LOG_RANK << kaminpar::logger::RED << "[Error] ")

#undef FATAL_ERROR
#undef FATAL_PERROR
#define FATAL_ERROR (kaminpar::debug::DisposableLogger<true>(std::cout) << LOG_RANK << " " << kaminpar::logger::RED << "[Fatal] ")
#define FATAL_PERROR (kaminpar::debug::DisposableLogger<true>(std::cout, std::string(": ") + std::strerror(errno) + "\n") << LOG_RANK << " " << kaminpar::logger::RED << "[Fatal] ")

#define DLOG (kaminpar::Logger() << LOG_RANK << " ")
#define SLOG (dkaminpar::SynchronizedLogger())
#define SLOGP(root, comm) (dkaminpar::SynchronizedLogger(root, comm))
// clang-format on
} // namespace dkaminpar