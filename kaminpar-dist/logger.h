/*******************************************************************************
 * Redefines the logging macros for the distributed setting:
 *
 * - DBG(C) also print the PE rank
 * - (L)LOG(_*) only print on PE 0
 * - DLOG prints the output from all PEs in undefined order, possibly interleaved
 * - SLOG collects the output from all PEs and prints it on PE 0
 *
 * @file:   logger.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 ******************************************************************************/
#pragma once

#include <mpi.h>

#include "kaminpar-mpi/utils.h"

#include "kaminpar-common/logger.h"

#define LOG_RANK "[PE" << kaminpar::mpi::get_comm_rank(MPI_COMM_WORLD) << "]"

#undef DBGC
#define DBGC(cond)                                                                                 \
  (kDebug && (cond)) && kaminpar::DisposableLogger<false>(std::cout)                               \
                            << kaminpar::logger::CYAN << LOG_RANK << kaminpar::logger::MAGENTA     \
                            << POSITION << " " << kaminpar::logger::DEFAULT_TEXT

#define DBGX(R)                                                                                    \
  (kDebug && kaminpar::mpi::get_comm_rank(MPI_COMM_WORLD) == R) &&                                 \
      kaminpar::DisposableLogger<false>(std::cout)                                                 \
          << kaminpar::logger::CYAN << LOG_RANK << kaminpar::logger::MAGENTA << POSITION << " "    \
          << kaminpar::logger::DEFAULT_TEXT
#define DBG0 DBGX(0)

#define IF_DBGX(R) if (kDebug && kaminpar::mpi::get_comm_rank(MPI_COMM_WORLD) == R)
#define IF_DBG0 IF_DBGX(0)

#undef LOG
#undef LLOG
#define LOG                                                                                        \
  (kaminpar::mpi::get_comm_rank(MPI_COMM_WORLD) == 0) &&                                           \
      kaminpar::DisposableLogger<false>(std::cout)
#define LLOG                                                                                       \
  (kaminpar::mpi::get_comm_rank(MPI_COMM_WORLD) == 0) &&                                           \
      kaminpar::DisposableLogger<false>(std::cout, "")

#undef STATS
#define STATS                                                                                      \
  kStatistics && (kaminpar::mpi::get_comm_rank(MPI_COMM_WORLD) == 0) &&                            \
      kaminpar::DisposableLogger<false>(std::cout) << kaminpar::logger::CYAN

#undef LOG_ERROR
#define LOG_ERROR (kaminpar::Logger(std::cout) << LOG_RANK << kaminpar::logger::RED << "[Error] ")

#undef FATAL_ERROR
#undef FATAL_PERROR
#define FATAL_ERROR                                                                                \
  (kaminpar::DisposableLogger<true>(std::cout)                                                     \
   << LOG_RANK << " " << kaminpar::logger::RED << "[Fatal] ")
#define FATAL_PERROR                                                                               \
  (kaminpar::DisposableLogger<true>(std::cout, std::string(": ") + std::strerror(errno) + "\n")    \
   << LOG_RANK << " " << kaminpar::logger::RED << "[Fatal] ")

#define DLOG (kaminpar::Logger() << LOG_RANK << " ")
#define SLOG (kaminpar::dist::SynchronizedLogger())
#define SLOGP(root, comm) (kaminpar::dist::SynchronizedLogger(root, comm))

namespace kaminpar::dist {
class SynchronizedLogger {
public:
  explicit SynchronizedLogger(const int root = 0, MPI_Comm comm = MPI_COMM_WORLD)
      : _buf{},
        _logger{_buf},
        _root{root},
        _comm{comm} {}

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

      for (int pe = 0; pe < size; ++pe) {
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

          delete[] str;
        }
      }

      logger << "-------------------------------------------";
    }
  }

  template <typename Arg> SynchronizedLogger &operator<<(Arg &&arg) {
    _logger << std::forward<Arg>(arg);
    return *this;
  }

private:
  std::ostringstream _buf;
  Logger _logger;
  int _root;
  MPI_Comm _comm;
};
} // namespace kaminpar::dist
