/*******************************************************************************
 * Functions to annotate the timer tree with aggregate timer information from
 * all PEs.
 *
 * @file:   timer.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 ******************************************************************************/
#pragma once

#include <mpi.h>

#include "kaminpar-dist/dkaminpar.h"

#include "kaminpar-common/timer.h"

#ifdef KAMINPAR_ENABLE_TIMER_BARRIERS
#define TIMER_BARRIER(comm) MPI_Barrier(comm)
#else // KAMINPAR_ENABLE_TIMER_BARRIERS
#define TIMER_BARRIER(comm)
#endif // KAMINPAR_ENABLE_TIMER_BARRIERS

namespace kaminpar::dist {
void finalize_distributed_timer(Timer &timer, MPI_Comm comm);
} // namespace kaminpar::dist
