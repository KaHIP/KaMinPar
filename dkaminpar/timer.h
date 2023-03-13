/*******************************************************************************
 * @file:   timer.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Functions to annotate the timer on the root PE with min/max/mean/sd
 * timings of other PEs.
 ******************************************************************************/
#pragma once

#include <mpi.h>

#include "dkaminpar/definitions.h"

#include "common/timer.h"

namespace kaminpar::dist {
void finalize_distributed_timer(Timer &timer, MPI_Comm comm = MPI_COMM_WORLD);
}
