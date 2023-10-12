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

namespace kaminpar::dist {
void finalize_distributed_timer(Timer &timer, MPI_Comm comm);
} // namespace kaminpar::dist
