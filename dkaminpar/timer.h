/*******************************************************************************
 * @file:   distributed_timer.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:
 ******************************************************************************/
#pragma once

#include "common/timer.h"
#include "dkaminpar/definitions.h"

namespace kaminpar::dist {
void finalize_distributed_timer(Timer& timer, MPI_Comm comm = MPI_COMM_WORLD);
}
