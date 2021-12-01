/*******************************************************************************
 * @file:   distributed_timer.h
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:
 ******************************************************************************/
#pragma once

#include "dkaminpar/distributed_definitions.h"
#include "kaminpar/utility/timer.h"

namespace dkaminpar::timer {
void collect_and_annotate_distributed_timer(shm::Timer &timer, MPI_Comm comm = MPI_COMM_WORLD);
}