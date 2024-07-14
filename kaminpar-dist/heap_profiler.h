/*******************************************************************************
 * Functions to annotate the heap profiler tree with aggregated information from
 * all PEs.
 *
 * @file:   heap_profiler.h
 * @author: Daniel Salwasser
 * @date:   16.06.2024
 ******************************************************************************/
#pragma once

#include <mpi.h>

#include "kaminpar-common/heap_profiler.h"

namespace kaminpar::dist {

/**
 * Annotates a heap profiler tree with aggregated information from all PEs.
 *
 * @param heap_profiler The heap profiler to annotate.
 * @param comm The group of processes whose information to aggregate.
 * @return The rank of the process that stores the annotated heap profile.
 */
int finalize_distributed_heap_profiler(heap_profiler::HeapProfiler &heap_profiler, MPI_Comm comm);

} // namespace kaminpar::dist
