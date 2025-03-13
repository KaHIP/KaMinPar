/*******************************************************************************
 * Utilities for graph generation with streaming KaGen.
 *
 * @file:   dist_skagen.h
 * @author: Daniel Salwasser
 * @date:   13.07.2024
 ******************************************************************************/
#pragma once

#include <string>

#include <mpi.h>

#include "kaminpar-dist/datastructures/distributed_compressed_graph.h"
#include "kaminpar-dist/datastructures/distributed_csr_graph.h"
#include "kaminpar-dist/dkaminpar.h"

namespace kaminpar::dist::io::skagen {

DistributedCSRGraph csr_streaming_generate(
    const std::string &graph_options, const PEID chunks_per_pe, const MPI_Comm comm
);

DistributedCompressedGraph compressed_streaming_generate(
    const std::string &graph_options, const PEID chunks_per_pe, const MPI_Comm comm
);

} // namespace kaminpar::dist::io::skagen
