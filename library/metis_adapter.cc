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
#include "kaminpar.h"

#include <iostream>
#include <limits>
#include <tbb/task_arena.h>
#include <type_traits>

using namespace libkaminpar;

extern "C" {
#include <metis.h>

METIS_API(int)
METIS_PartGraphKway(idx_t *nvtxs, idx_t *ncon, idx_t *xadj, idx_t *adjncy, idx_t *vwgt, idx_t *vsize, idx_t *adjwgt,
                    idx_t *nparts, real_t *tpwgts, real_t *ubvec, idx_t *options, idx_t *edgecut, idx_t *part) {
  // ensure that idx_t is compatible with KaMinPar data types
  using uidx_t = std::make_unsigned_t<idx_t>;
  static_assert(std::numeric_limits<uidx_t>::digits == std::numeric_limits<NodeID>::digits);
  static_assert(std::numeric_limits<uidx_t>::digits == std::numeric_limits<EdgeID>::digits);
  static_assert(std::numeric_limits<uidx_t>::digits == std::numeric_limits<BlockID>::digits);
  static_assert(std::numeric_limits<idx_t>::digits == std::numeric_limits<NodeWeight>::digits);
  static_assert(std::numeric_limits<idx_t>::digits == std::numeric_limits<EdgeWeight>::digits);
  static_assert(std::numeric_limits<idx_t>::digits == std::numeric_limits<BlockWeight>::digits);

  // print warnings if unsupported METIS features are used
  if (vsize != nullptr) { std::cout << "ignoring vsize (unsupported)" << std::endl; }
  if (tpwgts != nullptr) { std::cout << "ignoring tpwgts (unsupported)" << std::endl; }
  if (*ncon != 1) { std::cout << "ignoring additional balancing constraints (unsupported)" << std::endl; }

  const NodeID n = *nvtxs;
  const BlockID k = *nparts;
  EdgeID *nodes = reinterpret_cast<EdgeID *>(xadj);
  NodeID *edges = reinterpret_cast<NodeID *>(adjncy);
  NodeWeight *node_weights = reinterpret_cast<NodeWeight *>(vwgt);
  EdgeWeight *edge_weights = reinterpret_cast<EdgeWeight *>(adjwgt);
  double epsilon = (ubvec == nullptr) ? 0.001 : (*ubvec) - 1.0;
  std::size_t repetitions = 1;
  BlockID base = 0;

  auto builder = PartitionerBuilder::from_adjacency_array(n, nodes, edges);
  if (node_weights != nullptr) { builder.with_node_weights(node_weights); }
  if (edge_weights != nullptr) { builder.with_edge_weights(edge_weights); }
  auto partitioner = builder.create();

  // set options
  if (options != nullptr) {
    if (options[METIS_OPTION_MINCONN]) { std::cout << "ignoring METIS_OPTION_MINCONN (unsupported)" << std::endl; }
    if (options[METIS_OPTION_CONTIG]) { std::cout << "ignoring METIS_OPTION_CONTIG (unsupported)" << std::endl; }
    if (options[METIS_OPTION_COMPRESS]) { std::cout << "ignoring METIS_OPTION_COMPRESS (unsupported)" << std::endl; }
    if (options[METIS_OPTION_PFACTOR]) { std::cout << "ignoring METIS_OPTION_PFACTOR (unsupported)" << std::endl; }
    if (options[METIS_OPTION_OBJTYPE] != METIS_OBJTYPE_CUT) {
      std::cout << "ignoring METIS_OPTION_OBJTYPE (unsupported)" << std::endl;
    }

    if (options[METIS_OPTION_NCUTS]) { repetitions = options[METIS_OPTION_NCUTS]; }
    base = options[METIS_OPTION_NUMBERING];
    if (options[METIS_OPTION_NITER] > 0) {
      partitioner.set_option("--r-lp-num-iters", std::to_string(options[METIS_OPTION_NITER]));
      partitioner.set_option("--i-r-fm-iterations", std::to_string(options[METIS_OPTION_NITER]));
    }
    partitioner.set_option("--seed", std::to_string(options[METIS_OPTION_SEED]));
    if (options[METIS_OPTION_UFACTOR]) { epsilon = options[METIS_OPTION_UFACTOR] / 1000.0; }
  }

  partitioner.set_option("--epsilon", std::to_string(epsilon));
  partitioner.set_option("--threads", std::to_string(tbb::this_task_arena::max_concurrency()));

  // compute the partition
  std::unique_ptr<BlockID[]> best_partition;
  EdgeWeight best_edge_cut;

  for (std::size_t iter = 0; iter < repetitions; ++iter) {
    EdgeWeight edge_cut;
    auto partition = partitioner.partition(k, edge_cut);

    if (iter == 0 || edge_cut < best_edge_cut) {
      best_edge_cut = edge_cut;
      best_partition = std::move(partition);
      *edgecut = edge_cut;
    }
  }

  // copy the partition to part
  for (NodeID u = 0; u < n; ++u) { part[u] = base + best_partition[u]; }

  return METIS_OK;
}

METIS_API(int)
METIS_PartGraphRecursive(idx_t *nvtxs, idx_t *ncon, idx_t *xadj, idx_t *adjncy, idx_t *vwgt, idx_t *vsize,
                         idx_t *adjwgt, idx_t *nparts, real_t *tpwgts, real_t *ubvec, idx_t *options, idx_t *edgecut,
                         idx_t *part) {
  // TODO adjust ufactor parameter
  return METIS_PartGraphKway(nvtxs, ncon, xadj, adjncy, vwgt, vsize, adjwgt, nparts, tpwgts, ubvec, options, edgecut,
                             part);
}

METIS_API(int)
METIS_MeshToDual(idx_t *, idx_t *, idx_t *, idx_t *, idx_t *, idx_t *, idx_t **, idx_t **) { return METIS_ERROR; }

METIS_API(int)
METIS_MeshToNodal(idx_t *, idx_t *, idx_t *, idx_t *, idx_t *, idx_t **, idx_t **) { return METIS_ERROR; }

METIS_API(int)
METIS_PartMeshNodal(idx_t *, idx_t *, idx_t *, idx_t *, idx_t *, idx_t *, idx_t *, real_t *, idx_t *, idx_t *, idx_t *,
                    idx_t *) {
  return METIS_ERROR;
}

METIS_API(int)
METIS_PartMeshDual(idx_t *, idx_t *, idx_t *, idx_t *, idx_t *, idx_t *, idx_t *, idx_t *, real_t *, idx_t *, idx_t *,
                   idx_t *, idx_t *) {
  return METIS_ERROR;
}

METIS_API(int)
METIS_NodeND(idx_t *, idx_t *, idx_t *, idx_t *, idx_t *, idx_t *, idx_t *) { return METIS_ERROR; }

METIS_API(int) METIS_Free(void *) { return METIS_ERROR; }

METIS_API(int) METIS_SetDefaultOptions(idx_t *options) {
  options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
  options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
  options[METIS_OPTION_NCUTS] = 1;
  options[METIS_OPTION_NUMBERING] = 0;
  options[METIS_OPTION_SEED] = 0;
  options[METIS_OPTION_NITER] = 0;
  options[METIS_OPTION_MINCONN] = 0;
  options[METIS_OPTION_CONTIG] = 0;
  options[METIS_OPTION_COMPRESS] = 0;
  options[METIS_OPTION_PFACTOR] = 0;
  options[METIS_OPTION_UFACTOR] = 30;
  options[METIS_OPTION_DBGLVL] = METIS_DBG_INFO;
  return METIS_OK;
}

METIS_API(int)
METIS_NodeNDP(idx_t, idx_t *, idx_t *, idx_t *, idx_t, idx_t *, idx_t *, idx_t *, idx_t *) { return METIS_ERROR; }

METIS_API(int)
METIS_ComputeVertexSeparator(idx_t *, idx_t *, idx_t *, idx_t *, idx_t *, idx_t *, idx_t *) { return METIS_ERROR; }

METIS_API(int)
METIS_NodeRefine(idx_t, idx_t *, idx_t *, idx_t *, idx_t *, idx_t *, real_t) { return METIS_ERROR; }
}