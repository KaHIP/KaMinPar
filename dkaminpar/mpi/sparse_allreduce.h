/*******************************************************************************
 * @file:   sparse_allreduce.h
 * @author: Daniel Seemaier
 * @date:   27.03.2023
 * @brief:  Allreduce for sparse key-value-pairs.
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <utility>

#include <mpi.h>
#include <tbb/parallel_for.h>

#include "dkaminpar/definitions.h"
#include "dkaminpar/mpi/wrapper.h"

#include "common/noinit_vector.h"

namespace kaminpar::mpi {
namespace tag {
struct mpi_allreduce_tag {};
struct sparse_allreduce_tag {};

constexpr static mpi_allreduce_tag mpi_allreduce;
constexpr static sparse_allreduce_tag sparse_allreduce;

// Used if no other implementation has priority
constexpr static auto default_sparse_allreduce = mpi_allreduce;
} // namespace tag

template <typename Value, typename Buffer>
void sparse_allreduce(tag::mpi_allreduce_tag, Buffer &buffer,
                      const std::size_t size, MPI_Op op, MPI_Comm comm) {
  MPI_Allreduce(MPI_IN_PLACE, buffer.data(), asserting_cast<int>(size),
                type::get<Value>(), op, comm);
}

template <typename Value, typename Buffer>
void sparse_allreduce(Buffer &buffer, const std::size_t size, MPI_Op op,
                      MPI_Comm comm) {
  sparse_allreduce(tag::default_sparse_allreduce, buffer, size, op, comm);
}
} // namespace kaminpar::mpi

