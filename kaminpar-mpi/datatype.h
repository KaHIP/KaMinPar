/*******************************************************************************
 * Utility functions to automatically select the right MPI type.
 * If a type does not map to any of the predefined MPI data types, a new
 * contiguous MPI type is created.
 *
 * @file:   datatype.h
 * @author: Daniel Seemaier
 * @date:   10.06.2022
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <utility>

#include <mpi.h>

namespace kaminpar::mpi::type {
template <std::size_t N> inline MPI_Datatype custom() {
  static MPI_Datatype type = MPI_DATATYPE_NULL;
  if (type == MPI_DATATYPE_NULL) {
    MPI_Type_contiguous(N, MPI_CHAR, &type);
    MPI_Type_commit(&type);
  }
  return type;
}

// Map to default MPI type
template <typename T> inline MPI_Datatype get() {
  if constexpr (std::is_same_v<T, bool>) {
    return MPI_CXX_BOOL;
  } else if constexpr (std::is_same_v<T, std::uint8_t>) {
    return MPI_UINT8_T;
  } else if constexpr (std::is_same_v<T, std::int8_t>) {
    return MPI_INT8_T;
  } else if constexpr (std::is_same_v<T, std::uint16_t>) {
    return MPI_UINT16_T;
  } else if constexpr (std::is_same_v<T, std::int16_t>) {
    return MPI_INT16_T;
  } else if constexpr (std::is_same_v<T, std::uint32_t>) {
    return MPI_UINT32_T;
  } else if constexpr (std::is_same_v<T, std::int32_t>) {
    return MPI_INT32_T;
  } else if constexpr (std::is_same_v<T, std::uint64_t>) {
    return MPI_UINT64_T;
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    return MPI_INT64_T;
  } else if constexpr (std::is_same_v<T, float>) {
    return MPI_FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    return MPI_DOUBLE;
  } else if constexpr (std::is_same_v<T, long double>) {
    return MPI_LONG_DOUBLE;
  } else if constexpr (std::is_same_v<T, std::pair<float, int>>) {
    return MPI_FLOAT_INT;
  } else if constexpr (std::is_same_v<T, std::pair<double, int>>) {
    return MPI_DOUBLE_INT;
  } else if constexpr (std::is_same_v<T, std::pair<long double, int>>) {
    return MPI_LONG_DOUBLE_INT;
  } else {
    return custom<sizeof(T)>();
  }
}
} // namespace kaminpar::mpi::type
