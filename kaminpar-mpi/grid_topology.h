/*******************************************************************************
 * Computes a mapping of an arbitrary number of PEs to a 2D grid.
 *
 * @file:   grid_topology.h
 * @author: Daniel Seemaier
 * @date:   09.09.2022
 ******************************************************************************/
#pragma once

#include <cmath>

#include "kaminpar-mpi/definitions.h"
#include "kaminpar-mpi/wrapper.h"

namespace kaminpar::mpi {
class GridTopology {
public:
  GridTopology(const PEID size) : _size(size), _sqrt(static_cast<PEID>(std::sqrt(size))) {}

  inline PEID row(const PEID pe) const {
    if (pe < num_pes_in_full_rectangle()) {
      return pe / num_cols();
    } else {
      return (pe - num_pes_in_full_rectangle()) / num_full_cols() + partial_column_size();
    }
  }

  inline PEID col(const PEID pe) const {
    if (pe < num_pes_in_full_rectangle()) {
      return pe % num_cols();
    } else {
      return (pe - num_pes_in_full_rectangle()) % num_full_cols();
    }
  }

  inline PEID virtual_col(const PEID pe) const {
    if (partial_column_size() == 0 || col(pe) < num_full_cols()) {
      return col(pe);
    } else {
      return row(pe);
    }
  }

  inline PEID row_size(const PEID row) const {
    if (row < partial_column_size()) {
      return num_cols();
    } else {
      return num_full_cols();
    }
  }

  inline PEID max_row_size() const {
    return num_cols();
  }

  inline PEID col_size(const PEID column) const {
    if (column < num_full_cols()) {
      return _sqrt;
    } else {
      return partial_column_size();
    }
  }

  inline PEID virtual_col_size(const PEID column) const {
    return _sqrt + (column < partial_column_size());
  }

  inline PEID max_col_size() const {
    return _sqrt;
  }

  inline PEID num_cols() const {
    return std::ceil(1.0 * _size / _sqrt);
  }

  inline PEID num_rows() const {
    return _sqrt;
  }

  inline PEID num_cols_in_row(const PEID row) const {
    return num_full_cols() + (row < partial_column_size());
  }

  inline PEID num_full_cols() const {
    return _size / _sqrt;
  }

  inline PEID virtual_element(const PEID row, const PEID virtual_col) const {
    if (row < partial_column_size()) {
      return row * num_cols() + virtual_col;
    } else if (row < num_rows()) {
      return row * num_full_cols() + virtual_col + partial_column_size();
    } else {
      return num_cols() - 1 + (num_cols() * virtual_col);
    }
  }

private:
  inline PEID partial_column_size() const {
    return _size - _sqrt * num_full_cols();
  }

  inline PEID num_pes_in_full_rectangle() const {
    return partial_column_size() == 0 ? _size : num_cols() * partial_column_size();
  }

  PEID _size;
  PEID _sqrt;
};

class GridCommunicator {
public:
  GridCommunicator(MPI_Comm comm) {
    const auto [size, rank] = get_comm_info(comm);
    GridTopology topo(size);
    MPI_Comm_split(comm, topo.row(rank), rank, &_row_comm);
    MPI_Comm_split(
        comm,
        topo.virtual_col(rank),
        topo.virtual_col(rank) == topo.col(rank) ? rank : size + rank,
        &_col_comm
    );
  }

  MPI_Comm row_comm() const {
    return _row_comm;
  }
  MPI_Comm col_comm() const {
    return _col_comm;
  }

  PEID row_comm_size() const {
    return get_comm_size(_row_comm);
  }
  PEID col_comm_size() const {
    return get_comm_size(_col_comm);
  }

private:
  MPI_Comm _row_comm;
  MPI_Comm _col_comm;
};

GridCommunicator &get_grid_communicator(MPI_Comm comm);
} // namespace kaminpar::mpi
