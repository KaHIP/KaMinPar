/*******************************************************************************
 * @file:   grid_topology.h
 * @author: Daniel Seemaier
 * @date:   09.09.2022
 * @brief:
 ******************************************************************************/
#pragma once

#include <cmath>

#include "dkaminpar/definitions.h"

#include "common/logger.h"

namespace kaminpar::dist::mpi {
class GridTopology {
    SET_DEBUG(true);

public:
    GridTopology(const PEID size) : _size(size), _sqrt(static_cast<PEID>(std::sqrt(size))) {}

    inline PEID row(const PEID pe) {
        if (pe < num_pes_in_full_rectangle()) {
            return pe / num_columns();
        } else {
            return (pe - num_pes_in_full_rectangle()) / num_full_columns() + partial_column_size();
        }
    }

    inline PEID column(const PEID pe) {
        if (pe < num_pes_in_full_rectangle()) {
            return pe % num_columns();
        } else {
            return (pe - num_pes_in_full_rectangle()) % num_full_columns();
        }
    }

    inline PEID virtual_column(const PEID pe) {
        if (partial_column_size() == 0 || column(pe) < num_full_columns()) {
            return column(pe);
        } else {
            return row(pe);
        }
    }

private:
    inline PEID num_columns() const {
        return std::ceil(1.0 * _size / _sqrt);
    }

    inline PEID num_full_columns() const {
        return _size / _sqrt;
    }

    inline PEID partial_column_size() const {
        return _size - _sqrt * num_full_columns();
    }

    inline PEID num_pes_in_full_rectangle() const {
        return partial_column_size() == 0 ? _size : num_columns() * partial_column_size();
    }

    PEID _size;
    PEID _sqrt;
};
} // namespace kaminpar::dist::mpi
