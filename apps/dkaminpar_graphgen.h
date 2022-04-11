/*******************************************************************************
 * @file:   dkaminpar_graphgen.h
 *
 * @author: Daniel Seemaier
 * @date:   26.11.21
 * @brief:  In-memory graph generator using KaGen.
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/definitions.h"
#include "kaminpar/application/arguments.h"

namespace dkaminpar::graphgen {
enum class GeneratorType {
    NONE,
    GNM,
    RGG2D,
    RDG2D,
    RHG,
    BA,
    KRONECKER,
};

DECLARE_ENUM_STRING_CONVERSION(GeneratorType, generator_type);

struct GeneratorContext {
    GeneratorType type{GeneratorType::NONE};
    GlobalNodeID  n{0};
    GlobalEdgeID  m{0};
    unsigned long k{0};
    NodeID        d{0};
    double        p{0};
    double        r{0};
    double        gamma{0};
    bool          save_graph{false};
    bool          redistribute_edges{false};
    int           scale{1};
};

DistributedGraph generate(GeneratorContext ctx, int seed = 0);
} // namespace dkaminpar::graphgen
