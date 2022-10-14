/*******************************************************************************);
 * @file:   dkaminpar_graphgen.h
 * @author: Daniel Seemaier
 * @date:   26.11.21
 * @brief:  In-memory graph generator using KaGen.
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/definitions.h"

#include "common/utils/enum_string_conversion.h"

namespace kaminpar::dist {
enum class GeneratorType {
    NONE,
    GNM,
    RGG2D,
    RGG3D,
    RDG2D,
    RDG3D,
    RHG,
    GRID2D,
    GRID3D,
    RMAT,
};

DECLARE_ENUM_STRING_CONVERSION(GeneratorType, generator_type);

struct GeneratorContext {
    GeneratorType type           = GeneratorType::NONE;
    GlobalNodeID  n              = 0;
    GlobalEdgeID  m              = 0;
    double        p              = 0;
    double        gamma          = 0;
    PEID          scale          = 1;
    bool          periodic       = false;
    bool          validate_graph = false;
    bool          save_graph     = false;
    int           seed           = 0;
    double        prob_a         = 0.0;
    double        prob_b         = 0.0;
    double        prob_c         = 0.0;
    bool          advanced_stats = false;
};

DistributedGraph generate(GeneratorContext ctx);

std::string generate_filename(GeneratorContext ctx);
} // namespace kaminpar::dist
