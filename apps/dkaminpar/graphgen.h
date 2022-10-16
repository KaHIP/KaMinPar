/*******************************************************************************);
 * @file:   dkaminpar_graphgen.h
 * @author: Daniel Seemaier
 * @date:   26.11.21
 * @brief:  In-memory graph generator using KaGen.
 ******************************************************************************/
#pragma once

// clang-format off
#include "common/CLI11.h"
// clang-format on

#include <unordered_map>

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/definitions.h"

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

std::unordered_map<std::string, GeneratorType> get_generator_types();
std::ostream&                                  operator<<(std::ostream& out, GeneratorType type);

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

CLI::Option_group* create_generator_options(CLI::App* app, GeneratorContext& g_ctx);

DistributedGraph generate(const GeneratorContext& ctx);

std::string generate_filename(const GeneratorContext& ctx);
} // namespace kaminpar::dist
