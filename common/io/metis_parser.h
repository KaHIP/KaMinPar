#pragma once

#include <kassert/kassert.hpp>

#include "common/io/mmap_toker.h"
#include "common/logger.h"

namespace kaminpar::io::metis {
struct Format {
    std::uint64_t number_of_nodes  = 0;
    std::uint64_t number_of_edges  = 0;
    bool          has_node_weights = false;
    bool          has_edge_weights = false;
};

inline Format parse_header(MappedFileToker& toker) {
    toker.skip_spaces();
    while (toker.current() == '%') {
        toker.skip_line();
        toker.skip_spaces();
    }

    const std::uint64_t number_of_nodes = toker.scan_uint<std::uint64_t>();
    const std::uint64_t number_of_edges = toker.scan_uint<std::uint64_t>();
    const std::uint64_t format          = (toker.current() != '\n') ? toker.scan_uint<std::uint64_t>() : 0;
    toker.consume_char('\n');

    [[maybe_unused]] const bool has_node_sizes   = format / 100;        // == 1xx
    const bool                  has_node_weights = (format % 100) / 10; // == x1x
    const bool                  has_edge_weights = format % 10;         // == xx1

    if (has_node_sizes) {
        LOG_WARNING << "ignoring node sizes";
    }

    return {
        .number_of_nodes  = number_of_nodes,
        .number_of_edges  = number_of_edges,
        .has_node_weights = has_node_weights,
        .has_edge_weights = has_edge_weights,
    };
}

inline Format parse_header(const std::string& filename) {
    MappedFileToker toker(filename);
    return parse_header(toker);
}

template <typename GraphFormatCB, typename NextNodeCB, typename NextEdgeCB>
void parse(MappedFileToker& toker, GraphFormatCB&& format_cb, NextNodeCB&& next_node_cb, NextEdgeCB&& next_edge_cb) {
    static_assert(std::is_invocable_v<GraphFormatCB, Format>);
    static_assert(std::is_invocable_v<NextNodeCB, std::uint64_t>);
    static_assert(std::is_invocable_v<NextEdgeCB, std::uint64_t, std::uint64_t>);

    constexpr bool stoppable = std::is_invocable_r_v<bool, NextNodeCB, std::uint64_t>;

    const Format format            = parse_header(toker);
    const bool   read_node_weights = format.has_node_weights;
    const bool   read_edge_weights = format.has_edge_weights;
    format_cb(format);

    for (std::uint64_t u = 0; u < format.number_of_nodes; ++u) {
        toker.skip_spaces();
        while (toker.current() == '%') {
            toker.skip_line();
            toker.skip_spaces();
        }

        std::uint64_t node_weight = 1;
        if (format.has_node_weights) {
            if (read_node_weights) {
                node_weight = toker.scan_uint<std::uint64_t>();
            } else {
                toker.scan_uint<std::uint64_t>();
            }
        }
        if constexpr (stoppable) {
            if (!next_node_cb(node_weight)) {
                break;
            }
        } else {
            next_node_cb(node_weight);
        }

        while (std::isdigit(toker.current())) {
            const std::uint64_t v           = toker.scan_uint<std::uint64_t>() - 1;
            std::uint64_t       edge_weight = 1;
            if (format.has_edge_weights) {
                if (read_edge_weights) {
                    edge_weight = toker.scan_uint<std::uint64_t>();
                } else {
                    toker.scan_uint<std::uint64_t>();
                }
            }
            next_edge_cb(edge_weight, v);
        }

        if (toker.current() == '\n') {
            toker.advance();
        }
    }
}

template <typename GraphFormatCB, typename NextNodeCB, typename NextEdgeCB>
void parse(
    const std::string& filename, GraphFormatCB&& format_cb, NextNodeCB&& next_node_cb, NextEdgeCB&& next_edge_cb
) {
    MappedFileToker toker(filename);
    parse(
        toker, std::forward<GraphFormatCB>(format_cb), std::forward<NextNodeCB>(next_node_cb),
        std::forward<NextEdgeCB>(next_edge_cb)
    );
}
} // namespace kaminpar::io::metis
