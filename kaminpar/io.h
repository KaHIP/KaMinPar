/*******************************************************************************
 * @file:   io.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Graph and partition IO functions.
 ******************************************************************************/
#pragma once

#include <cctype>
#include <cerrno>
#include <concepts>
#include <cstring>
#include <fstream>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "kaminpar/datastructure/graph.h"
#include "kaminpar/definitions.h"

namespace kaminpar::io {
namespace internal {
struct MappedFile {
    const int         fd;
    std::size_t       position;
    const std::size_t length;
    char*             contents;

    [[nodiscard]] inline bool valid_position() const {
        return position < length;
    }
    [[nodiscard]] inline char current() const {
        return contents[position];
    }
    inline void advance() {
        ++position;
    }
};

inline int open_file(const std::string& filename) {
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd < 0)
        FATAL_PERROR << "Error while opening " << filename;
    return fd;
}

inline std::size_t file_size(const int fd) {
    struct stat file_info {};
    if (fstat(fd, &file_info) == -1) {
        close(fd);
        FATAL_PERROR << "Error while determining file size";
    }
    return static_cast<std::size_t>(file_info.st_size);
}

inline MappedFile mmap_file_from_disk(const std::string& filename) {
    const int         fd     = open_file(filename);
    const std::size_t length = file_size(fd);

    char* contents = static_cast<char*>(mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd, 0));
    if (contents == MAP_FAILED) {
        close(fd);
        FATAL_PERROR << "Error while mapping file to memory";
    }

    return {
        .fd       = fd,
        .position = 0,
        .length   = length,
        .contents = contents,
    };
}

inline void munmap_file_from_disk(const MappedFile& mapped_file) {
    if (munmap(mapped_file.contents, mapped_file.length) == -1) {
        close(mapped_file.fd);
        FATAL_PERROR << "Error while unmapping file from memory";
    }
    close(mapped_file.fd);
}

inline void skip_spaces(MappedFile& mapped_file) {
    while (mapped_file.valid_position() && mapped_file.current() == ' ') {
        mapped_file.advance();
    }
}

inline void skip_comment(MappedFile& mapped_file) {
    while (mapped_file.valid_position() && mapped_file.current() != '\n') {
        mapped_file.advance();
    }
    if (mapped_file.valid_position()) {
        ASSERT(mapped_file.current() == '\n');
        mapped_file.advance();
    }
}

inline void skip_nl(MappedFile& mapped_file) {
    ASSERT(mapped_file.valid_position() && mapped_file.current() == '\n');
    mapped_file.advance();
}

inline std::uint64_t scan_uint(MappedFile& mapped_file) {
    std::uint64_t number = 0;
    while (mapped_file.valid_position() && std::isdigit(mapped_file.current())) {
        const int digit = mapped_file.current() - '0';
        number          = number * 10 + digit;
        mapped_file.advance();
    }
    skip_spaces(mapped_file);
    return number;
}
} // namespace internal

namespace metis {
struct GraphInfo {
    std::uint64_t total_node_weight;
    std::uint64_t total_edge_weight;
    bool          has_isolated_nodes;
};

struct GraphFormat {
    std::uint64_t number_of_nodes;
    std::uint64_t number_of_edges;
    bool          has_node_weights;
    bool          has_edge_weights;
};

inline GraphFormat read_graph_header(internal::MappedFile& mapped_file) {
    skip_spaces(mapped_file);
    while (mapped_file.current() == '%') {
        skip_comment(mapped_file);
        skip_spaces(mapped_file);
    }

    const std::uint64_t number_of_nodes = scan_uint(mapped_file);
    const std::uint64_t number_of_edges = scan_uint(mapped_file);
    const std::uint64_t format          = (mapped_file.current() != '\n') ? scan_uint(mapped_file) : 0;
    skip_nl(mapped_file);

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

Graph     read(const std::string& filename, bool ignore_node_weights = false, bool ignore_edge_weights = false);
GraphInfo read(
    const std::string& filename, StaticArray<EdgeID>& nodes, StaticArray<NodeID>& edges,
    StaticArray<NodeWeight>& node_weights, StaticArray<EdgeWeight>& edge_weights);
void read_format(const std::string& filename, NodeID& n, EdgeID& m, bool& has_node_weights, bool& has_edge_weights);
GraphFormat read_format(const std::string& filename);
void        write(const std::string& filename, const Graph& graph, const std::string& comment = "");

template <typename GraphFormatCB, typename NextNodeCB, typename NextEdgeCB>
GraphInfo read_observable(
    const std::string& filename, GraphFormatCB&& format_cb, NextNodeCB&& next_node_cb, NextEdgeCB&& next_edge_cb) {
    static_assert(std::is_invocable_v<GraphFormatCB, GraphFormat>);
    static_assert(std::is_invocable_v<NextNodeCB, std::uint64_t>);
    static_assert(std::is_invocable_v<NextEdgeCB, std::uint64_t, std::uint64_t>);

    using namespace internal;
    constexpr bool stoppable = std::is_invocable_r_v<bool, NextNodeCB, std::uint64_t>;

    MappedFile        mapped_file       = mmap_file_from_disk(filename);
    const GraphFormat format            = metis::read_graph_header(mapped_file);
    const bool        read_node_weights = format.has_node_weights;
    const bool        read_edge_weights = format.has_edge_weights;
    format_cb(format);

    GraphInfo info{};
    bool      unit_node_weights = true;
    bool      unit_edge_weights = true;

    for (NodeID u = 0; u < format.number_of_nodes; ++u) {
        skip_spaces(mapped_file);
        while (mapped_file.current() == '%') {
            skip_comment(mapped_file);
            skip_spaces(mapped_file);
        }

        std::uint64_t node_weight = 1;
        if (format.has_node_weights) {
            if (read_node_weights) {
                node_weight       = scan_uint(mapped_file);
                unit_node_weights = unit_node_weights && node_weight == 1;
                info.total_node_weight += node_weight;
            } else {
                scan_uint(mapped_file);
            }
        }
        if constexpr (stoppable) {
            if (!next_node_cb(node_weight)) {
                break;
            }
        } else {
            next_node_cb(node_weight);
        }

        const bool isolated_node = !std::isdigit(mapped_file.current());
        while (std::isdigit(mapped_file.current())) {
            const std::uint64_t v           = scan_uint(mapped_file) - 1;
            std::uint64_t       edge_weight = 1;
            if (format.has_edge_weights) {
                if (read_edge_weights) {
                    edge_weight       = scan_uint(mapped_file);
                    unit_edge_weights = unit_edge_weights && edge_weight == 1;
                    info.total_edge_weight += edge_weight;
                } else {
                    scan_uint(mapped_file);
                }
            }
            next_edge_cb(edge_weight, v);
        }
        info.has_isolated_nodes |= isolated_node;

        if (mapped_file.current() == '\n') {
            skip_nl(mapped_file);
        }
    }

    munmap_file_from_disk(mapped_file);

    if (!read_node_weights) {
        info.total_node_weight = format.number_of_nodes;
    }
    if (!read_edge_weights) {
        info.total_edge_weight = 2 * format.number_of_edges;
    }
    return info;
}
} // namespace metis

namespace partition {
void write(const std::string& filename, const std::vector<BlockID>& partition);
void write(const std::string& filename, const PartitionedGraph& p_graph);
void write(const std::string& filename, const StaticArray<BlockID>& partition, const StaticArray<NodeID>& permutation);
void write(const std::string& filename, const PartitionedGraph& p_graph, const StaticArray<NodeID>& permutation);

template <typename Container = std::vector<BlockID>>
Container read(const std::string& filename) {
    using namespace internal;

    auto                 mapped_file = mmap_file_from_disk(filename);
    std::vector<BlockID> partition;
    while (mapped_file.valid_position()) {
        partition.push_back(scan_uint(mapped_file));
        skip_nl(mapped_file);
    }
    munmap_file_from_disk(mapped_file);

    if constexpr (std::is_same_v<Container, std::vector<BlockID>>) {
        return partition;
    } else {
        Container copy(partition.size());
        std::copy(partition.begin(), partition.end(), copy.begin());
        return copy;
    }
}
std::vector<BlockID> read(const std::string& filename);
} // namespace partition
} // namespace kaminpar::io
