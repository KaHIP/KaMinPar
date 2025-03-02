# Graph File Formats

This document provides an overview of the supported graph file formats. It includes detailed descriptions of the METIS and ParHIP formats as well as the format used to store compressed graphs.

## METIS Graph File Format

The METIS graph file format is a text-based format to represent graphs. It starts with a header line, followed by lines that describe each node and its adjacency list.

The header line in a METIS file contains two or three numbers. The first number indicates the number of nodes in the graph, while the second number specifies the number of edges. The optional third number denotes the whether the graph is weighted. If this third number is omitted or set to 00, the graph is unweighted. If it is set to 10, the graph includes node weights. If it is set to 01, the graph includes edge weights. If it is set to 11, the graph includes both node and edge weights.

After the header, each subsequent line represents a node and lists its adjacent nodes. The adjacent nodes are identified by their node IDs starting at one, where the IDs correspond to the node order specified by the lines of the file. If node weights are present, an additional number at the beginning of the line indicates the node's weight. If edge weights are present, the adjacency list consists of pairs of numbers, where the first number is the node ID and the second number is the edge weight.

```
+------------+---------------------------------------------------------------------------------------+
| Line Desc. | Line Content                                                                          |
|------------+---------------------------------------------------------------------------------------+
| Header     | num_nodes num_edges [weights_flag]                                                    |
| Node 1     | [node_weight] first_neighbor [first_edge_weight] ... last_neighbor [last_edge_weight] |
| Node 2     | [node_weight] first_neighbor [first_edge_weight] ... last_neighbor [last_edge_weight] |
| ...        | ...                                                                                   |
| Node n     | [node_weight] first_neighbor [first_edge_weight] ... last_neighbor [last_edge_weight] |
+--------+-------------------------------------------------------------------------------------------+
```

## ParHIP Graph File Format

The ParHIP graph file format is a binary format to represent graphs. The format includes a header, offsets, adjacency lists, node weights, and edge weights.

The header is 24 bytes long and contains three 8-byte fields: a version, number of nodes, and number of edges. The version field is a bit-field that encodes information about the presence of edge weights, node weights, and the bit-width of IDs:
- The least significant bit indicates whether edge weights are present (0) or not (1).
- The next bit indicates whether node weights are present (0) or not (1).
- The next bit indicates whether edge IDs are 32-bit (1) or 64-bit (0).
- The next bit indicates whether node IDs are 32-bit (1) or 64-bit (0).
- The next bit indicates whether node weights are 32-bit (1) or 64-bit (0).
- The most significant bit indicates whether edge weights are 32-bit (1) or 64-bit (0).

The offsets section contains addresses relative to the start of the file, such that each offset points to the first neighbor of a node in the adjacency lists. The adjacency lists section contains the adjacent nodes for each node. The node weights section contains the weights of the nodes, if present. The edge weights section contains the weights of the edges, if present.
```
+------------------------------------+
| Header (24 bytes)                  |
+------------------------------------+
| Offsets ([n + 1] * EID bytes)      |
+------------------------------------+
| Adjacency lists (m * NID bytes)    |
+------------------------------------+
| Node weights (n * NWGT bytes)      |
+------------------------------------+
| Edge weights (m * EWGT bytes)      |
+------------------------------------+
```

## Compressed Graph File Format

The compressed graph file format is a binary format to represent compressed graphs. This format includes several sections: a header, offsets, compressed adjacency lists, and optional node weights. The header starts with a magic number and contains metadata about the graph, such as the number of nodes and edges, whether the graph is weighted, whether 64-bit IDs are used, and additional information about the compression format. The offsets section contains the starting position (in bytes) of each node's adjacency list in the compressed adjacency lists section. The compressed adjacency lists section contains the adjacency lists of all nodes in a compressed form. If node weights are present, this section also stores the weights of adjacent nodes for each node.

To read and write a compresed graph, you can use the methods provided in KaMinPar's I/O module. For information on how to obtain a compressed graph for storing, please refer to the [Graph Compression Documentation](/docs/graph_compression.md).

```cpp
#include <kaminpar-io/graph_compression_binary.h>

using namespace kaminpar::shm::io;

std::optional<CompressedGraph> compressed_binary::read(const std::string &filename);

void compressed_binary::write(const std::string &filename, const CompressedGraph &graph);
```
