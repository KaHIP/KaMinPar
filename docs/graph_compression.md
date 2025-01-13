# Graph Compression

KaMinPar/TeraPart offers graph compression to store the input graph with a space-efficient encoding in memory. There are three methods to obtain a compressed graph: compressing an uncompressed graph already in memory, loading and compressing a graph during I/O if it is stored in a supported file format, or using the compressed graph builder interface to create compressed graphs.

## Compress an Uncompressed Graph

We offer both sequential and parallel interfaces for graph compression. While this method is the simplest to use, it is only practical if the input graph can fit in memory without compression, making it only useful in specific situations.

```cpp
#include <kaminpar-shm/graphutils/compressed_graph_builder.h>

using namespace kaminpar::shm;

// To compress a graph in CSR format sequentially.
CompressedGraph compressed_graph = compress(
    std::span<EdgeID> nodes,
    std::span<NodeID> edges,
    std::span<NodeWeight> node_weights = {},
    std::span<EdgeWeight> edge_weights = {}
);

// To compress a graph in CSR format in parallel.
CompressedGraph compressed_graph = parallel_compress(
    std::span<EdgeID> nodes,
    std::span<NodeID> edges,
    std::span<NodeWeight> node_weights = {},
    std::span<EdgeWeight> edge_weights = {}
);
```

## Compress a Graph during I/O

If the graph to compress is stored on disk in METIS or ParHIP format (see [Graph File Format Documentation](/docs/graph_file_format.md)), you can obtain the graph in compressed format by using the graph compression I/O interface.

```cpp
#include <kaminpar-io/parhip_parser.h>

using namespace kaminpar::shm::io;

// To read a graph stored in METIS format.
CompressedGraph compressed_graph = metis::compressed_read(const std::string &filename);

// To read a graph stored in ParHIP format.
CompressedGraph compressed_graph = parhip::compressed_read(const std::string &filename);
```

If the graph is stored in compressed format on disk (see [Graph File Format Documentation](/docs/graph_file_format.md)), then it can be read directly.

```cpp
#include <kaminpar-io/compressed_graph_binary.h>

using namespace kaminpar::shm::io;

// To read a graph stored in compressed format.
CompressedGraph compressed_graph = compressed_binary::read(const std::string &filename);
```

## Compress a Graph using the Builder Interface

The compressed graph builder is the most powerful method of creating a compressed graph as it does not require a specific uncompressed graph layout as opposed to the previous two methods. We provide three builders to construct compressed graphs: a sequential graph builder, a parallel two-pass graph builder, and a parallel single-pass graph builder that requires random access on the graph to compress.

### Sequential Graph Builder

This section outlines how to use the sequential builder to compress graphs. For usage examples, please refer to the end of this section.

To begin using the `CompressedGraphBuilder`, you must first instantiate the class by providing information about the graph to compress. The constructor requires the number of nodes and edges in the graph you intend to compress. Additionally, you need to specify whether the graph includes node or edge weights. There is also an option to indicate if the nodes are stored in degree-bucket order, which is not related to graph compression and can therefore be safely ignored, i.e., set to `false`.

```cpp
#include <kaminpar-shm/graphutils/compressed_graph_builder.h>

using namespace kaminpar::shm;

CompressedGraphBuilder builder(
    NodeID num_nodes,
    EdgeID num_edges,
    bool has_node_weights,
    bool has_edge_weights,
    bool sorted = false
);
```

Once the `CompressedGraphBuilder` is initialized, you can proceed to add nodes to the builder one by one using the `add_node` method. It is required to add the nodes in increasing order, starting from node `0`, followed by node `1`, and so forth. The `add_node` method supports multiple formats for specifying neighborhoods:

```cpp
// Adds the next node by specifying its adjacent nodes.
builder.add_node(
    std::span<NodeID> neighbors
);

// Adds the next node by specifying its adjacent nodes
// and corresponding edge weights.
builder.add_node(
    std::span<std::pair<NodeID, EdgeWeight>> neighborhood
);

// Adds the next node by separately specifying its adjacent
// nodes and corresponding edge weights.
builder.add_node(
    std::span<NodeID> neighbors,
    std::span<EdgeWeight> edge_weights
);
```

In addition to adding nodes, you have the option to assign weights to individual nodes using the `add_node_weight` method. Unlike the `add_node` method, `add_node_weight` does not require nodes to be added in any specific order. Note that this method can only be used if the `CompressedGraphBuilder` was constructed with `has_node_weights` set to `true`. Nodes that are not explicitly assigned a weight will default to a weight of one.

```cpp
builder.add_node_weight(NodeID node, NodeWeight weight);
```

After all nodes and optionally their weights have been added, you finalize the compression process by calling the `build()` method. This method constructs and returns the `CompressedGraph` object. It is important to note that once `build()` is invoked, the `CompressedGraphBuilder` instance cannot be reused to create another compressed graph. If you need to compress a different graph, you must instantiate a new `CompressedGraphBuilder`.

```cpp
CompressedGraph compressed_graph = builder.build();
```

#### Example Usage

```cpp
#include <kaminpar-shm/graphutils/compressed_graph_builder.h>

using namespace kaminpar::shm;

CompressedGraph create_cycle(NodeID num_nodes, NodeWeight node_weight, EdgeWeight edge_weight) {
    CompressedGraphBuilder builder(num_nodes, 2 * num_nodes, true, true);

    std::vector<std::pair<NodeID, EdgeWeight>> neighborhood;
    for (NodeID node = 0; node < num_nodes; ++node) {
        NodeID prev_adjacent_node = (node > 0) ? (u - 1) : (num_nodes - 1);
        NodeID next_adjacent_node = (u + 1) % num_nodes;

        neighborhood.clear();
        neighborhood.emplace_back(prev_adjacent_node, edge_weight);
        neighborhood.emplace_back(next_adjacent_node, edge_weight);

        builder.add_node(neighborhood)
        builder.add_node_weight(node, node_weight);
    }

    return builder.build();
}
```

### Parallel Graph Builder

#### Single-Pass Builder

The single-pass builder allows you to create a compressed graph in one pass. This approach is the most efficient for scenarios where you can provide the necessary information about the neighborhoods in a random-access fashion.

```cpp
#include <kaminpar-shm/graphutils/compressed_graph_builder.h>

using namespace kaminpar::shm;

NodeID num_nodes = ...;
EdgeID num_edges = ...;

// Create a function object that returns the degree of a node upon calling
// (in this case a lambda expression).
auto fetch_degree = [&](NodeID u) -> NodeID {
    /* Return the degree of node u. */
};

// Create a function object that fills a buffer with the adjacent nodes of
// a node upon calling.
auto fetch_neighborhood = [&](NodeID u, std::span<NodeID> adjacent_nodes) {
    /* Fill adjacent_nodes with the neighbors of node u. */
};

// Create a wrapper around the node weights, if present. Note that the memory
// pointed to has to outlive the compressed graph instance.
NodeWeight *vwgt = ...;
StaticArray<NodeWeight> node_weights(num_nodes, vwgt);

CompressedGraph compressed_graph = parallel_compress(
    num_nodes,
    num_edges,
    fetch_degree,
    fetch_neighborhood,
    node_weights
);
```

To compress graphs with edge weights, use the `parallel_compress_weighted` method that requires you to fill a `std::span<std::span<NodeID, EdgeWeight>>` when `fetch_neighborhood` is called.

#### Two-Pass Builder

The two-pass builder allows you to create a compressed graph in two passes.

```cpp
#include <kaminpar-shm/graphutils/compressed_graph_builder.h>

using namespace kaminpar::shm;

ParallelCompressedGraphBuilder builder(
    NodeID num_nodes,
    EdgeID num_edges,
    bool has_node_weights,
    bool has_edge_weights,
    bool sorted = false
);
```

In the first pass, register the neighborhoods of each node by calling `register_neighborhood`. Note that the neighborhoods of the nodes can be registered in an arbitrary order.

```cpp
builder.register_neighborhood(
    NodeID node,
    std::span<NodeID> neighbors
);

builder.register_neighborhood(
    NodeID node,
    std::span<std::pair<NodeID, EdgeWeight>> neighborhood
);

builder.register_neighborhood(
    NodeID node,
    std::span<NodeID> neighbors,
    std::span<EdgeWeight> edge_weights
);
```
After registering all neighborhoods, the offsets into the edge array for each node can be computed.

```cpp
builder.compute_offsets();
```

Then, you can add the neighborhoods of each node by calling `add_neighborhood`.

```cpp
builder.add_neighborhood(
    NodeID node,
    std::span<NodeID> neighbors
);

builder.add_neighborhood(
    NodeID node,
    std::span<std::pair<NodeID, EdgeWeight>> neighborhood
);

builder.add_neighborhood(
    NodeID node,
    std::span<NodeID> neighbors,
    std::span<EdgeWeight> edge_weights
);
```

After all nodes have been added, build the compressed graph. It is important to note that once `build()` is invoked, the `ParallelCompressedGraphBuilder` instance cannot be reused to create another compressed graph. If you need to compress a different graph, you must instantiate a new instance.

```cpp
CompressedGraph compressed_graph = builder.build();
```

##### Examples

```cpp
#include <kaminpar-shm/graphutils/compressed_graph_builder.h>

using namespace kaminpar::shm;

ParallelCompressedGraphBuilder builder(num_nodes, num_edges, has_node_weights, has_edge_weights);

...
```
