#pragma once

#include <gtest/gtest.h>

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/datastructure/distributed_graph_builder.h"
#include "dkaminpar/definitions.h"
#include "dkaminpar/mpi/wrapper.h"

namespace dkaminpar::testing::fixtures {
class DistributedTestFixture : public ::testing::Test {
protected:
    void SetUp() override {
        std::tie(size, rank) = mpi::get_comm_info(MPI_COMM_WORLD);
    }

    PEID rank;
    PEID size;
};

/*
 * Each PE has a single vertex without any edges.
 */
class DistributedIsolatedNodesGraphFixture : public DistributedTestFixture {
protected:
    void SetUp() override {
        DistributedTestFixture::SetUp();
        graph = create_graph();
    }

private:
    DistributedGraph create_graph() {
        dkaminpar::graph::Builder builder(MPI_COMM_WORLD);
        builder.initialize(1);
        builder.create_node(1);
        return builder.finalize();
    }

protected:
    DistributedGraph graph;
};

/*
 * Each PE has a single node that are connected in a ring.
 */
class DistributedCircleGraphFixture : public DistributedTestFixture {
protected:
    void SetUp() override {
        DistributedTestFixture::SetUp();
        graph = create_graph();
    }

private:
    DistributedGraph create_graph() {
        dkaminpar::graph::Builder builder(MPI_COMM_WORLD);
        builder.initialize(1);

        const GlobalNodeID prev = static_cast<GlobalNodeID>(rank > 0 ? rank - 1 : size - 1);
        const GlobalNodeID next = static_cast<GlobalNodeID>((rank + 1) % size);

        builder.create_node(1);
        if (prev != next) {
            builder.create_edge(1, prev);
            builder.create_edge(1, next);
        }

        return builder.finalize();
    }

protected:
    DistributedGraph graph;
};

/*
 * Each PE has a local triangle. Additionally, each node is connected to the same node on the
 * next and previous PE.
 * On a single PE, this is just a triangle.
 */
class DistributedTrianglesGraphFixture : public DistributedTestFixture {
protected:
    void SetUp() override {
        DistributedTestFixture::SetUp();
        num_nodes_per_pe = 3;
        my_n0            = rank * num_nodes_per_pe;
        prev_n0          = rank > 0 ? (rank - 1) * num_nodes_per_pe : (size - 1) * num_nodes_per_pe;
        next_n0          = rank + 1 < size ? (rank + 1) * num_nodes_per_pe : 0;
        graph            = create_graph();
    }

private:
    DistributedGraph create_graph() {
        dkaminpar::graph::Builder builder(MPI_COMM_WORLD);
        builder.initialize(num_nodes_per_pe);

        // PE 0: 2 <-- 0 --> 1
        builder.create_node(1);
        builder.create_edge(1, my_n0 + 1);
        builder.create_edge(1, my_n0 + num_nodes_per_pe - 1);
        if (size > 1) {
            builder.create_edge(1, next_n0);
            builder.create_edge(1, prev_n0);
        }

        // PE 0: 2 <-- 1 --> 0
        builder.create_node(1);
        builder.create_edge(1, my_n0);
        builder.create_edge(1, my_n0 + num_nodes_per_pe - 1);
        if (size > 1) {
            builder.create_edge(1, next_n0 + 1);
            builder.create_edge(1, prev_n0 + 1);
        }

        // PE 0: 0 <-- 2 --> 1
        builder.create_node(1);
        builder.create_edge(1, my_n0);
        builder.create_edge(1, my_n0 + 1);
        if (size > 1) {
            builder.create_edge(1, next_n0 + 2);
            builder.create_edge(1, prev_n0 + 2);
        }

        return builder.finalize();
    }

protected:
    PEID             rank;
    PEID             size;
    DistributedGraph graph;
    GlobalNodeID     num_nodes_per_pe;
    GlobalNodeID     my_n0;
    GlobalNodeID     prev_n0;
    GlobalNodeID     next_n0;
};
} // namespace dkaminpar::testing::fixtures
