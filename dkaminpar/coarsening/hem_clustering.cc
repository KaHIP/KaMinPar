#include "dkaminpar/coarsening/hem_clustering.h"

#include "dkaminpar/algorithms/greedy_node_coloring.h"

#include "common/timer.h"

namespace kaminpar::dist {
HEMClustering::HEMClustering(const Context& ctx) : _input_ctx(ctx), _ctx(ctx.coarsening.hem) {}

void HEMClustering::initialize(const DistributedGraph& graph) {
    mpi::barrier(graph.communicator());

    SCOPED_TIMER("Colored LP refinement");
    SCOPED_TIMER("Initialization");

    const auto coloring = [&] {
        // Graph is already sorted by a coloring -> reconstruct this coloring
        // @todo if we always want to do this, optimize this refiner
        if (graph.color_sorted()) {
            LOG << "Graph sorted by colors: using precomputed coloring";

            NoinitVector<ColorID> coloring(graph.n()); // We do not actually need the colors for ghost nodes

            // @todo parallelize
            NodeID pos = 0;
            for (ColorID c = 0; c < graph.number_of_colors(); ++c) {
                const std::size_t size = graph.color_size(c);
                std::fill(coloring.begin() + pos, coloring.begin() + pos + size, c);
                pos += size;
            }

            return coloring;
        }

        // Otherwise, compute a coloring now
        LOG << "Computing new coloring";
        return compute_node_coloring_sequentially(graph, _ctx.num_coloring_chunks);
    }();

    const ColorID num_local_colors = *std::max_element(coloring.begin(), coloring.end()) + 1;
    const ColorID num_colors       = mpi::allreduce(num_local_colors, MPI_MAX, graph.communicator());

    TIMED_SCOPE("Allocation") {
        _color_sorted_nodes.resize(graph.n());
        _color_sizes.resize(num_colors + 1);
        _color_blacklist.resize(num_colors);
        tbb::parallel_for<std::size_t>(0, _color_sorted_nodes.size(), [&](const std::size_t i) {
            _color_sorted_nodes[i] = 0;
        });
        tbb::parallel_for<std::size_t>(0, _color_sizes.size(), [&](const std::size_t i) { _color_sizes[i] = 0; });
        tbb::parallel_for<std::size_t>(0, _color_blacklist.size(), [&](const std::size_t i) {
            _color_blacklist[i] = 0;
        });
    };

    TIMED_SCOPE("Count color sizes") {
        if (graph.color_sorted()) {
            const auto& color_sizes = graph.get_color_sizes();
            _color_sizes.assign(color_sizes.begin(), color_sizes.end());
        } else {
            graph.pfor_nodes([&](const NodeID u) {
                const ColorID c = coloring[u];
                KASSERT(c < num_colors);
                __atomic_fetch_add(&_color_sizes[c], 1, __ATOMIC_RELAXED);
            });
            parallel::prefix_sum(_color_sizes.begin(), _color_sizes.end(), _color_sizes.begin());
        }
    };

    TIMED_SCOPE("Sort nodes") {
        if (graph.color_sorted()) {
            // @todo parallelize
            std::iota(_color_sorted_nodes.begin(), _color_sorted_nodes.end(), 0);
        } else {
            graph.pfor_nodes([&](const NodeID u) {
                const ColorID     c = coloring[u];
                const std::size_t i = __atomic_sub_fetch(&_color_sizes[c], 1, __ATOMIC_SEQ_CST);
                KASSERT(i < _color_sorted_nodes.size());
                _color_sorted_nodes[i] = u;
            });
        }
    };

    TIMED_SCOPE("Compute color blacklist") {
        if (_ctx.small_color_blacklist == 0
            || (_ctx.only_blacklist_input_level && graph.global_n() != _input_ctx.partition.graph.global_n())) {
            return;
        }

        NoinitVector<GlobalNodeID> global_color_sizes(num_colors);
        tbb::parallel_for<ColorID>(0, num_colors, [&](const ColorID c) {
            global_color_sizes[c] = _color_sizes[c + 1] - _color_sizes[c];
        });
        MPI_Allreduce(
            MPI_IN_PLACE, global_color_sizes.data(), asserting_cast<int>(num_colors), mpi::type::get<GlobalNodeID>(),
            MPI_SUM, graph.communicator()
        );

        // @todo parallelize the rest of this section
        std::vector<ColorID> sorted_by_size(num_colors);
        std::iota(sorted_by_size.begin(), sorted_by_size.end(), 0);
        std::sort(sorted_by_size.begin(), sorted_by_size.end(), [&](const ColorID lhs, const ColorID rhs) {
            return global_color_sizes[lhs] < global_color_sizes[rhs];
        });

        GlobalNodeID excluded_so_far = 0;
        for (const ColorID c: sorted_by_size) {
            excluded_so_far += global_color_sizes[c];
            const double percentage = 1.0 * excluded_so_far / graph.global_n();
            if (percentage <= _ctx.small_color_blacklist) {
                _color_blacklist[c] = 1;
            } else {
                break;
            }
        }
    };

    KASSERT(_color_sizes.front() == 0u);
    KASSERT(_color_sizes.back() == graph.n());
}

const HEMClustering::AtomicClusterArray&
HEMClustering::compute_clustering(const DistributedGraph& graph, GlobalNodeWeight max_cluster_weight) {
    // Compute graph coloring
    initialize(graph);

    ((void)max_cluster_weight);
    return _matching;
}
} // namespace kaminpar::dist
