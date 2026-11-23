/*******************************************************************************
 * Common preprocessing utilities for cluster contraction implementations.
 *
 * @file:   cluster_contraction_preprocessing.cc
 * @author: Daniel Seemaier
 * @author: Daniel Salwasser
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/coarsening/contraction/cluster_contraction_preprocessing.h"

#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::contraction {

void fill_leader_mapping(
    const Graph &graph, const StaticArray<NodeID> &clustering, StaticArray<NodeID> &leader_mapping
) {
  reified(graph, [&](const auto &graph) {
    TIMED_SCOPE("Allocation") {
      if (leader_mapping.size() < graph.n()) {
        leader_mapping.resize(graph.n(), static_array::noinit);
      }
    };

    TIMED_SCOPE("Preprocessing") {
      NodeID *__restrict__ lm_ptr = leader_mapping.data();
      const NodeID *__restrict__ cl_ptr = clustering.data();
      tbb::parallel_for(tbb::blocked_range<NodeID>(0, graph.n(), 4096), [&](const auto &r) {
        for (NodeID u = r.begin(); u != r.end(); ++u) {
          lm_ptr[u] = 0;
        }
      });
      tbb::parallel_for(tbb::blocked_range<NodeID>(0, graph.n(), 4096), [&](const auto &r) {
        for (NodeID u = r.begin(); u != r.end(); ++u) {
          const NodeID c = cl_ptr[u];
          if (__atomic_load_n(&lm_ptr[c], __ATOMIC_RELAXED) == 0) {
            __atomic_store_n(&lm_ptr[c], 1, __ATOMIC_RELAXED);
          }
        }
      });
      parallel::prefix_sum(
          leader_mapping.begin(), leader_mapping.begin() + graph.n(), leader_mapping.begin()
      );
    };
  });
}

StaticArray<NodeID> compute_mapping(
    const Graph &graph, StaticArray<NodeID> clustering, const StaticArray<NodeID> &leader_mapping
) {
  reified(graph, [&](const auto &graph) {
    TIMED_SCOPE("Preprocessing") {
      const NodeID *__restrict__ lm_ptr = leader_mapping.data();
      NodeID *__restrict__ cl_ptr = clustering.data();
      tbb::parallel_for(tbb::blocked_range<NodeID>(0, graph.n(), 4096), [&](const auto &r) {
        for (NodeID u = r.begin(); u != r.end(); ++u) {
          cl_ptr[u] = lm_ptr[cl_ptr[u]] - 1;
        }
      });
    };
  });

  return clustering;
}

std::pair<NodeID, StaticArray<NodeID>>
compute_mapping(const Graph &graph, StaticArray<NodeID> clustering, MemoryContext &m_ctx) {
  SCOPED_HEAP_PROFILER("Compute mapping");

  fill_leader_mapping(graph, clustering, m_ctx.leader_mapping);
  StaticArray<NodeID> mapping = compute_mapping(graph, std::move(clustering), m_ctx.leader_mapping);
  const NodeID c_n = m_ctx.leader_mapping[graph.n() - 1];

  TIMED_SCOPE("Deallocation") {
    m_ctx.leader_mapping.free();
  };

  return {c_n, std::move(mapping)};
}

void fill_cluster_buckets(
    const NodeID c_n,
    const Graph &graph,
    const StaticArray<NodeID> &mapping,
    StaticArray<NodeID> &buckets_index,
    StaticArray<NodeID> &buckets
) {
  SCOPED_HEAP_PROFILER("Fill cluster buckets");

  TIMED_SCOPE("Allocation") {
    if (buckets.size() < graph.n()) {
      buckets.resize(graph.n(), static_array::noinit);
    }

    if (buckets_index.size() < c_n + 1) {
      buckets_index.resize(c_n + 1, static_array::noinit);
    }
  };

  TIMED_SCOPE("Preprocessing") {
    NodeID *__restrict__ bi_ptr = buckets_index.data();
    NodeID *__restrict__ b_ptr = buckets.data();
    const NodeID *__restrict__ map_ptr = mapping.data();
    const NodeID n = graph.n();

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, c_n + 1, 4096), [&](const auto &r) {
      for (NodeID i = r.begin(); i != r.end(); ++i) {
        bi_ptr[i] = 0;
      }
    });
    tbb::parallel_for(tbb::blocked_range<NodeID>(0, n, 4096), [&](const auto &r) {
      const NodeID end = r.end();
      NodeID u = r.begin();
      while (u < end) {
        const NodeID c = map_ptr[u];
        NodeID run = 1;
        while (u + run < end && map_ptr[u + run] == c) {
          ++run;
        }
        __atomic_fetch_add(&bi_ptr[c], run, __ATOMIC_RELAXED);
        u += run;
      }
    });
    parallel::prefix_sum(
        buckets_index.begin(), buckets_index.begin() + c_n + 1, buckets_index.begin()
    );
    tbb::parallel_for(tbb::blocked_range<NodeID>(0, n, 4096), [&](const auto &r) {
      const NodeID end = r.end();
      NodeID u = r.begin();
      while (u < end) {
        const NodeID c = map_ptr[u];
        NodeID run = 1;
        while (u + run < end && map_ptr[u + run] == c) {
          ++run;
        }
        NodeID pos = __atomic_sub_fetch(&bi_ptr[c], run, __ATOMIC_RELAXED);
        for (NodeID k = 0; k < run; ++k) {
          b_ptr[pos + k] = u + k;
        }
        u += run;
      }
    });
  };
}

} // namespace kaminpar::shm::contraction
