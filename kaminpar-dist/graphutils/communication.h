/*******************************************************************************
 * Collective MPI operations for graphs.
 *
 * @file:   graph_communication.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 ******************************************************************************/
#pragma once

#include <type_traits>

#include <tbb/task_arena.h>

#include "kaminpar-mpi/sparse_alltoall.h"
#include "kaminpar-mpi/utils.h"

#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/timer.h"

#include "kaminpar-common/datastructures/cache_aligned_vector.h"
#include "kaminpar-common/datastructures/marker.h"
#include "kaminpar-common/datastructures/noinit_vector.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/parallel/loops.h"

#define SPARSE_ALLTOALL_NOFILTER                                                                   \
  [](NodeID) {                                                                                     \
    return true;                                                                                   \
  }

namespace kaminpar::mpi::graph {

using namespace kaminpar::dist;
SET_DEBUG(false);

namespace internal {

template <typename Data> void inclusive_col_prefix_sum(Data &data) {
  if (data.empty()) {
    return;
  }

  const std::size_t height = data.size();
  const std::size_t width = data.front().size();

  for (std::size_t i = 1; i < height; ++i) {
    for (std::size_t j = 0; j < width; ++j) {
      data[i][j] += data[i - 1][j];
    }
  }
}

} // namespace internal

/**
 * All-to-all communication between interface vertices and their ghost neighbors.
 *
 * This function exchanges a message for each pair of (a) interface node and (b) a ghost node
 * neighbor of the interface vertex. In other words, this exchanges one message for each cut edge of
 * the distributed graph.
 *
 * This function is farely generic:
 *
 * - The `from` and `to` parameters can be used to specify a range of interface nodes. This can
 * be useful to exchange messages after batch-wise computation.
 *
 * - The `mapper` parameter can be used to change the iteration order of nodes.
 *
 * - The `filter` parameter can be used to ignore some interface nodes, e.g., ignore interface
 * nodes for which no property has changed.
 *
 * - The `builder` and `receiver` parameters are used to construct and receive the messages.
 *
 * There are several overloads which default some of the parameters. The related functions with
 * `_get` suffix can be used to obtain the received message as return value rather than having them
 * passed to a lambda.
 *
 * @tparam Message The type of the message to be exchanged.
 * @tparam Buffer The vector type to be used for receiving the messages.
 *
 * @param graph The distributed graph.
 *
 * @param from Only consider interface nodes with ID >= `from`.
 *
 * @param to Only consider interface nodes with ID < `to`.
 *
 * @param mapper A function to map node ID from range `[from, to)` to any other ID. This allows
 * batch-wise communication when iterating over the graph in a different order. The expected
 * signature of the lambda is as follows:
 * ```
 * NodeID mapper(const NodeID seq_u);
 * ```
 *
 * @param filter A function to filter interface nodes for which no message should be exchanged
 * (return `false`). The expected signature of the lambda is as follows:
 * ```
 * bool filter(const NodeID u [, const EdgeID e, const NodeID v]);
 * ```
 *
 * @param builder A function to construct the message send for some interface node. The expected
 * signature of the lambda is as follows:
 * ```
 * Message builder(const NodeID u, const EdgeID e, const NodeID v [, const PEID to_pe]);
 * ```
 *
 * @param receiver A function invoked for each other PE, passing the received messages from that PE.
 * The expected signature of the lambda is as follows:
 * ```
 * void receiver(Buffer recv_buffer [, const PEID from_pe]);
 * ```
 */
template <
    typename Message,
    typename Buffer = NoinitVector<Message>,
    typename Graph,
    typename Mapper,
    typename Filter,
    typename Builder,
    typename Receiver>
void sparse_alltoall_interface_to_ghost_custom_range(
    const Graph &graph,
    const NodeID from,
    const NodeID to,
    Mapper &&mapper,
    Filter &&filter,
    Builder &&builder,
    Receiver &&receiver
) {
  TIMER_BARRIER(graph.communicator());
  SCOPED_TIMER("Sparse AllToAll");

  constexpr bool builder_invocable_with_pe =
      std::is_invocable_r_v<Message, Builder, NodeID, EdgeID, NodeID, EdgeWeight, PEID>;
  constexpr bool builder_invocable_without_pe =
      std::is_invocable_r_v<Message, Builder, NodeID, EdgeID, NodeID, EdgeWeight>;
  static_assert(builder_invocable_with_pe || builder_invocable_without_pe, "bad builder type");

  constexpr bool receiver_invocable_with_pe =
      std::is_invocable_r_v<void, Receiver, const Buffer &, PEID>;
  constexpr bool receiver_invocable_without_pe =
      std::is_invocable_r_v<void, Receiver, const Buffer &>;
  static_assert(receiver_invocable_with_pe || receiver_invocable_without_pe, "bad receiver type");

  constexpr bool filter_invocable_with_edge =
      std::is_invocable_r_v<bool, Filter, NodeID, EdgeID, NodeID, EdgeWeight>;
  constexpr bool filter_invocable_with_node = std::is_invocable_r_v<bool, Filter, NodeID>;
  static_assert(filter_invocable_with_edge || filter_invocable_with_node, "bad filter type");

  const auto [size, rank] = mpi::get_comm_info(graph.communicator());

  // Allocate message counters
  const PEID num_threads = tbb::this_task_arena::max_concurrency();
  std::vector<CacheAlignedVector<std::size_t>> num_messages(
      num_threads, CacheAlignedVector<std::size_t>(size)
  );

  // Count messages to each PE for each thread
  parallel::deterministic_for<NodeID>(
      from,
      to,
      [&](const NodeID range_from, const NodeID range_to, const PEID thread) {
        for (NodeID seq_u = range_from; seq_u < range_to; ++seq_u) {
          const NodeID u = mapper(seq_u);

          if constexpr (filter_invocable_with_node) {
            if (!filter(u)) {
              continue;
            }
          }

          graph.neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
            if (graph.is_ghost_node(v)) {
              if constexpr (filter_invocable_with_edge) {
                if (!filter(u, e, v, w)) {
                  return;
                }
              }

              const PEID owner = graph.ghost_owner(v);
              ++num_messages[thread][owner];
            }
          });
        }
      }
  );

  // Offset messages for each thread
  internal::inclusive_col_prefix_sum(num_messages);

  // Allocate send buffers
  std::vector<Buffer> send_buffers(size);
  tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
    send_buffers[pe].resize(num_messages.back()[pe]);
  });

  parallel::deterministic_for<NodeID>(
      from,
      to,
      [&](const NodeID range_from, const NodeID range_to, const PEID thread) {
        for (NodeID seq_u = range_from; seq_u < range_to; ++seq_u) {
          const NodeID u = mapper(seq_u);

          if constexpr (filter_invocable_with_node) {
            if (!filter(u)) {
              continue;
            }
          }

          graph.neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
            if (graph.is_ghost_node(v)) {
              if constexpr (filter_invocable_with_edge) {
                if (!filter(u, e, v, w)) {
                  return;
                }
              }

              const PEID pe = graph.ghost_owner(v);
              const std::size_t slot = --num_messages[thread][pe];
              if constexpr (builder_invocable_with_pe) {
                send_buffers[pe][slot] = builder(u, e, v, w, pe);
              } else /* if (builder_invocable_without_pe) */ {
                send_buffers[pe][slot] = builder(u, e, v, w);
              }
            }
          });
        }
      }
  );

  sparse_alltoall<Message, Buffer>(
      std::move(send_buffers), std::forward<decltype(receiver)>(receiver), graph.communicator()
  );
}

template <
    typename Message,
    typename Buffer = NoinitVector<Message>,
    typename Graph,
    typename Filter,
    typename Builder,
    typename Receiver>
void sparse_alltoall_interface_to_ghost(
    const Graph &graph,
    const NodeID from,
    const NodeID to,
    Filter &&filter,
    Builder &&builder,
    Receiver &&receiver
) {
  sparse_alltoall_interface_to_ghost_custom_range<Message, Buffer>(
      graph,
      from,
      to,
      [](const NodeID u) { return u; },
      std::forward<Filter>(filter),
      std::forward<Builder>(builder),
      std::forward<Receiver>(receiver)
  );
}

template <
    typename Message,
    typename Buffer = NoinitVector<Message>,
    typename Graph,
    typename Mapper,
    typename Filter,
    typename Builder>
std::vector<Buffer> sparse_alltoall_interface_to_ghost_custom_range_get(
    const Graph &graph,
    const NodeID from,
    const NodeID to,
    Mapper &&mapper,
    Filter &&filter,
    Builder &&builder
) {
  std::vector<Buffer> recv_buffers(mpi::get_comm_size(graph.communicator()));
  sparse_alltoall_interface_to_ghost_custom_range<Message, Buffer>(
      graph,
      from,
      to,
      std::forward<Mapper>(mapper),
      std::forward<Filter>(filter),
      std::forward<Builder>(builder),
      [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); }
  );
  return recv_buffers;
}

template <
    typename Message,
    typename Buffer = NoinitVector<Message>,
    typename Graph,
    typename Filter,
    typename Builder,
    typename Receiver>
void sparse_alltoall_interface_to_ghost(
    const Graph &graph, Filter &&filter, Builder &&builder, Receiver &&receiver
) {
  sparse_alltoall_interface_to_ghost<Message, Buffer>(
      graph,
      0,
      graph.n(),
      std::forward<Filter>(filter),
      std::forward<Builder>(builder),
      std::forward<Receiver>(receiver)
  );
}

template <
    typename Message,
    typename Buffer = NoinitVector<Message>,
    typename Graph,
    typename Builder,
    typename Receiver>
void sparse_alltoall_interface_to_ghost(
    const Graph &graph, Builder &&builder, Receiver &&receiver
) {
  sparse_alltoall_interface_to_ghost<Message, Buffer>(
      graph,
      SPARSE_ALLTOALL_NOFILTER,
      std::forward<Builder>(builder),
      std::forward<Receiver>(receiver)
  );
}

template <
    typename Message,
    typename Buffer = NoinitVector<Message>,
    typename Graph,
    typename Filter,
    typename Builder>
std::vector<Buffer> sparse_alltoall_interface_to_ghost_get(
    const Graph &graph, const NodeID from, const NodeID to, Filter &&filter, Builder &&builder
) {
  std::vector<Buffer> recv_buffers(mpi::get_comm_size(graph.communicator()));
  sparse_alltoall_interface_to_ghost<Message, Buffer>(
      graph,
      from,
      to,
      std::forward<Filter>(filter),
      std::forward<Builder>(builder),
      [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); }
  );
  return recv_buffers;
}

template <
    typename Message,
    typename Buffer = NoinitVector<Message>,
    typename Graph,
    typename Filter,
    typename Builder>
std::vector<Buffer>
sparse_alltoall_interface_to_ghost_get(const Graph &graph, Filter &&filter, Builder &&builder) {
  std::vector<Buffer> recv_buffers(mpi::get_comm_size(graph.communicator()));
  sparse_alltoall_interface_to_ghost<Message, Buffer>(
      graph,
      std::forward<Filter>(filter),
      std::forward<Builder>(builder),
      [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); }
  );
  return recv_buffers;
}

template <
    typename Message,
    typename Buffer = NoinitVector<Message>,
    typename Graph,
    typename Builder>
std::vector<Buffer> sparse_alltoall_interface_to_ghost_get(const Graph &graph, Builder &&builder) {
  std::vector<Buffer> recv_buffers(mpi::get_comm_size(graph.communicator()));
  sparse_alltoall_interface_to_ghost<Message, Buffer>(
      graph,
      std::forward<Builder>(builder),
      [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); }
  );
  return recv_buffers;
}

/**
 * All-to-all communication between interface vertices and PEs containing a ghost replicate of the
 * interface vertex.
 *
 * This function exchanges a message for each pair of (a) interface vertex and (b) PE containing a
 * ghost replicate of the interface vertex. Typically, this is used to update ghost vertices after
 * changing some property of the interface vertex.
 *
 * This function is farely generic:
 *
 * - The `from` and `to` parameters can be used to specify a range of interface vertices. This can
 * be useful to exchange messages after batch-wise computation.
 *
 * - The `mapper` parameter can be used to change the iteration order of vertices.
 *
 * - The `filter` parameter can be used to ignore some interface vertices, e.g., ignore interface
 * vertices for which no property has changed.
 *
 * - The `builder` and `receiver` parameters are used to construct and receive the messages.
 *
 * There are several overloads which default some of the parameters. The related functions with
 * `_get` suffix can be used to obtain the received message as return value rather than having them
 * passed to a lambda.
 *
 * @tparam Message The type of the message to be exchanged.
 * @tparam Buffer The vector type to be used for receiving the messages.
 *
 * @param graph The distributed graph.
 *
 * @param from Only consider interface vertices with ID >= `from`.
 *
 * @param to Only consider interface vertices with ID < `to`.
 *
 * @param mapper A function to map vertex ID from range `[from, to)` to any other ID. This allows
 * batch-wise communication when iterating over the graph in a different order. The expected
 * signature of the lambda is as follows:
 * ```
 * NodeID mapper(const NodeID seq_u);
 * ```
 *
 * @param filter A function to filter interface vertices for which no message should be exchanged
 * (return `false`). The expected signature of the lambda is as follows:
 * ```
 * bool filter(const NodeID u);
 * ```
 *
 * @param builder A function to construct the message send for some interface vertex. The expected
 * signature of the lambda is as follows:
 * ```
 * Message builder(const NodeID u, [, const PEID to_pe]);
 * ```
 *
 * @param receiver A function invoked for each other PE, passing the received messages from that PE.
 * The expected signature of the lambda is as follows:
 * ```
 * void receiver(Buffer recv_buffer [, const PEID from_pe]);
 * ```
 */
template <
    typename Message,
    typename Buffer = NoinitVector<Message>,
    typename Graph,
    typename Mapper,
    typename Filter,
    typename Builder,
    typename Receiver>
void sparse_alltoall_interface_to_pe_custom_range(
    const Graph &graph,
    const NodeID from,
    const NodeID to,
    Mapper &&mapper,
    Filter &&filter,
    Builder &&builder,
    Receiver &&receiver
) {
  TIMER_BARRIER(graph.communicator());
  SCOPED_TIMER("Sparse AllToAll");

  constexpr bool builder_invocable_with_pe = std::is_invocable_r_v<Message, Builder, NodeID, PEID>;
  constexpr bool builder_invocable_with_pe_and_unmapped_node =
      std::is_invocable_r_v<Message, Builder, NodeID, NodeID, PEID>;
  constexpr bool builder_invocable_without_pe = std::is_invocable_r_v<Message, Builder, NodeID>;
  static_assert(
      builder_invocable_with_pe || builder_invocable_with_pe_and_unmapped_node ||
          builder_invocable_without_pe,
      "bad builder type"
  );

  constexpr bool filter_invocable_with_unmapped_node =
      std::is_invocable_r_v<bool, Filter, NodeID, NodeID>;
  constexpr bool filter_invocable_without_unmapped_node =
      std::is_invocable_r_v<bool, Filter, NodeID>;
  static_assert(filter_invocable_with_unmapped_node || filter_invocable_without_unmapped_node);

  const PEID size = mpi::get_comm_size(graph.communicator());

  // START_TIMER("Message construction");

  // Allocate message counters
  const PEID num_threads = tbb::this_task_arena::max_concurrency();
  std::vector<CacheAlignedVector<std::size_t>> num_messages(
      num_threads, CacheAlignedVector<std::size_t>(size)
  );

  parallel::deterministic_for<NodeID>(
      from,
      to,
      [&](const NodeID range_from, const NodeID range_to, const PEID thread) {
        Marker<> created_message_for_pe(static_cast<std::size_t>(size));

        for (NodeID seq_u = range_from; seq_u < range_to; ++seq_u) {
          const NodeID u = mapper(seq_u);

          if constexpr (filter_invocable_with_unmapped_node) {
            if (!filter(seq_u, u)) {
              continue;
            }
          } else {
            if (!filter(u)) {
              continue;
            }
          }

          graph.adjacent_nodes(u, [&](const NodeID v) {
            if (!graph.is_ghost_node(v)) {
              return;
            }

            const PEID pe = graph.ghost_owner(v);

            if (created_message_for_pe.get(pe)) {
              return;
            }
            created_message_for_pe.set(pe);

            ++num_messages[thread][pe];
          });

          created_message_for_pe.reset();
        }
      }
  );

  // Offset messages for each thread
  internal::inclusive_col_prefix_sum(num_messages);

  // Allocate send buffers
  std::vector<Buffer> send_buffers(size);
  tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
    send_buffers[pe].resize(num_messages.back()[pe]);
  });

  parallel::deterministic_for<NodeID>(
      from,
      to,
      [&](const NodeID range_from, const NodeID range_to, const PEID thread) {
        Marker<> created_message_for_pe(static_cast<std::size_t>(size));

        for (NodeID seq_u = range_from; seq_u < range_to; ++seq_u) {
          const NodeID u = mapper(seq_u);

          if constexpr (filter_invocable_with_unmapped_node) {
            if (!filter(seq_u, u)) {
              continue;
            }
          } else {
            if (!filter(u)) {
              continue;
            }
          }

          graph.adjacent_nodes(u, [&](const NodeID v) {
            if (!graph.is_ghost_node(v)) {
              return;
            }

            const PEID pe = graph.ghost_owner(v);

            if (created_message_for_pe.get(pe)) {
              return;
            }
            created_message_for_pe.set(pe);

            const auto slot = --num_messages[thread][pe];

            if constexpr (builder_invocable_with_pe) {
              send_buffers[pe][slot] = builder(u, pe);
            } else if constexpr (builder_invocable_with_pe_and_unmapped_node) {
              send_buffers[pe][slot] = builder(seq_u, u, pe);
            } else {
              send_buffers[pe][slot] = builder(u);
            }
          });

          created_message_for_pe.reset();
        }
      }
  );

  sparse_alltoall<Message, Buffer>(
      std::move(send_buffers), std::forward<Receiver>(receiver), graph.communicator()
  );
} // namespace dkaminpar::mpi::graph

template <
    typename Message,
    typename Buffer = NoinitVector<Message>,
    typename Graph,
    typename Filter,
    typename Builder,
    typename Receiver>
void sparse_alltoall_interface_to_pe(
    const Graph &graph,
    const NodeID from,
    const NodeID to,
    Filter &&filter,
    Builder &&builder,
    Receiver &&receiver
) {
  sparse_alltoall_interface_to_pe_custom_range<Message, Buffer>(
      graph,
      from,
      to,
      [](const NodeID u) { return u; },
      std::forward<Filter>(filter),
      std::forward<Builder>(builder),
      std::forward<Receiver>(receiver)
  );
}

template <
    typename Message,
    typename Buffer = NoinitVector<Message>,
    typename Graph,
    typename Filter,
    typename Builder,
    typename Receiver>
void sparse_alltoall_interface_to_pe(
    const Graph &graph, Filter &&filter, Builder &&builder, Receiver &&receiver
) {
  sparse_alltoall_interface_to_pe<Message, Buffer>(
      graph,
      0,
      graph.n(),
      std::forward<Filter>(filter),
      std::forward<Builder>(builder),
      std::forward<Receiver>(receiver)
  );
}

template <
    typename Message,
    typename Buffer = NoinitVector<Message>,
    typename Graph,
    typename Builder,
    typename Receiver>
void sparse_alltoall_interface_to_pe(const Graph &graph, Builder &&builder, Receiver &&receiver) {
  sparse_alltoall_interface_to_pe<Message, Buffer>(
      graph,
      SPARSE_ALLTOALL_NOFILTER,
      std::forward<Builder>(builder),
      std::forward<Receiver>(receiver)
  );
}

template <
    typename Message,
    typename Buffer = NoinitVector<Message>,
    typename Graph,
    typename Filter,
    typename Builder>
std::vector<Buffer> sparse_alltoall_interface_to_pe_get(
    const Graph &graph, const NodeID from, const NodeID to, Filter &&filter, Builder &&builder
) {
  std::vector<Buffer> recv_buffers(mpi::get_comm_size(graph.communicator()));
  sparse_alltoall_interface_to_pe<Message, Buffer>(
      graph,
      from,
      to,
      std::forward<Filter>(filter),
      std::forward<Builder>(builder),
      [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); }
  );
  return recv_buffers;
}

template <
    typename Message,
    typename Buffer = NoinitVector<Message>,
    typename Graph,
    typename Mapper,
    typename Filter,
    typename Builder>
std::vector<Buffer> sparse_alltoall_interface_to_pe_custom_range_get(
    const Graph &graph,
    const NodeID from,
    const NodeID to,
    Mapper &&mapper,
    Filter &&filter,
    Builder &&builder
) {
  std::vector<Buffer> recv_buffers(mpi::get_comm_size(graph.communicator()));
  sparse_alltoall_interface_to_pe_custom_range<Message, Buffer>(
      graph,
      from,
      to,
      std::forward<Mapper>(mapper),
      std::forward<Filter>(filter),
      std::forward<Builder>(builder),
      [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); }
  );
  return recv_buffers;
}

template <
    typename Message,
    typename Buffer = NoinitVector<Message>,
    typename Graph,
    typename Filter,
    typename Builder>
std::vector<Buffer>
sparse_alltoall_interface_to_pe_get(const Graph &graph, Filter &&filter, Builder &&builder) {
  std::vector<Buffer> recv_buffers(mpi::get_comm_size(graph.communicator()));
  sparse_alltoall_interface_to_pe<Message, Buffer>(
      graph,
      0,
      graph.n(),
      std::forward<Filter>(filter),
      std::forward<Builder>(builder),
      [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); }
  );
  return recv_buffers;
}

template <
    typename Message,
    typename Buffer = NoinitVector<Message>,
    typename Graph,
    typename Builder>
std::vector<Buffer> sparse_alltoall_interface_to_pe_get(const Graph &graph, Builder &&builder) {
  std::vector<Buffer> recv_buffers(mpi::get_comm_size(graph.communicator()));
  sparse_alltoall_interface_to_pe<Message, Buffer>(
      graph,
      0,
      graph.n(),
      SPARSE_ALLTOALL_NOFILTER,
      std::forward<Builder>(builder),
      [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); }
  );
  return recv_buffers;
}

template <
    typename Message,
    typename Buffer = NoinitVector<Message>,
    typename Graph,
    typename Filter,
    typename PEGetter,
    typename Builder,
    typename Receiver>
void sparse_alltoall_custom(
    const Graph &graph,
    const NodeID from,
    const NodeID to,
    Filter &&filter,
    PEGetter &&pe_getter,
    Builder &&builder,
    Receiver &&receiver
) {
  static_assert(std::is_invocable_r_v<bool, Filter, NodeID>, "bad filter type");
  static_assert(std::is_invocable_r_v<Message, Builder, NodeID>, "bad builder type");
  static_assert(std::is_invocable_r_v<PEID, PEGetter, NodeID>, "bad pe getter type");

  PEID size, rank;
  std::tie(size, rank) = mpi::get_comm_info(graph.communicator());

  // START_TIMER("Message construction");

  // Allocate message counters
  const PEID num_threads = tbb::this_task_arena::max_concurrency();
  std::vector<CacheAlignedVector<std::size_t>> num_messages(
      num_threads, CacheAlignedVector<std::size_t>(size)
  );

  // Count messages to each PE for each thread
  parallel::deterministic_for<NodeID>(
      from,
      to,
      [&](const NodeID range_from, const NodeID range_to, const PEID thread) {
        for (NodeID u = range_from; u < range_to; ++u) {
          if (filter(u)) {
            ++num_messages[thread][pe_getter(u)];
          }
        }
      }
  );

  // Offset messages for each thread
  internal::inclusive_col_prefix_sum(num_messages);

  // Allocate send buffers
  std::vector<Buffer> send_buffers(size);
  tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
    send_buffers[pe].resize(num_messages.back()[pe]);
  });

  // Fill buffers
  parallel::deterministic_for<NodeID>(
      from,
      to,
      [&](const NodeID range_from, const NodeID range_to, const PEID thread) {
        for (NodeID u = range_from; u < range_to; ++u) {
          if (filter(u)) {
            const PEID pe = pe_getter(u);
            const auto slot = --num_messages[thread][pe];
            send_buffers[pe][slot] = builder(u);
          }
        }
      }
  );

  sparse_alltoall<Message, Buffer>(
      std::move(send_buffers), std::forward<Receiver>(receiver), graph.communicator()
  );
}

template <
    typename Message,
    typename Buffer = NoinitVector<Message>,
    typename Graph,
    typename Filter,
    typename PEGetter,
    typename Builder>
std::vector<Buffer> sparse_alltoall_custom(
    const Graph &graph,
    const NodeID from,
    const NodeID to,
    Filter &&filter,
    PEGetter &&pe_getter,
    Builder &&builder
) {
  auto size = mpi::get_comm_size(graph.communicator());
  std::vector<Buffer> recv_buffers(size);
  sparse_alltoall_custom<Message, Buffer>(
      graph,
      from,
      to,
      std::forward<Filter>(filter),
      std::forward<PEGetter>(pe_getter),
      std::forward<Builder>(builder),
      [&](auto recv_buffer, const PEID pe) { recv_buffers[pe] = std::move(recv_buffer); }
  );
  return recv_buffers;
}

} // namespace kaminpar::mpi::graph
