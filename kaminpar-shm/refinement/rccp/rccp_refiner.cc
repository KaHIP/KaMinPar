/*******************************************************************************
 * Repair-certified cut-packet refiner.
 *
 * @file:   rccp_refiner.cc
 ******************************************************************************/
#include "kaminpar-shm/refinement/rccp/rccp_refiner.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/fm/fm_refiner.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
namespace {

constexpr double kEpsilon = 1e-9;

class Dinic {
  struct Edge {
    int to;
    int rev;
    double cap;
  };

public:
  explicit Dinic(const int n) : _adj(n), _level(n), _next(n) {}

  void add_directed_edge(const int from, const int to, const double capacity) {
    if (capacity <= kEpsilon) {
      return;
    }

    Edge forward{to, static_cast<int>(_adj[to].size()), capacity};
    Edge backward{from, static_cast<int>(_adj[from].size()), 0.0};
    _adj[from].push_back(forward);
    _adj[to].push_back(backward);
  }

  void add_undirected_edge(const int u, const int v, const double capacity) {
    add_directed_edge(u, v, capacity);
    add_directed_edge(v, u, capacity);
  }

  double max_flow(const int source, const int sink) {
    double flow = 0.0;
    while (build_levels(source, sink)) {
      std::fill(_next.begin(), _next.end(), 0);
      while (const double pushed =
                 push_flow(source, sink, std::numeric_limits<double>::infinity())) {
        flow += pushed;
      }
    }
    return flow;
  }

  [[nodiscard]] std::vector<std::uint8_t> source_reachable(const int source) const {
    std::vector<std::uint8_t> reachable(_adj.size(), 0);
    std::queue<int> queue;
    reachable[source] = 1;
    queue.push(source);

    while (!queue.empty()) {
      const int u = queue.front();
      queue.pop();

      for (const Edge &edge : _adj[u]) {
        if (edge.cap > kEpsilon && !reachable[edge.to]) {
          reachable[edge.to] = 1;
          queue.push(edge.to);
        }
      }
    }

    return reachable;
  }

private:
  [[nodiscard]] bool build_levels(const int source, const int sink) {
    std::fill(_level.begin(), _level.end(), -1);
    std::queue<int> queue;
    _level[source] = 0;
    queue.push(source);

    while (!queue.empty()) {
      const int u = queue.front();
      queue.pop();

      for (const Edge &edge : _adj[u]) {
        if (edge.cap > kEpsilon && _level[edge.to] < 0) {
          _level[edge.to] = _level[u] + 1;
          queue.push(edge.to);
        }
      }
    }

    return _level[sink] >= 0;
  }

  double push_flow(const int u, const int sink, const double flow) {
    if (u == sink) {
      return flow;
    }

    for (int &i = _next[u]; i < static_cast<int>(_adj[u].size()); ++i) {
      Edge &edge = _adj[u][i];
      if (edge.cap <= kEpsilon || _level[edge.to] != _level[u] + 1) {
        continue;
      }

      const double pushed = push_flow(edge.to, sink, std::min(flow, edge.cap));
      if (pushed > kEpsilon) {
        edge.cap -= pushed;
        _adj[edge.to][edge.rev].cap += pushed;
        return pushed;
      }
    }

    return 0.0;
  }

  std::vector<std::vector<Edge>> _adj;
  std::vector<int> _level;
  std::vector<int> _next;
};

enum class PacketType : std::uint8_t {
  SINGLETON,
  MIN_CUT,
};

enum class PacketGenerationMode : std::uint8_t {
  BASELINE,
  PRIMARY,
  REPAIR,
};

enum class BudgetMode : std::uint8_t {
  BASELINE,
  PRIMARY,
};

struct Packet {
  std::size_t id;
  BlockID source;
  BlockID target;
  std::vector<NodeID> vertices;
  BlockWeight weight;
  EdgeWeight gain;
  PacketType type;
};

struct ActivePair {
  BlockID source;
  BlockID target;
  EdgeWeight cut_weight = 0;
  std::vector<NodeID> seeds;
};

struct MasterState {
  std::vector<std::size_t> selected_packets;
  std::vector<NodeID> moved_vertices;
  std::vector<BlockWeight> block_weights;
  EdgeWeight additive_gain = 0;
  BlockWeight moved_weight = 0;
  double score = 0.0;
  std::size_t next_packet = 0;
};

struct MasterSearchConfig {
  std::size_t max_solutions;
  std::size_t depth;
  std::size_t beam_width;
  std::size_t branching_factor;
  BlockWeight max_moved_weight;
  BudgetMode budget_mode;
  bool require_final_feasibility;
  bool require_positive_additive_gain;
};

[[nodiscard]] bool
intersects_sorted(const std::vector<NodeID> &lhs, const std::vector<NodeID> &rhs) {
  auto lhs_it = lhs.begin();
  auto rhs_it = rhs.begin();
  while (lhs_it != lhs.end() && rhs_it != rhs.end()) {
    if (*lhs_it == *rhs_it) {
      return true;
    }
    if (*lhs_it < *rhs_it) {
      ++lhs_it;
    } else {
      ++rhs_it;
    }
  }

  return false;
}

[[nodiscard]] std::vector<NodeID>
merge_sorted_vertices(const std::vector<NodeID> &lhs, const std::vector<NodeID> &rhs) {
  std::vector<NodeID> result;
  result.reserve(lhs.size() + rhs.size());
  std::merge(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::back_inserter(result));
  return result;
}

[[nodiscard]] double packet_priority(const Packet &packet) {
  const double density = static_cast<double>(packet.gain) / std::max<BlockWeight>(1, packet.weight);
  if (packet.gain > 0) {
    return 1e12 + 1e6 * density + packet.gain;
  }

  return density;
}

[[nodiscard]] bool better_state(const MasterState &lhs, const MasterState &rhs) {
  if (lhs.score != rhs.score) {
    return lhs.score > rhs.score;
  }
  if (lhs.additive_gain != rhs.additive_gain) {
    return lhs.additive_gain > rhs.additive_gain;
  }
  return lhs.next_packet < rhs.next_packet;
}

} // namespace

template <typename Graph> class RccpRefinerImpl {
public:
  explicit RccpRefinerImpl(const Context &ctx) : _ctx(ctx), _r_ctx(ctx.refinement.rccp) {}

  void initialize(const PartitionedGraph &p_graph) {
    _local_index.assign(p_graph.n(), -1);
  }

  bool refine(PartitionedGraph &p_graph, const Graph &graph, const PartitionContext &p_ctx) {
    SCOPED_TIMER("RCCP Refiner");

    if (_r_ctx.enable_quality_mode) {
      return refine_quality_mode(p_graph, graph, p_ctx);
    }

    return refine_baseline(p_graph, graph, p_ctx);
  }

private:
  bool
  refine_baseline(PartitionedGraph &p_graph, const Graph &graph, const PartitionContext &p_ctx) {
    bool improved = false;
    const int max_iterations =
        _r_ctx.num_iterations == 0 ? std::numeric_limits<int>::max() : _r_ctx.num_iterations;

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
      std::vector<Packet> packets = generate_packets(p_graph, graph, p_ctx);
      if (packets.empty()) {
        break;
      }

      const std::vector<std::size_t> selected_packets =
          solve_master(p_graph, graph, p_ctx, packets);
      if (selected_packets.empty()) {
        break;
      }

      const EdgeWeight selected_gain = exact_gain(p_graph, graph, packets, selected_packets);
      if (selected_gain <= 0) {
        break;
      }

      apply_packets(p_graph, packets, selected_packets);
      improved = true;
    }

    return improved;
  }

  bool refine_quality_mode(
      PartitionedGraph &p_graph, const Graph &graph, const PartitionContext &p_ctx
  ) {
    bool improved = false;
    const int max_iterations =
        _r_ctx.num_iterations == 0 ? std::numeric_limits<int>::max() : _r_ctx.num_iterations;

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
      bool improved_this_iteration = false;

      if (_r_ctx.run_baseline_first) {
        const bool baseline_improved = run_single_baseline_iteration(p_graph, graph, p_ctx);
        improved |= baseline_improved;
        improved_this_iteration |= baseline_improved;
      }

      const EdgeWeight root_cut = metrics::edge_cut(p_graph);
      const std::vector<BlockID> root_partition = snapshot_partition(p_graph, graph);
      std::vector<BlockID> best_partition = root_partition;
      EdgeWeight best_cut = root_cut;

      std::vector<Packet> primary_packets =
          generate_packets(p_graph, graph, p_ctx, PacketGenerationMode::PRIMARY);
      if (!primary_packets.empty()) {
        const MasterSearchConfig primary_config{
            .max_solutions = std::min(_r_ctx.primary_scenarios, _r_ctx.candidate_prefilter),
            .depth = _r_ctx.master_depth,
            .beam_width = _r_ctx.master_beam_width,
            .branching_factor = _r_ctx.master_branching_factor,
            .max_moved_weight = moved_weight_limit(p_ctx, _r_ctx.max_primary_moved_weight_fraction),
            .budget_mode = BudgetMode::PRIMARY,
            .require_final_feasibility = false,
            .require_positive_additive_gain = false,
        };
        std::vector<MasterState> primary_scenarios =
            search_packet_sets(p_graph, graph, p_ctx, primary_packets, primary_config);

        for (const MasterState &primary : primary_scenarios) {
          restore_partition(p_graph, graph, root_partition);
          apply_packets(p_graph, primary_packets, primary.selected_packets);

          if (!is_within_primary_budget(current_block_weights(p_graph), p_ctx)) {
            continue;
          }

          std::vector<std::uint8_t> primary_vertices(p_graph.n(), 0);
          for (const NodeID u : primary.moved_vertices) {
            primary_vertices[u] = 1;
          }

          std::vector<Packet> repair_packets = generate_packets(
              p_graph, graph, p_ctx, PacketGenerationMode::REPAIR, primary_vertices
          );
          if (repair_packets.empty()) {
            if (metrics::is_feasible(p_graph, p_ctx)) {
              if (_r_ctx.enable_fm_closure && _r_ctx.fm_closure_iterations > 0) {
                run_fm_closure(p_graph, p_ctx);
              }

              if (metrics::is_feasible(p_graph, p_ctx)) {
                const EdgeWeight candidate_cut = metrics::edge_cut(p_graph);
                if (candidate_cut < best_cut) {
                  best_cut = candidate_cut;
                  best_partition = snapshot_partition(p_graph, graph);
                }
              }
            }
            continue;
          }

          const MasterSearchConfig repair_config{
              .max_solutions = _r_ctx.repair_scenarios,
              .depth = _r_ctx.master_depth,
              .beam_width = _r_ctx.master_beam_width,
              .branching_factor = _r_ctx.master_branching_factor,
              .max_moved_weight =
                  moved_weight_limit(p_ctx, _r_ctx.max_repair_moved_weight_fraction),
              .budget_mode = BudgetMode::PRIMARY,
              .require_final_feasibility = true,
              .require_positive_additive_gain = false,
          };
          std::vector<MasterState> repair_scenarios =
              search_packet_sets(p_graph, graph, p_ctx, repair_packets, repair_config);

          for (const MasterState &repair : repair_scenarios) {
            restore_partition(p_graph, graph, root_partition);
            apply_packets(p_graph, primary_packets, primary.selected_packets);
            apply_packets(p_graph, repair_packets, repair.selected_packets);

            if (!metrics::is_feasible(p_graph, p_ctx)) {
              continue;
            }

            if (_r_ctx.enable_fm_closure && _r_ctx.fm_closure_iterations > 0) {
              run_fm_closure(p_graph, p_ctx);
            }

            if (!metrics::is_feasible(p_graph, p_ctx)) {
              continue;
            }

            const EdgeWeight candidate_cut = metrics::edge_cut(p_graph);
            if (candidate_cut < best_cut) {
              best_cut = candidate_cut;
              best_partition = snapshot_partition(p_graph, graph);
            }
          }
        }
      }

      if (best_cut < root_cut) {
        restore_partition(p_graph, graph, best_partition);
        improved = true;
        improved_this_iteration = true;
      } else {
        restore_partition(p_graph, graph, root_partition);
      }

      if (!improved_this_iteration) {
        break;
      }
    }

    return improved;
  }

  bool run_single_baseline_iteration(
      PartitionedGraph &p_graph, const Graph &graph, const PartitionContext &p_ctx
  ) {
    std::vector<Packet> packets = generate_packets(p_graph, graph, p_ctx);
    if (packets.empty()) {
      return false;
    }

    const std::vector<std::size_t> selected_packets = solve_master(p_graph, graph, p_ctx, packets);
    if (selected_packets.empty()) {
      return false;
    }

    if (exact_gain(p_graph, graph, packets, selected_packets) <= 0) {
      return false;
    }

    apply_packets(p_graph, packets, selected_packets);
    return true;
  }

  [[nodiscard]] std::vector<Packet> generate_packets(
      const PartitionedGraph &p_graph,
      const Graph &graph,
      const PartitionContext &p_ctx,
      const PacketGenerationMode mode = PacketGenerationMode::BASELINE,
      const std::vector<std::uint8_t> &forbidden_vertices = {}
  ) {
    std::vector<Packet> packets;
    packets.reserve(_r_ctx.max_total_packets);

    const std::vector<std::uint8_t> repair_sources =
        mode == PacketGenerationMode::REPAIR ? find_repair_source_blocks(p_graph, graph, p_ctx)
                                             : std::vector<std::uint8_t>{};

    if (_r_ctx.enable_singleton_packets) {
      generate_singleton_packets(
          p_graph, graph, p_ctx, mode, forbidden_vertices, repair_sources, packets
      );
    }

    if (_r_ctx.enable_mincut_packets) {
      generate_mincut_packets(
          p_graph, graph, p_ctx, mode, forbidden_vertices, repair_sources, packets
      );
    }

    deduplicate_packets(packets);
    filter_packets(packets);
    assign_packet_ids(packets);

    return packets;
  }

  void generate_singleton_packets(
      const PartitionedGraph &p_graph,
      const Graph &graph,
      const PartitionContext &p_ctx,
      const PacketGenerationMode mode,
      const std::vector<std::uint8_t> &forbidden_vertices,
      const std::vector<std::uint8_t> &repair_sources,
      std::vector<Packet> &packets
  ) const {
    const BlockWeight max_weight = max_packet_weight(p_ctx);

    for (const NodeID u : graph.nodes()) {
      if (!forbidden_vertices.empty() && forbidden_vertices[u]) {
        continue;
      }

      const BlockID source = p_graph.block(u);
      const NodeWeight u_weight = p_graph.node_weight(u);
      if (u_weight > max_weight) {
        continue;
      }

      if (mode == PacketGenerationMode::REPAIR && !repair_sources[source]) {
        continue;
      }

      std::vector<std::pair<BlockID, EdgeWeight>> connectivity;
      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
        const BlockID block = p_graph.block(v);
        auto it = std::find_if(connectivity.begin(), connectivity.end(), [&](const auto &entry) {
          return entry.first == block;
        });
        if (it == connectivity.end()) {
          connectivity.emplace_back(block, weight);
        } else {
          it->second += weight;
        }
      });

      EdgeWeight source_conn = 0;
      for (const auto &[block, conn] : connectivity) {
        if (block == source) {
          source_conn = conn;
          break;
        }
      }

      for (const auto &[target, target_conn] : connectivity) {
        if (target == source) {
          continue;
        }
        if (mode == PacketGenerationMode::REPAIR &&
            p_graph.block_weight(target) >=
                p_ctx.max_block_weight(target) + trust_region(p_ctx, target)) {
          continue;
        }

        const EdgeWeight gain = target_conn - source_conn;
        if (!accept_packet_gain(gain, PacketType::SINGLETON, mode)) {
          continue;
        }

        packets.push_back(
            Packet{
                .id = packets.size(),
                .source = source,
                .target = target,
                .vertices = {u},
                .weight = u_weight,
                .gain = gain,
                .type = PacketType::SINGLETON,
            }
        );
      }
    }
  }

  void generate_mincut_packets(
      const PartitionedGraph &p_graph,
      const Graph &graph,
      const PartitionContext &p_ctx,
      const PacketGenerationMode mode,
      const std::vector<std::uint8_t> &forbidden_vertices,
      const std::vector<std::uint8_t> &repair_sources,
      std::vector<Packet> &packets
  ) {
    std::vector<ActivePair> active_pairs =
        find_active_pairs(p_graph, graph, mode, repair_sources);
    std::sort(active_pairs.begin(), active_pairs.end(), [](const auto &lhs, const auto &rhs) {
      return lhs.cut_weight > rhs.cut_weight;
    });
    if (active_pairs.size() > _r_ctx.max_active_pairs) {
      active_pairs.resize(_r_ctx.max_active_pairs);
    }

    const double price_unit = std::max<double>(
        1.0,
        static_cast<double>(graph.total_edge_weight()) /
            std::max<NodeWeight>(1, graph.total_node_weight())
    );
    const std::vector<double> prices = price_grid(price_unit, mode);

    for (ActivePair &pair : active_pairs) {
      std::sort(pair.seeds.begin(), pair.seeds.end());
      pair.seeds.erase(std::unique(pair.seeds.begin(), pair.seeds.end()), pair.seeds.end());

      const std::vector<NodeID> region =
          build_source_region(p_graph, graph, pair, forbidden_vertices);
      if (region.empty()) {
        continue;
      }

      for (std::size_t i = 0; i < region.size(); ++i) {
        _local_index[region[i]] = static_cast<int>(i);
      }

      for (const double price : prices) {
        generate_mincut_packet_for_price(
            p_graph, graph, p_ctx, pair, region, price, {}, mode, packets
        );
      }

      if (mode == PacketGenerationMode::PRIMARY && _r_ctx.enable_seeded_primary_packets) {
        std::vector<NodeID> seeds = select_forced_primary_seeds(p_graph, graph, pair);
        for (const NodeID seed : seeds) {
          if (!forbidden_vertices.empty() && forbidden_vertices[seed]) {
            continue;
          }

          const std::vector<NodeID> forced_seed = {seed};
          generate_mincut_packet_for_price(
              p_graph, graph, p_ctx, pair, region, -2.0 * price_unit, forced_seed, mode, packets
          );
          generate_mincut_packet_for_price(
              p_graph, graph, p_ctx, pair, region, 0.0, forced_seed, mode, packets
          );

          if (const NodeID mate = strongest_same_block_neighbor(p_graph, graph, seed);
              mate != kInvalidNodeID) {
            const std::vector<NodeID> forced_pair = {seed, mate};
            generate_mincut_packet_for_price(
                p_graph, graph, p_ctx, pair, region, 0.0, forced_pair, mode, packets
            );
          }
        }
      }

      for (const NodeID u : region) {
        _local_index[u] = -1;
      }
    }
  }

  [[nodiscard]] std::vector<ActivePair> find_active_pairs(
      const PartitionedGraph &p_graph,
      const Graph &graph,
      const PacketGenerationMode mode,
      const std::vector<std::uint8_t> &repair_sources
  ) const {
    std::unordered_map<std::uint64_t, ActivePair> active_pair_map;

    for (const NodeID u : graph.nodes()) {
      const BlockID source = p_graph.block(u);
      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
        const BlockID target = p_graph.block(v);
        if (source == target) {
          return;
        }
        if (mode == PacketGenerationMode::REPAIR && !repair_sources[source]) {
          return;
        }

        const std::uint64_t key =
            static_cast<std::uint64_t>(source) * p_graph.k() + static_cast<std::uint64_t>(target);
        auto [it, inserted] = active_pair_map.try_emplace(key);
        if (inserted) {
          it->second.source = source;
          it->second.target = target;
        }
        it->second.cut_weight += weight;
        it->second.seeds.push_back(u);
      });
    }

    std::vector<ActivePair> active_pairs;
    active_pairs.reserve(active_pair_map.size());
    for (auto &[_, pair] : active_pair_map) {
      active_pairs.push_back(std::move(pair));
    }

    return active_pairs;
  }

  [[nodiscard]] std::vector<std::uint8_t> find_repair_source_blocks(
      const PartitionedGraph &p_graph, const Graph &graph, const PartitionContext &p_ctx
  ) const {
    std::vector<std::vector<BlockID>> outgoing(p_graph.k());
    std::vector<std::vector<BlockID>> incoming(p_graph.k());

    for (const NodeID u : graph.nodes()) {
      const BlockID source = p_graph.block(u);
      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight) {
        const BlockID target = p_graph.block(v);
        if (source == target) {
          return;
        }

        outgoing[source].push_back(target);
        incoming[target].push_back(source);
      });
    }

    for (BlockID block = 0; block < p_graph.k(); ++block) {
      std::sort(outgoing[block].begin(), outgoing[block].end());
      outgoing[block].erase(
          std::unique(outgoing[block].begin(), outgoing[block].end()), outgoing[block].end()
      );
      std::sort(incoming[block].begin(), incoming[block].end());
      incoming[block].erase(
          std::unique(incoming[block].begin(), incoming[block].end()), incoming[block].end()
      );
    }

    std::vector<std::uint8_t> from_overloaded(p_graph.k(), 0);
    std::queue<BlockID> queue;
    for (const BlockID block : p_graph.blocks()) {
      if (p_graph.block_weight(block) > p_ctx.max_block_weight(block)) {
        from_overloaded[block] = 1;
        queue.push(block);
      }
    }

    while (!queue.empty()) {
      const BlockID block = queue.front();
      queue.pop();

      for (const BlockID target : outgoing[block]) {
        if (!from_overloaded[target]) {
          from_overloaded[target] = 1;
          queue.push(target);
        }
      }
    }

    std::vector<std::uint8_t> reaches_slack(p_graph.k(), 0);
    for (const BlockID block : p_graph.blocks()) {
      if (p_graph.block_weight(block) < p_ctx.max_block_weight(block)) {
        reaches_slack[block] = 1;
        queue.push(block);
      }
    }

    while (!queue.empty()) {
      const BlockID block = queue.front();
      queue.pop();

      for (const BlockID source : incoming[block]) {
        if (!reaches_slack[source]) {
          reaches_slack[source] = 1;
          queue.push(source);
        }
      }
    }

    std::vector<std::uint8_t> repair_sources(p_graph.k(), 0);
    for (const BlockID block : p_graph.blocks()) {
      repair_sources[block] = from_overloaded[block] && reaches_slack[block];
    }

    return repair_sources;
  }

  [[nodiscard]] std::vector<NodeID> build_source_region(
      const PartitionedGraph &p_graph,
      const Graph &graph,
      const ActivePair &pair,
      const std::vector<std::uint8_t> &forbidden_vertices
  ) {
    std::vector<NodeID> region;
    std::queue<NodeID> queue;
    region.reserve(std::min<NodeID>(_r_ctx.max_region_vertices, p_graph.n()));

    const auto add = [&](const NodeID u, const int distance) {
      if ((!forbidden_vertices.empty() && forbidden_vertices[u]) || _local_index[u] >= 0 ||
          p_graph.block(u) != pair.source || region.size() >= _r_ctx.max_region_vertices) {
        return;
      }

      _local_index[u] = distance;
      region.push_back(u);
      queue.push(u);
    };

    for (const NodeID seed : pair.seeds) {
      add(seed, 0);
    }

    while (!queue.empty() && region.size() < _r_ctx.max_region_vertices) {
      const NodeID u = queue.front();
      queue.pop();

      const int distance = _local_index[u];
      if (distance >= static_cast<int>(_r_ctx.active_region_radius)) {
        continue;
      }

      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight) { add(v, distance + 1); });
    }

    for (const NodeID u : region) {
      _local_index[u] = -1;
    }
    std::sort(region.begin(), region.end());

    return region;
  }

  [[nodiscard]] std::vector<double>
  price_grid(const double price_unit, const PacketGenerationMode mode) const {
    switch (mode) {
    case PacketGenerationMode::PRIMARY:
      return {-8.0 * price_unit, -4.0 * price_unit, -2.0 * price_unit, 0.0, 2.0 * price_unit};

    case PacketGenerationMode::REPAIR:
      return {-price_unit, 0.0, price_unit, 2.0 * price_unit, 4.0 * price_unit};

    case PacketGenerationMode::BASELINE:
      return {-price_unit, 0.0, price_unit, 2.0 * price_unit, 4.0 * price_unit};
    }

    __builtin_unreachable();
  }

  [[nodiscard]] std::vector<NodeID> select_forced_primary_seeds(
      const PartitionedGraph &p_graph, const Graph &graph, const ActivePair &pair
  ) const {
    std::vector<std::pair<EdgeWeight, NodeID>> rated_seeds;
    rated_seeds.reserve(pair.seeds.size());

    for (const NodeID seed : pair.seeds) {
      EdgeWeight target_conn = 0;
      EdgeWeight source_conn = 0;
      graph.adjacent_nodes(seed, [&](const NodeID v, const EdgeWeight weight) {
        const BlockID block = p_graph.block(v);
        if (block == pair.target) {
          target_conn += weight;
        } else if (block == pair.source) {
          source_conn += weight;
        }
      });
      rated_seeds.emplace_back(target_conn - source_conn, seed);
    }

    std::sort(rated_seeds.begin(), rated_seeds.end(), [](const auto &lhs, const auto &rhs) {
      return lhs > rhs;
    });

    std::vector<NodeID> seeds;
    const std::size_t max_seeds =
        std::min(_r_ctx.frustrated_seed_count_per_pair, rated_seeds.size());
    seeds.reserve(max_seeds);
    for (std::size_t i = 0; i < max_seeds; ++i) {
      seeds.push_back(rated_seeds[i].second);
    }

    return seeds;
  }

  [[nodiscard]] NodeID strongest_same_block_neighbor(
      const PartitionedGraph &p_graph, const Graph &graph, const NodeID u
  ) const {
    const BlockID block = p_graph.block(u);
    NodeID best_neighbor = kInvalidNodeID;
    EdgeWeight best_weight = std::numeric_limits<EdgeWeight>::min();

    graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
      if (p_graph.block(v) == block && weight > best_weight) {
        best_weight = weight;
        best_neighbor = v;
      }
    });

    return best_neighbor;
  }

  void generate_mincut_packet_for_price(
      const PartitionedGraph &p_graph,
      const Graph &graph,
      const PartitionContext &p_ctx,
      const ActivePair &pair,
      const std::vector<NodeID> &region,
      const double price,
      const std::vector<NodeID> &forced_vertices,
      const PacketGenerationMode mode,
      std::vector<Packet> &packets
  ) const {
    const int source = static_cast<int>(region.size());
    const int sink = source + 1;
    Dinic cut(region.size() + 2);

    for (std::size_t i = 0; i < region.size(); ++i) {
      const NodeID u = region[i];

      double target_conn = 0.0;
      double source_outside_conn = 0.0;
      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
        const BlockID block = p_graph.block(v);
        if (block == pair.target) {
          target_conn += weight;
        } else if (block == pair.source && _local_index[v] < 0) {
          source_outside_conn += weight;
        }
      });

      const double unary = target_conn - source_outside_conn - price * p_graph.node_weight(u);
      if (unary > kEpsilon) {
        cut.add_directed_edge(source, static_cast<int>(i), unary);
      } else if (unary < -kEpsilon) {
        cut.add_directed_edge(static_cast<int>(i), sink, -unary);
      }
    }

    const double infinite_capacity =
        std::max<double>(1.0, graph.total_edge_weight() + graph.total_node_weight()) * 1024.0;
    for (const NodeID forced_vertex : forced_vertices) {
      const int local = _local_index[forced_vertex];
      if (local >= 0) {
        cut.add_directed_edge(source, local, infinite_capacity);
      }
    }

    for (std::size_t i = 0; i < region.size(); ++i) {
      const NodeID u = region[i];
      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
        const int j = _local_index[v];
        if (j >= 0 && u < v) {
          cut.add_undirected_edge(static_cast<int>(i), j, weight);
        }
      });
    }

    cut.max_flow(source, sink);
    const std::vector<std::uint8_t> selected = cut.source_reachable(source);

    std::vector<std::uint8_t> in_packet(region.size(), 0);
    std::size_t selected_count = 0;
    for (std::size_t i = 0; i < region.size(); ++i) {
      if (selected[i]) {
        in_packet[i] = 1;
        ++selected_count;
      }
    }
    if (selected_count == 0) {
      return;
    }

    std::vector<std::vector<NodeID>> components;
    split_selected_region_into_components(p_graph, graph, region, in_packet, components);

    for (std::vector<NodeID> &component : components) {
      insert_packet(
          p_graph,
          graph,
          p_ctx,
          pair.source,
          pair.target,
          std::move(component),
          PacketType::MIN_CUT,
          mode,
          packets
      );
    }

    if (components.size() > 1) {
      std::vector<NodeID> full_packet;
      full_packet.reserve(selected_count);
      for (std::size_t i = 0; i < region.size(); ++i) {
        if (in_packet[i]) {
          full_packet.push_back(region[i]);
        }
      }
      insert_packet(
          p_graph,
          graph,
          p_ctx,
          pair.source,
          pair.target,
          std::move(full_packet),
          PacketType::MIN_CUT,
          mode,
          packets
      );
    }
  }

  void split_selected_region_into_components(
      const PartitionedGraph &p_graph,
      const Graph &graph,
      const std::vector<NodeID> &region,
      const std::vector<std::uint8_t> &in_packet,
      std::vector<std::vector<NodeID>> &components
  ) const {
    std::vector<std::uint8_t> visited(region.size(), 0);
    std::queue<int> queue;

    for (std::size_t i = 0; i < region.size(); ++i) {
      if (!in_packet[i] || visited[i]) {
        continue;
      }

      std::vector<NodeID> component;
      visited[i] = 1;
      queue.push(static_cast<int>(i));

      while (!queue.empty()) {
        const int local_u = queue.front();
        queue.pop();

        const NodeID u = region[local_u];
        component.push_back(u);

        graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight) {
          const int local_v = _local_index[v];
          if (local_v >= 0 && in_packet[local_v] && !visited[local_v] &&
              p_graph.block(v) == p_graph.block(u)) {
            visited[local_v] = 1;
            queue.push(local_v);
          }
        });
      }

      std::sort(component.begin(), component.end());
      components.push_back(std::move(component));
    }
  }

  void insert_packet(
      const PartitionedGraph &p_graph,
      const Graph &graph,
      const PartitionContext &p_ctx,
      const BlockID source,
      const BlockID target,
      std::vector<NodeID> vertices,
      const PacketType type,
      const PacketGenerationMode mode,
      std::vector<Packet> &packets
  ) const {
    std::sort(vertices.begin(), vertices.end());
    vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());
    if (vertices.empty()) {
      return;
    }

    BlockWeight weight = 0;
    for (const NodeID u : vertices) {
      weight += p_graph.node_weight(u);
    }
    if (weight > max_packet_weight(p_ctx)) {
      return;
    }

    const EdgeWeight gain = exact_packet_gain(p_graph, graph, source, target, vertices);
    if (!accept_packet_gain(gain, type, mode)) {
      return;
    }

    packets.push_back(
        Packet{
            .id = packets.size(),
            .source = source,
            .target = target,
            .vertices = std::move(vertices),
            .weight = weight,
            .gain = gain,
            .type = type,
        }
    );
  }

  [[nodiscard]] EdgeWeight exact_packet_gain(
      const PartitionedGraph &p_graph,
      const Graph &graph,
      const BlockID source,
      const BlockID target,
      const std::vector<NodeID> &vertices
  ) const {
    EdgeWeight gain = 0;

    for (const NodeID u : vertices) {
      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
        const BlockID block = p_graph.block(v);
        if (block == target) {
          gain += weight;
        } else if (block == source && !std::binary_search(vertices.begin(), vertices.end(), v)) {
          gain -= weight;
        }
      });
    }

    return gain;
  }

  [[nodiscard]] std::vector<std::size_t> solve_master(
      const PartitionedGraph &p_graph,
      const Graph &graph,
      const PartitionContext &p_ctx,
      const std::vector<Packet> &packets
  ) const {
    const MasterSearchConfig config{
        .max_solutions = _r_ctx.master_beam_width,
        .depth = _r_ctx.master_depth,
        .beam_width = _r_ctx.master_beam_width,
        .branching_factor = _r_ctx.master_branching_factor,
        .max_moved_weight = 0,
        .budget_mode = BudgetMode::BASELINE,
        .require_final_feasibility = true,
        .require_positive_additive_gain = true,
    };
    std::vector<MasterState> feasible_states =
        search_packet_sets(p_graph, graph, p_ctx, packets, config);

    EdgeWeight best_exact_gain = 0;
    std::vector<std::size_t> best_selection;
    for (const MasterState &state : feasible_states) {
      const EdgeWeight gain = exact_gain(p_graph, graph, packets, state.selected_packets);
      if (gain > best_exact_gain) {
        best_exact_gain = gain;
        best_selection = state.selected_packets;
      }
    }

    return best_selection;
  }

  [[nodiscard]] std::vector<MasterState> search_packet_sets(
      const PartitionedGraph &p_graph,
      const Graph &graph,
      const PartitionContext &p_ctx,
      const std::vector<Packet> &packets,
      const MasterSearchConfig &config
  ) const {
    if (config.depth == 0 || config.beam_width == 0 || config.branching_factor == 0 ||
        config.max_solutions == 0) {
      return {};
    }

    MasterState initial_state;
    initial_state.block_weights = current_block_weights(p_graph);

    const double balance_penalty_unit = std::max<double>(
        1.0,
        static_cast<double>(graph.total_edge_weight()) /
            std::max<NodeWeight>(1, graph.total_node_weight())
    );

    std::vector<MasterState> states = {std::move(initial_state)};
    std::vector<MasterState> accepted_states;

    for (std::size_t depth = 0; depth < config.depth; ++depth) {
      std::vector<MasterState> next_states;

      for (const MasterState &state : states) {
        std::vector<MasterState> local_extensions;

        for (std::size_t packet_id = state.next_packet; packet_id < packets.size(); ++packet_id) {
          const Packet &packet = packets[packet_id];
          if (intersects_sorted(state.moved_vertices, packet.vertices)) {
            continue;
          }

          MasterState next = state;
          next.next_packet = packet_id + 1;
          next.selected_packets.push_back(packet_id);
          next.moved_vertices = merge_sorted_vertices(state.moved_vertices, packet.vertices);
          next.block_weights[packet.source] -= packet.weight;
          next.block_weights[packet.target] += packet.weight;
          next.moved_weight += packet.weight;

          if (config.max_moved_weight > 0 && next.moved_weight > config.max_moved_weight) {
            continue;
          }
          if (!is_within_budget(next.block_weights, p_ctx, config.budget_mode)) {
            continue;
          }

          next.additive_gain += packet.gain;
          next.score =
              next.additive_gain - balance_penalty_unit * balance_debt(next.block_weights, p_ctx);
          local_extensions.push_back(std::move(next));
        }

        keep_best_states(local_extensions, config.branching_factor);
        std::move(
            local_extensions.begin(), local_extensions.end(), std::back_inserter(next_states)
        );
      }

      if (next_states.empty()) {
        break;
      }

      keep_best_states(next_states, config.beam_width);

      for (const MasterState &state : next_states) {
        if (config.require_positive_additive_gain && state.additive_gain <= 0) {
          continue;
        }
        if (config.require_final_feasibility && !is_feasible(state.block_weights, p_ctx)) {
          continue;
        }

        accepted_states.push_back(state);
      }

      states = std::move(next_states);
    }

    keep_best_states(accepted_states, config.max_solutions);
    return accepted_states;
  }

  [[nodiscard]] EdgeWeight exact_gain(
      const PartitionedGraph &p_graph,
      const Graph &graph,
      const std::vector<Packet> &packets,
      const std::vector<std::size_t> &selected_packets
  ) const {
    std::vector<std::pair<NodeID, BlockID>> moves;
    for (const std::size_t packet_id : selected_packets) {
      const Packet &packet = packets[packet_id];
      for (const NodeID u : packet.vertices) {
        moves.emplace_back(u, packet.target);
      }
    }
    std::sort(moves.begin(), moves.end());

    if (std::adjacent_find(moves.begin(), moves.end(), [](const auto &lhs, const auto &rhs) {
          return lhs.first == rhs.first;
        }) != moves.end()) {
      return std::numeric_limits<EdgeWeight>::min();
    }

    const auto find_move = [&](const NodeID u) {
      return std::lower_bound(moves.begin(), moves.end(), u, [](const auto &entry, const NodeID v) {
        return entry.first < v;
      });
    };

    EdgeWeight gain = 0;
    for (const auto &[u, new_u_block] : moves) {
      const BlockID old_u_block = p_graph.block(u);

      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
        const auto v_move = find_move(v);
        const bool v_is_moved = v_move != moves.end() && v_move->first == v;
        if (v_is_moved && v < u) {
          return;
        }

        const BlockID old_v_block = p_graph.block(v);
        const BlockID new_v_block = v_is_moved ? v_move->second : old_v_block;

        const bool old_cut = old_u_block != old_v_block;
        const bool new_cut = new_u_block != new_v_block;
        if (old_cut && !new_cut) {
          gain += weight;
        } else if (!old_cut && new_cut) {
          gain -= weight;
        }
      });
    }

    return gain;
  }

  void apply_packets(
      PartitionedGraph &p_graph,
      const std::vector<Packet> &packets,
      const std::vector<std::size_t> &selected_packets
  ) const {
    for (const std::size_t packet_id : selected_packets) {
      const Packet &packet = packets[packet_id];
      for (const NodeID u : packet.vertices) {
        p_graph.set_block(u, packet.target);
      }
    }
  }

  [[nodiscard]] bool accept_packet_gain(
      const EdgeWeight gain, const PacketType type, const PacketGenerationMode mode
  ) const {
    switch (mode) {
    case PacketGenerationMode::REPAIR:
      return gain >= -_r_ctx.repair_max_negative_gain;

    case PacketGenerationMode::PRIMARY:
      return gain > 0 || gain >= -_r_ctx.max_negative_gain;

    case PacketGenerationMode::BASELINE:
      return gain > 0 || (type == PacketType::SINGLETON && gain >= -_r_ctx.max_negative_gain);
    }

    __builtin_unreachable();
  }

  [[nodiscard]] BlockWeight max_packet_weight(const PartitionContext &p_ctx) const {
    if (_r_ctx.max_packet_weight_fraction <= 0.0) {
      return std::numeric_limits<BlockWeight>::max();
    }

    const double average_block_weight =
        static_cast<double>(p_ctx.total_node_weight) / std::max<BlockID>(1, p_ctx.k);
    return std::max<BlockWeight>(
        p_ctx.max_node_weight,
        static_cast<BlockWeight>(
            std::ceil(_r_ctx.max_packet_weight_fraction * average_block_weight)
        )
    );
  }

  [[nodiscard]] BlockWeight trust_region(const PartitionContext &p_ctx, const BlockID block) const {
    return std::max<BlockWeight>(
        p_ctx.max_node_weight,
        static_cast<BlockWeight>(
            std::ceil(_r_ctx.trust_region_factor * p_ctx.max_block_weight(block))
        )
    );
  }

  [[nodiscard]] bool is_within_budget(
      const std::vector<BlockWeight> &block_weights,
      const PartitionContext &p_ctx,
      const BudgetMode mode
  ) const {
    switch (mode) {
    case BudgetMode::BASELINE:
      return is_within_trust_region(block_weights, p_ctx);

    case BudgetMode::PRIMARY:
      return is_within_primary_budget(block_weights, p_ctx);
    }

    __builtin_unreachable();
  }

  [[nodiscard]] bool is_within_trust_region(
      const std::vector<BlockWeight> &block_weights, const PartitionContext &p_ctx
  ) const {
    for (BlockID block = 0; block < block_weights.size(); ++block) {
      const BlockWeight trust = trust_region(p_ctx, block);
      if (block_weights[block] > p_ctx.max_block_weight(block) + trust ||
          block_weights[block] < p_ctx.min_block_weight(block) - trust) {
        return false;
      }
    }

    return true;
  }

  [[nodiscard]] bool is_within_primary_budget(
      const std::vector<BlockWeight> &block_weights, const PartitionContext &p_ctx
  ) const {
    for (BlockID block = 0; block < block_weights.size(); ++block) {
      const BlockWeight excess_budget = static_cast<BlockWeight>(
          std::ceil(_r_ctx.primary_temporary_excess_fraction * p_ctx.max_block_weight(block))
      );
      const BlockWeight deficit_budget = static_cast<BlockWeight>(
          std::ceil(_r_ctx.primary_temporary_deficit_fraction * p_ctx.max_block_weight(block))
      );

      if (block_weights[block] > p_ctx.max_block_weight(block) + excess_budget ||
          block_weights[block] < p_ctx.min_block_weight(block) - deficit_budget) {
        return false;
      }
    }

    return true;
  }

  [[nodiscard]] bool
  is_feasible(const std::vector<BlockWeight> &block_weights, const PartitionContext &p_ctx) const {
    for (BlockID block = 0; block < block_weights.size(); ++block) {
      if (block_weights[block] > p_ctx.max_block_weight(block) ||
          block_weights[block] < p_ctx.min_block_weight(block)) {
        return false;
      }
    }

    return true;
  }

  [[nodiscard]] double
  balance_debt(const std::vector<BlockWeight> &block_weights, const PartitionContext &p_ctx) const {
    double debt = 0.0;
    for (BlockID block = 0; block < block_weights.size(); ++block) {
      if (block_weights[block] > p_ctx.max_block_weight(block)) {
        debt += block_weights[block] - p_ctx.max_block_weight(block);
      } else if (block_weights[block] < p_ctx.min_block_weight(block)) {
        debt += p_ctx.min_block_weight(block) - block_weights[block];
      }
    }

    return debt;
  }

  [[nodiscard]] std::vector<BlockWeight>
  current_block_weights(const PartitionedGraph &p_graph) const {
    std::vector<BlockWeight> block_weights(p_graph.k());
    for (const BlockID block : p_graph.blocks()) {
      block_weights[block] = p_graph.block_weight(block);
    }
    return block_weights;
  }

  [[nodiscard]] BlockWeight
  moved_weight_limit(const PartitionContext &p_ctx, const double fraction) const {
    if (fraction <= 0.0) {
      return 0;
    }
    return std::max<BlockWeight>(
        p_ctx.max_node_weight,
        static_cast<BlockWeight>(std::ceil(fraction * p_ctx.total_node_weight))
    );
  }

  [[nodiscard]] std::vector<BlockID>
  snapshot_partition(const PartitionedGraph &p_graph, const Graph &graph) const {
    std::vector<BlockID> partition(graph.n());
    for (const NodeID u : graph.nodes()) {
      partition[u] = p_graph.block(u);
    }
    return partition;
  }

  void restore_partition(
      PartitionedGraph &p_graph, const Graph &graph, const std::vector<BlockID> &partition
  ) const {
    for (const NodeID u : graph.nodes()) {
      if (p_graph.block(u) != partition[u]) {
        p_graph.set_block(u, partition[u]);
      }
    }
  }

  void run_fm_closure(PartitionedGraph &p_graph, const PartitionContext &p_ctx) const {
    Context closure_ctx = _ctx;
    closure_ctx.refinement.kway_fm.num_iterations = _r_ctx.fm_closure_iterations;
    closure_ctx.refinement.kway_fm.dbg_compute_batch_stats = false;
    closure_ctx.refinement.kway_fm.dbg_report_progress = false;

    FMRefiner fm(closure_ctx);
    fm.initialize(p_graph);
    fm.refine(p_graph, p_ctx);
  }

  void deduplicate_packets(std::vector<Packet> &packets) const {
    std::sort(packets.begin(), packets.end(), [](const Packet &lhs, const Packet &rhs) {
      return std::tie(lhs.source, lhs.target, lhs.vertices) <
             std::tie(rhs.source, rhs.target, rhs.vertices);
    });

    std::vector<Packet> deduplicated;
    for (Packet &packet : packets) {
      if (!deduplicated.empty() && deduplicated.back().source == packet.source &&
          deduplicated.back().target == packet.target &&
          deduplicated.back().vertices == packet.vertices) {
        if (packet.gain > deduplicated.back().gain) {
          deduplicated.back() = std::move(packet);
        }
      } else {
        deduplicated.push_back(std::move(packet));
      }
    }

    packets = std::move(deduplicated);
  }

  void filter_packets(std::vector<Packet> &packets) const {
    if (packets.size() <= _r_ctx.max_total_packets) {
      std::sort(packets.begin(), packets.end(), [](const Packet &lhs, const Packet &rhs) {
        return packet_priority(lhs) > packet_priority(rhs);
      });
      return;
    }

    std::vector<Packet> positive_packets;
    std::vector<Packet> repair_packets;
    for (Packet &packet : packets) {
      if (packet.gain > 0) {
        positive_packets.push_back(std::move(packet));
      } else {
        repair_packets.push_back(std::move(packet));
      }
    }

    const auto by_priority = [](const Packet &lhs, const Packet &rhs) {
      return packet_priority(lhs) > packet_priority(rhs);
    };
    std::sort(positive_packets.begin(), positive_packets.end(), by_priority);
    std::sort(repair_packets.begin(), repair_packets.end(), by_priority);

    packets.clear();
    const std::size_t positive_budget =
        std::min(positive_packets.size(), 3 * _r_ctx.max_total_packets / 4);
    std::move(
        positive_packets.begin(),
        positive_packets.begin() + positive_budget,
        std::back_inserter(packets)
    );

    const std::size_t repair_budget =
        std::min(repair_packets.size(), _r_ctx.max_total_packets - packets.size());
    std::move(
        repair_packets.begin(), repair_packets.begin() + repair_budget, std::back_inserter(packets)
    );

    const std::size_t remaining_budget = _r_ctx.max_total_packets - packets.size();
    const std::size_t remaining_positive_packets =
        std::min(remaining_budget, positive_packets.size() - positive_budget);
    std::move(
        positive_packets.begin() + positive_budget,
        positive_packets.begin() + positive_budget + remaining_positive_packets,
        std::back_inserter(packets)
    );

    std::sort(packets.begin(), packets.end(), by_priority);
  }

  void assign_packet_ids(std::vector<Packet> &packets) const {
    for (std::size_t id = 0; id < packets.size(); ++id) {
      packets[id].id = id;
    }
  }

  void keep_best_states(std::vector<MasterState> &states, const std::size_t max_size) const {
    if (states.size() > max_size) {
      std::nth_element(states.begin(), states.begin() + max_size, states.end(), better_state);
      states.resize(max_size);
    }
    std::sort(states.begin(), states.end(), better_state);
  }

private:
  const Context &_ctx;
  const RccpRefinementContext &_r_ctx;

  std::vector<int> _local_index;
};

RccpRefiner::RccpRefiner(const Context &ctx)
    : _csr_impl(std::make_unique<RccpRefinerCSRImpl>(ctx)),
      _compressed_impl(std::make_unique<RccpRefinerCompressedImpl>(ctx)) {}

RccpRefiner::~RccpRefiner() = default;

std::string RccpRefiner::name() const {
  return "RCCP";
}

void RccpRefiner::initialize(const PartitionedGraph &p_graph) {
  _csr_impl->initialize(p_graph);
  _compressed_impl->initialize(p_graph);
}

bool RccpRefiner::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  return reified(
      p_graph,
      [&](const auto &graph) { return _csr_impl->refine(p_graph, graph, p_ctx); },
      [&](const auto &graph) { return _compressed_impl->refine(p_graph, graph, p_ctx); }
  );
}

} // namespace kaminpar::shm
