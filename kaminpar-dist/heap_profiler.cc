/*******************************************************************************
 * Functions to annotate the heap profiler tree with aggregated information from
 * all PEs.
 *
 * @file:   heap_profiler.h
 * @author: Daniel Salwasser
 * @date:   16.06.2024
 ******************************************************************************/
#include "kaminpar-dist/heap_profiler.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <numeric>
#include <sstream>

#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-common/heap_profiler.h"

namespace kaminpar::dist {

namespace {

using HeapProfiler = heap_profiler::HeapProfiler;
using HeapProfilerTree = HeapProfiler::HeapProfileTree;
using HeapProfilerTreeNode = HeapProfiler::HeapProfileTreeNode;

std::string to_megabytes(std::size_t bytes) {
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << (bytes / (float)(1024 * 1024));
  return stream.str();
}

template <std::size_t kSize>
std::vector<std::string> gather_trunc_string(const std::string_view str, MPI_Comm comm) {
  std::array<char, kSize> trunc;
  const std::size_t len = std::min(kSize - 1, str.length());
  str.copy(trunc.data(), len);
  trunc[len] = 0;

  const auto [size, rank] = mpi::get_comm_info(comm);
  std::vector<char> recv_buffer(size * kSize);
  mpi::allgather(trunc.data(), kSize, recv_buffer.data(), kSize, comm);

  std::vector<std::string> strings;
  for (mpi::PEID pe = 0; pe < size; ++pe) {
    strings.emplace_back(recv_buffer.data() + pe * kSize);
  }

  return strings;
}

void generate_statistics(
    HeapProfilerTreeNode *node,
    const std::size_t mem_str_width,
    const std::size_t pe_str_width,
    const int root,
    MPI_Comm comm
) {
  constexpr std::size_t kTruncSize = 1024;

  const auto names = gather_trunc_string<kTruncSize>(node->name, comm);
  const bool diverged_node = std::all_of(names.begin(), names.end(), [&](const std::string &name) {
    return name.substr(0, kTruncSize) != node->name.substr(0, kTruncSize);
  });

  if (diverged_node) {
    return;
  }

  const auto stats = mpi::gather<std::size_t>(node->peak_memory, root, comm);
  const auto num_children = mpi::allgather(node->children.size(), comm);
  const bool is_root = mpi::get_comm_rank(comm) == root;

  if (is_root) {
    const auto min_it = std::min_element(stats.begin(), stats.end());
    const mpi::PEID min_pe = std::distance(stats.begin(), min_it);
    const std::size_t min = *min_it;

    const auto max_it = std::max_element(stats.begin(), stats.end());
    const mpi::PEID max_pe = std::distance(stats.begin(), max_it);
    const std::size_t max = *max_it;

    const auto sum = static_cast<double>(std::accumulate(stats.begin(), stats.end(), 0.0));
    const auto mean = sum / static_cast<double>(stats.size());

    const auto pad = [](auto value, const std::size_t width) {
      std::string str;
      if constexpr (std::is_same_v<decltype(value), std::string>) {
        str = std::move(value);
      } else {
        str = std::to_string(value);
      }

      if (str.length() < width) {
        str = std::string(width - str.length(), ' ') + str;
      }

      return str;
    };

    std::stringstream stream;
    stream << "[ " << pad(min_pe, pe_str_width) << " : " << pad(to_megabytes(min), mem_str_width)
           << " mb | " << pad(to_megabytes(mean), mem_str_width) << " mb | "
           << pad(max_pe, pe_str_width) << " : " << pad(to_megabytes(max), mem_str_width)
           << " mb ]";

    node->annotation = stream.str();
  }

  const bool nondiverged_children =
      std::all_of(num_children.begin(), num_children.end(), [&](const std::size_t num) {
        return num == node->children.size();
      });
  if (nondiverged_children) {
    for (HeapProfilerTreeNode *child : node->children) {
      generate_statistics(child, mem_str_width, pe_str_width, root, comm);
    }
  }
}

std::pair<mpi::PEID, std::size_t>
gather_max_peak_memory(const HeapProfilerTreeNode *node, MPI_Comm comm) {
  const auto stats = mpi::allgather<std::size_t>(node->peak_memory, comm);

  const auto max_it = std::max_element(stats.begin(), stats.end());
  const mpi::PEID max_pe = std::distance(stats.begin(), max_it);
  const std::size_t max = *max_it;

  return std::make_pair(max_pe, max);
}

} // namespace

int finalize_distributed_heap_profiler(heap_profiler::HeapProfiler &heap_profiler, MPI_Comm comm) {
  HeapProfilerTree &tree = heap_profiler.tree_root();

  const auto [root, max_peak_memory] = gather_max_peak_memory(&tree.root, comm);
  const std::size_t mem_str_width = to_megabytes(max_peak_memory).length();
  const std::size_t pe_str_width = std::to_string(mpi::get_comm_size(comm)).length();

  std::stringstream stream;
  stream << "PE" << std::string(pe_str_width - 1, ' ') << " : "
         << "min" << std::string(mem_str_width + 3, ' ') << "avg"
         << std::string(mem_str_width + 2, ' ') << "PE" << std::string(pe_str_width - 1, ' ')
         << " : max";

  tree.annotation = stream.str();
  generate_statistics(&tree.root, mem_str_width, pe_str_width, root, comm);
  return root;
}

} // namespace kaminpar::dist
