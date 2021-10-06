#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <allocator/alignedallocator.hpp>
#include <data-structures/table_config.hpp>
#include <tbb/enumerable_thread_specific.h>
#include <utils/hash/murmur2_hash.hpp>
#pragma GCC diagnostic pop

int main() {
  using hasher_type = utils_tm::hash_tm::murmur2_hash;
  using allocator_type = growt::AlignedAllocator<>;
  using table_type = typename growt::table_config<std::uint64_t, std::int64_t, hasher_type, allocator_type,
                                                  hmod::growable, hmod::deletion>::table_type;

  table_type _cluster_weights{7};

  {
    auto handle = _cluster_weights.get_handle();
    handle.insert(1, 1);
  }
  {
    auto handle = _cluster_weights.get_handle();
    handle.insert(2, 1);
  }
  {
    auto handle = _cluster_weights.get_handle();
    handle.insert(3, 1);
  }
  {
    auto handle = _cluster_weights.get_handle();
    handle.insert(8, 1);
  }
  {
    auto handle = _cluster_weights.get_handle();
    handle.insert(4, 1);
  }
  {
    auto handle = _cluster_weights.get_handle();
    handle.insert(6, 1);
  }
  {
    auto handle = _cluster_weights.get_handle();
    handle.insert(9, 1);
  }


  const std::int32_t delta = 1;

  std::cout << "jetzt das update:" << std::endl;
  auto handle = _cluster_weights.get_handle();
  [[maybe_unused]] const auto [old_it, old_found] = handle.update(
      1, [](auto &lhs, const auto rhs) { return lhs -= rhs; }, delta);

  std::cout << "worked out!" << std::endl;
  return 0;
}