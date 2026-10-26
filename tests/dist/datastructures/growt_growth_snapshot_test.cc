#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <thread>

// Only the generated test-private header calls this hook. It pauses a handle
// after actual protection, allowing actual Growt growth to publish a newer table
// before that handle checks its captured table version against the epoch.
namespace growt_snapshot_test {
std::atomic<bool> captured{false};
std::atomic<bool> resume{false};
thread_local bool armed = false;

void after_protect() {
  if (!armed) {
    return;
  }
  armed = false;
  captured.store(true, std::memory_order_release);
  while (!resume.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
}
} // namespace growt_snapshot_test

#include <allocator/alignedallocator.hpp>
#include <data-structures/table_config.hpp>
#include <utils/hash/murmur2_hash.hpp>

namespace {
using Table = growt::table_config<
    std::uint64_t,
    std::uint64_t,
    utils_tm::hash_tm::murmur2_hash,
    growt::AlignedAllocator<>,
    hmod::growable,
    hmod::deletion>::table_type;

[[noreturn]] void fail(const char *message) {
  std::cerr << message << std::endl;
  std::_Exit(EXIT_FAILURE);
}

void wait_for(const std::atomic<bool> &flag) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!flag.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      fail("Growt regression did not reach its scheduled race");
    }
    std::this_thread::yield();
  }
}
} // namespace

int main(const int argc, char **argv) {
  if (argc != 2) {
    fail("Expected init or load mode");
  }
  const std::string_view mode = argv[1];
  if (mode != "init" && mode != "load") {
    fail("Expected init or load mode");
  }

  Table table(0);
  auto creator = table.get_handle();
  std::uint64_t inserted = 1;
  if (!creator.insert(1, 18).second) {
    fail("Could not insert initial key");
  }
  const auto force_growth = [&] {
    const auto before = creator.capacity();
    while (creator.capacity() == before) {
      if (++inserted > 10000000 || !creator.insert(inserted, inserted + 17).second) {
        fail("Could not trigger real table growth");
      }
    }
    if (creator.capacity() <= before) {
      fail("Table capacity did not increase");
    }
    std::cout << "GROWTH_COMPLETE " << before << " -> " << creator.capacity() << std::endl;
  };

  std::atomic<bool> ready{false};
  std::atomic<bool> start_load{false};
  std::thread target([&] {
    if (mode == "init") {
      growt_snapshot_test::armed = true;
    }
    auto handle = table.get_handle();
    if (mode == "load") {
      ready.store(true, std::memory_order_release);
      wait_for(start_load);
      growt_snapshot_test::armed = true;
    }
    auto it = handle.find(1);
    if (it == handle.end() || static_cast<std::uint64_t>((*it).second) != 18) {
      fail("Target handle did not recover the original payload");
    }
    std::cout << "TARGET_RETURNED" << std::endl;
  });

  if (mode == "load") {
    wait_for(ready);
    force_growth(); // Invalidate the target's cached epoch so find enters load().
    start_load.store(true, std::memory_order_release);
  }
  wait_for(growt_snapshot_test::captured);
  std::cout << "SNAPSHOT_CAPTURED " << mode << std::endl;
  force_growth(); // Advance the epoch while the target protects the old table.
  growt_snapshot_test::resume.store(true, std::memory_order_release);
  std::cout << "SNAPSHOT_RESUMED mode=" << mode << " inserted=" << inserted << std::endl;
  target.join(); // The old implementation spins forever; CTest bounds this wait.

  for (std::uint64_t key = 1; key <= inserted; ++key) {
    auto it = creator.find(key);
    if (it == creator.end() || static_cast<std::uint64_t>((*it).second) != key + 17) {
      fail("Growth lost a key or changed its payload");
    }
  }
  std::cout << "PASS " << mode << ": verified " << inserted << " key/value pairs\n";
}
