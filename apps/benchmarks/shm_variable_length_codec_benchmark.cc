/*******************************************************************************
 * Variable length codec benchmark for the shared-memory algorithm.
 *
 * @file:   shm_variable_length_codec_benchmark.cc
 * @author: Daniel Salwasser
 * @date:   12.11.2023
 ******************************************************************************/
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "kaminpar-cli/CLI11.h"

#include "kaminpar-common/console_io.h"
#include "kaminpar-common/graph_compression/streamvbyte.h"
#include "kaminpar-common/graph_compression/varint.h"
#include "kaminpar-common/graph_compression/varint_rle.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

using namespace kaminpar;

enum class IntType {
  INT_32,
  INT_64
};

std::unordered_map<std::string, IntType> get_int_types() {
  return {
      {"int32", IntType::INT_32},
      {"int64", IntType::INT_64},
  };
}

template <class T> static inline void do_not_optimize(T value) {
  asm volatile("" : "+m"(value) : : "memory");
}

template <typename Int> std::vector<Int> generate_random_values(const std::size_t count) {
  std::vector<Int> random_values;
  random_values.resize(count);

  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(
      std::numeric_limits<Int>::min(), std::numeric_limits<Int>::max()
  );
  for (std::size_t i = 0; i < count; ++i) {
    random_values[i] = dist(rng);
  }

  return random_values;
}

template <typename Int, typename Lambda>
std::unique_ptr<std::uint8_t[]>
encode_values(std::string_view name, const std::size_t count, Lambda &&l) {
  auto encoded_values = std::make_unique<std::uint8_t[]>(count * varint_max_length<Int>());

  TIMED_SCOPE(name) {
    std::uint8_t *ptr = encoded_values.get();

    for (std::size_t i = 0; i < count; ++i) {
      const std::size_t bytes_written = varint_encode(l(i), ptr);
      ptr += bytes_written;
    }
  };

  return encoded_values;
}

template <typename Int, typename Lambda>
std::unique_ptr<std::uint8_t[]>
encode_signed_values(std::string_view name, const std::size_t count, Lambda &&l) {
  auto encoded_values = std::make_unique<std::uint8_t[]>(count * varint_max_length<Int>());

  TIMED_SCOPE(name) {
    std::uint8_t *ptr = encoded_values.get();

    for (std::size_t i = 0; i < count; ++i) {
      const std::size_t bytes_written = signed_varint_encode(l(i), ptr);
      ptr += bytes_written;
    }
  };

  return encoded_values;
}

template <typename Int, typename Lambda>
std::unique_ptr<std::uint8_t[]>
rl_encode_values(std::string_view name, const std::size_t count, Lambda &&l) {
  auto encoded_values = std::make_unique<std::uint8_t[]>(count * sizeof(Int) + count);

  TIMED_SCOPE(name) {
    VarIntRunLengthEncoder<Int> encoder(encoded_values.get());

    for (std::size_t i = 0; i < count; ++i) {
      const std::size_t bytes_written = encoder.add(l(i));
      do_not_optimize(bytes_written);
    }

    encoder.flush();
  };

  return encoded_values;
}

template <typename Int, typename Lambda>
std::unique_ptr<std::uint8_t[]>
sv_encode_values(std::string_view name, const std::size_t count, Lambda &&l) {
  auto encoded_values = std::make_unique<std::uint8_t[]>(count * sizeof(Int) + count);

  TIMED_SCOPE(name) {
    streamvbyte::StreamVByteEncoder<Int> encoder(count, encoded_values.get());

    for (std::size_t i = 0; i < count; ++i) {
      const std::size_t bytes_written = encoder.add(l(i));
      do_not_optimize(bytes_written);
    }

    encoder.flush();
  };

  return encoded_values;
}

template <typename Int>
std::tuple<
    std::unique_ptr<std::uint8_t[]>,
    std::unique_ptr<std::uint8_t[]>,
    std::unique_ptr<std::uint8_t[]>>
encode_values(const std::size_t count, const std::vector<Int> &random_values) {
  SCOPED_TIMER("Encoding");

  return std::make_tuple(
      encode_values<Int>("Encoding zero values", count, [](std::size_t) { return 0; }),
      encode_values<Int>(
          "Encoding max values", count, [](std::size_t) { return std::numeric_limits<Int>::max(); }
      ),
      encode_values<Int>(
          "Encoding random values", count, [&](const std::size_t i) { return random_values[i]; }
      )
  );
}

template <typename Int>
std::tuple<
    std::unique_ptr<std::uint8_t[]>,
    std::unique_ptr<std::uint8_t[]>,
    std::unique_ptr<std::uint8_t[]>>
encode_signed_values(const std::size_t count, const std::vector<Int> &random_values) {
  SCOPED_TIMER("Encoding signed values");

  return std::make_tuple(
      encode_signed_values<Int>("Encoding zero values", count, [](std::size_t) { return 0; }),
      encode_signed_values<Int>(
          "Encoding max values", count, [](std::size_t) { return std::numeric_limits<Int>::max(); }
      ),
      encode_signed_values<Int>(
          "Encoding random values", count, [&](const std::size_t i) { return random_values[i]; }
      )
  );
}

template <typename Int>
std::tuple<
    std::unique_ptr<std::uint8_t[]>,
    std::unique_ptr<std::uint8_t[]>,
    std::unique_ptr<std::uint8_t[]>>
rl_encode_values(const std::size_t count, const std::vector<Int> &random_values) {
  SCOPED_TIMER("Encoding run-length");

  return std::make_tuple(
      rl_encode_values<Int>("Encoding zero values", count, [](std::size_t) { return 0; }),
      rl_encode_values<Int>(
          "Encoding max values", count, [](std::size_t) { return std::numeric_limits<Int>::max(); }
      ),
      rl_encode_values<Int>(
          "Encoding random values", count, [&](const std::size_t i) { return random_values[i]; }
      )
  );
}

template <typename Int>
std::tuple<
    std::unique_ptr<std::uint8_t[]>,
    std::unique_ptr<std::uint8_t[]>,
    std::unique_ptr<std::uint8_t[]>>
sv_encode_values(const std::size_t count, const std::vector<Int> &random_values) {
  SCOPED_TIMER("Encoding stream");

  return std::make_tuple(
      sv_encode_values<Int>("Encoding zero values", count, [](std::size_t) { return 0; }),
      sv_encode_values<Int>(
          "Encoding max values", count, [](std::size_t) { return std::numeric_limits<Int>::max(); }
      ),
      sv_encode_values<Int>(
          "Encoding random values", count, [&](const std::size_t i) { return random_values[i]; }
      )
  );
}

template <typename Lambda>
void benchmark(
    std::string_view name, const std::size_t count, const std::uint8_t *values_ptr, Lambda &&l
) {
  SCOPED_TIMER(name);

  for (std::size_t i = 0; i < count; ++i) {
    const auto value = l(&values_ptr);
    do_not_optimize(value);
  }
}

template <typename Int>
void benchmark_rle(std::string_view name, const std::size_t count, const std::uint8_t *values_ptr) {
  SCOPED_TIMER(name);

  VarIntRunLengthDecoder<Int> decoder(count, values_ptr);
  decoder.decode([](const Int value) { do_not_optimize(value); });
}

template <typename Int>
void benchmark_sve(std::string_view name, const std::size_t count, const std::uint8_t *values_ptr) {
  SCOPED_TIMER(name);

  streamvbyte::StreamVByteDecoder<Int> decoder(count, values_ptr);
  decoder.decode([](const Int value) { do_not_optimize(value); });
}

template <typename Lambda>
void benchmark(
    std::string_view name,
    const std::size_t count,
    const std::uint8_t *zero_values_ptr,
    const std::uint8_t *max_values_ptr,
    const std::uint8_t *random_values_ptr,
    Lambda &&l
) {
  SCOPED_TIMER(name);

  benchmark("Decoding zero", count, zero_values_ptr, std::forward<Lambda>(l));
  benchmark("Decoding max values", count, max_values_ptr, std::forward<Lambda>(l));
  benchmark("Decoding random values", count, random_values_ptr, std::forward<Lambda>(l));
}

template <typename Int>
void benchmark_rle(
    std::string_view name,
    const std::size_t count,
    const std::uint8_t *zero_values_ptr,
    const std::uint8_t *max_values_ptr,
    const std::uint8_t *random_values_ptr
) {
  SCOPED_TIMER(name);

  benchmark_rle<Int>("Decoding zero values", count, zero_values_ptr);
  benchmark_rle<Int>("Decoding max values", count, max_values_ptr);
  benchmark_rle<Int>("Decoding random values", count, random_values_ptr);
}

template <typename Int>
void benchmark_sve(
    std::string_view name,
    const std::size_t count,
    const std::uint8_t *zero_values_ptr,
    const std::uint8_t *max_values_ptr,
    const std::uint8_t *random_values_ptr
) {
  SCOPED_TIMER(name);

  benchmark_sve<Int>("Decoding zero values", count, zero_values_ptr);
  benchmark_sve<Int>("Decoding max values", count, max_values_ptr);
  benchmark_sve<Int>("Decoding random values", count, random_values_ptr);
}

template <typename Int> void run_benchmark(std::size_t count) {
  std::vector<Int> random_values = generate_random_values<Int>(count);

  const auto [encoded_zero_values, encoded_max_values, encoded_random_values] =
      encode_values<Int>(count, random_values);

  benchmark(
      "Decoding: loop",
      count,
      encoded_zero_values.get(),
      encoded_max_values.get(),
      encoded_random_values.get(),
      [](const std::uint8_t **ptr) { return varint_decode_loop<Int>(ptr); }
  );

  benchmark(
      "Decoding: unrolled + intrinsic",
      count,
      encoded_zero_values.get(),
      encoded_max_values.get(),
      encoded_random_values.get(),
      [](const std::uint8_t **ptr) { return varint_decode_pext_unrolled<Int>(ptr); }
  );

  /*
  std::vector<std::make_signed_t<Int>> random_signed_values =
      generate_random_values<std::make_signed_t<Int>>(count);

  const auto [encoded_zero_signed_values, encoded_max_signed_values, encoded_random_signed_values] =
      encode_signed_values<std::make_signed_t<Int>>(count, random_signed_values);

  benchmark(
      "Decoding signed: loop",
      count,
      encoded_zero_signed_values.get(),
      encoded_max_signed_values.get(),
      encoded_random_signed_values.get(),
      [](const std::uint8_t *ptr) {
        return signed_varint_decode_general<std::make_signed_t<Int>>(ptr);
      }
  );

  benchmark(
      "Decoding signed: unrolled + intrinsic",
      count,
      encoded_zero_signed_values.get(),
      encoded_max_signed_values.get(),
      encoded_random_signed_values.get(),
      [](const std::uint8_t *ptr) { return signed_varint_decode<std::make_signed_t<Int>>(ptr); }
  );
  */

  const auto [rl_encoded_zero_values, rl_encoded_max_values, rl_encoded_random_values] =
      rl_encode_values<Int>(count, random_values);

  benchmark_rle<Int>(
      "Decoding run-length",
      count,
      rl_encoded_zero_values.get(),
      rl_encoded_max_values.get(),
      rl_encoded_random_values.get()
  );

  if constexpr (sizeof(Int) == 4) {
    const auto [sv_encoded_zero_values, sv_encoded_max_values, sv_encoded_random_values] =
        sv_encode_values<Int>(count, random_values);

    benchmark_sve<Int>(
        "Decoding stream",
        count,
        sv_encoded_zero_values.get(),
        sv_encoded_max_values.get(),
        sv_encoded_random_values.get()
    );
  }
}

int main(int argc, char *argv[]) {
  // Parse CLI arguments
  IntType int_type = IntType::INT_32;
  std::size_t count = 100000000;

  CLI::App app("Shared-memory variable length codec benchmark");
  app.add_option("-n", count, "The amount of numbers to encode and decode")
      ->check(CLI::NonNegativeNumber)
      ->default_val(count);
  app.add_option("-i,--int", int_type)
      ->transform(CLI::CheckedTransformer(get_int_types()).description(""))
      ->description(R"(Select a int type. The options are:
                      - int32
                      - int64
        )");
  CLI11_PARSE(app, argc, argv);

  // Run Benchmark
  LOG << "Running the benchmark...";
  GLOBAL_TIMER.reset();

  switch (int_type) {
  case IntType::INT_32:
    run_benchmark<std::uint32_t>(count);
    break;
  case IntType::INT_64:
    run_benchmark<std::uint64_t>(count);
    break;
  };

  STOP_TIMER();

  // Print the result summary
  LOG;
  cio::print_delimiter("Result Summary");
  LOG << "Encoded and decoded " << count << " integers.";
  LOG;
  Timer::global().print_human_readable(std::cout);
}
