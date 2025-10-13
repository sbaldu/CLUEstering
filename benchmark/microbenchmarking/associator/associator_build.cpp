
#include "CLUEstering/data_structures/internal/MakeAssociator.hpp"
#include <benchmark/benchmark.h>
#include <ranges>

static void BM_AssociatorBuild(benchmark::State& state) {
  for (auto _ : state) {
    const auto elements = state.range(0);
	std::vector<int> associations(elements);
    std::ranges::transform(std::views::iota(0, elements), associations.data(), [](auto x) -> int32_t {
      return x % 2 == 0;
    });
    volatile auto associator = clue::internal::make_associator(associations, elements);
  }
}

BENCHMARK(BM_AssociatorBuild)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);

BENCHMARK_MAIN();
