
#include "header.hpp"
#include <iostream>

#include <algorithm>
#include <chrono>
#include <ranges>
#include <string>
#include <vector>

#include "utils/generation.hpp"

#include <oneapi/tbb/concurrent_vector.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#ifdef PYBIND11
#include <pybind11/embed.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
#endif

namespace cpu = serial;
namespace gpu = cuda;

// using Event = clue::PointsHost<3>;
// using EventPool = oneapi::tbb::concurrent_vector<Event>;
// using Queue = backend::Queue;
// using Platform = backend::Platform;

// using QueuePool = std::vector<Queue>;
// using ClustererPool = std::vector<clue::Clusterer<3>>;

using Times = oneapi::tbb::concurrent_vector<long long>;

constexpr float dc = 1.5f, rhoc = 10.f, dm = 1.5f;
constexpr int blocksize = 512;

int main() {
  auto c1 = serial::foo(1.f, 1.f, 1.f);
  auto c2 = cuda::foo(1.f, 1.f, 1.f);
}
