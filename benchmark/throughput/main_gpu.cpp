
#include <algorithm>
#include <chrono>
#include <fstream>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/DataFormats/PointsHost.hpp"
#include "CLUEstering/DataFormats/PointsDevice.hpp"

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

namespace backend = ALPAKA_ACCELERATOR_NAMESPACE_CLUE;

using Event = clue::PointsHost<3>;
using EventPool = oneapi::tbb::concurrent_vector<Event>;
using Queue = backend::Queue;
using Platform = backend::Platform;

using QueuePool = std::vector<Queue>;
using ClustererPool = std::vector<clue::Clusterer<3>>;

constexpr float dc = 1.5f, rhoc = 10.f, dm = 1.5f;
constexpr int blocksize = 512;

float mean(const std::vector<float>& values) {
  return std::accumulate(values.begin(), values.end(), 0.f) / values.size();
}

float stddev(const std::vector<float>& values) {
  auto mean_ = mean(values);
  auto view = values |
              std::views::transform([mean_](auto x) -> float { return (x - mean_) * (x - mean_); });
  auto sqSize = values.size() * (values.size() - 1);
  return std::sqrt(std::accumulate(view.begin(), view.end(), 0.f) / sqSize);
}

// #ifdef PYBIND11
// void plot(const TimeMeasures& measures, const std::string& filename) {
//   py::scoped_interpreter guard{};
//   py::module plt = py::module::import("matplotlib.pyplot");
//   py::bind_vector<std::vector<int>>(plt, "VectorInt");
//   py::bind_vector<std::vector<float>>(plt, "VectorFloat");
//   plt.attr("errorbar")(
//       measures.sizes, measures.time_averages, measures.time_stddevs, "fmt"_a = "r--^");
//   plt.attr("xlabel")("Number of points");
//   plt.attr("ylabel")("Throughput (events/s)");
//   plt.attr("grid")("ls"_a = "--", "lw"_a = .5);
//   plt.attr("savefig")(filename);
// }
// #endif

// void to_csv(const TimeMeasures& measures, const std::string& filename) {
//   std::ofstream file{filename};
//   if (!file.is_open()) {
//     std::cerr << "Error opening file " << filename << std::endl;
//     return;
//   }

//   file << "size,avg,std\n";
//   for (auto i = 0ul; i < measures.sizes.size(); ++i) {
//     file << measures.sizes[i] << "," << measures.time_averages[i] << "," << measures.time_stddevs[i]
//          << "\n";
//   }
//   file.close();
// }

double runEvents(int nThreads, int nEvents, int nClusters) {
  const auto device = alpaka::getDevByIdx(alpaka::Platform<backend::Acc1D>{}, 0u);

  EventPool eventPool;
  QueuePool queuePool;
  ClustererPool clustererPool;

  for (auto i = 0; i < nThreads; ++i) {
    queuePool.emplace_back(backend::Queue(device));
  }

  std::array<std::pair<float, float>, 3> boundaries = {
      std::make_pair(0.f, 300.f), std::make_pair(0.f, 300.f), std::make_pair(0.f, 100.f)};
  for (auto i = 0; i < nEvents; ++i) {
    eventPool.emplace_back(
        clue::utils::generateClustersWithEnergy<3>(queuePool[0],
                                                   120,
                                                   512,
                                                   boundaries,
                                                   std::make_pair(0.f, 3.f),
                                                   std::array<float, 3>({1.f, 1.f, 5.f}),
                                                   0));
  }
  for (auto& queue : queuePool) {
    clustererPool.emplace_back(clue::Clusterer<3>(queue, dc, rhoc, dm));
  }

  std::atomic<int> eventCounter = 0;
  tbb::task_arena arena(nThreads);
  auto start = std::chrono::high_resolution_clock::now();
  arena.execute([&] {
    tbb::parallel_for(0, nThreads, [&](int i) {
      auto& queue = queuePool[i];
      auto& clusterer = clustererPool[i];
      clue::PointsDevice<3, backend::Device> d_points(queue, eventPool[0].size());
      while (eventCounter < nEvents) {
        int eventId = eventCounter.fetch_add(1);
        if (eventId >= nEvents)
          return;

        auto& h_points = eventPool[eventId];
        clusterer.make_clusters(h_points, d_points, FlatKernel{.5f}, queue, blocksize);
      }
    });
  });
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  return (1000. * nEvents) / duration;
}

int main(int argc, char* argv[]) {
  auto min = std::stoi(argv[1]);
  auto max = std::stoi(argv[2]);
  auto nEvents = std::stoi(argv[3]);
  auto nThreads = std::stoi(argv[4]);
  auto range = max - min + 1;

  std::string oFilename{"measures.csv"};
  if (argc == 4) {
    oFilename = argv[5];
  }

  std::vector<int> sizes(range);
  std::vector<float> throughput(range);
  std::ranges::for_each(std::views::iota(min) | std::views::take(range), [&](auto i) -> void {
    const auto nClusters = static_cast<std::size_t>(std::pow(2, i));
    std::cout << nClusters << " " << runEvents(nThreads, nEvents, nClusters) << std::endl;
  });

  // #ifdef PYBIND11
  //   auto figname = oFilename.substr(0, oFilename.find_last_of('.')) + ".pdf";
  //   plot(measures, figname);
  // #endif
  //   to_csv(measures, oFilename);
}
