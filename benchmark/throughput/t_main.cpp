
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

// #include "utils/generation.hpp"

#include "clue_serial.hpp"
#include "clue_cuda.hpp"

#include <oneapi/tbb/concurrent_vector.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

// #ifdef PYBIND11
// #include <pybind11/embed.h>
// #include <pybind11/stl_bind.h>

// namespace py = pybind11;
// using namespace pybind11::literals;

// PYBIND11_MAKE_OPAQUE(std::vector<int>);
// PYBIND11_MAKE_OPAQUE(std::vector<float>);
// #endif

using Event = std::vector<float>;
using EventPool = oneapi::tbb::concurrent_vector<Event>;

constexpr float dc = 1.5f, rhoc = 10.f, outlier = 1.5f;
constexpr int blocksize = 512;

// struct TimeMeasures {
//   std::vector<int> sizes;
//   std::vector<float> time_averages;
//   std::vector<float> time_stddevs;

//   TimeMeasures(size_t size) : sizes(size), time_averages(size), time_stddevs(size) {}
// };

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
//   plt.attr("ylabel")("Execution time (ms)");
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

void runEvents(int nThreadsGPU, int nThreadsCPU, int nEvents) {
  const auto cpuDevice = alpaka::getDevByIdx(alpaka::Platform<alpaka_serial_sync::Acc1D>{}, 0u);
  const auto cudaDevice = alpaka::getDevByIdx(alpaka::Platform<alpaka_cuda_async::Acc1D>{}, 0u);

  EventPool eventPool(nEvents);
  serial::QueuePool cpuQueues(nThreadsCPU);
  serial::ClustererPool cpuClusterers(nThreadsCPU);
  cuda::QueuePool gpuQueues(nThreadsGPU);
  cuda::ClustererPool gpuClusterers(nThreadsGPU);

  std::generate(cpuQueues.begin(), cpuQueues.end(), [&cpuDevice] {
    return alpaka_serial_sync::Queue(cpuDevice);
  });
  std::generate(gpuQueues.begin(), gpuQueues.end(), [&cudaDevice] {
    return alpaka_cuda_async::Queue(cudaDevice);
  });

  std::ranges::transform(cpuQueues, cpuClusterers.begin(), [&queuePool](auto queue) {
    return alpaka_serial_sync::clue::Clusterer<3>(queue, dc, rhoc, dm);
  });
  std::ranges::transform(gpuQueues, clustererPool.begin(), [&queuePool](auto queue) {
    return alpaka_cuda_async::clue::Clusterer<3>(queue, dc, rhoc, dm);
  });

  std::atomic<int> eventCounter = 0;
  tbb::task_arena cpu_arena(nThreadsCPU);
  tbb::task_arena gpu_arena(nThreadsGPU);
  cpu_arena.execute([&] {
    tbb::parallel_for(0, nThreadsCPU, [&](int i) {
      auto& queue = cpuQueues[i];
      auto& clusterer = cpuClusterers[i];
      while (eventCounter < nEvents) {
        int eventId = eventCounter.fetch_add(1);
        if (eventId >= nEvents)
          return;

        // Create the points host and device objects
        // clue::PointsHost<2> h_points(queue, nPoints);
        // clue::PointsDevice<2, Device> d_points(queue, nPoints);
        // // clue::utils::generateRandomData<2>(h_points, 20, std::make_pair(-100.f, 100.f), 1.f);

        // clusterer.make_clusters(h_points, d_points, FlatKernel{.5f}, queue, blocksize);
        // alpaka::wait(queue);
      }
    });
  });
  gpu_arena.execute([&] {
    tbb::parallel_for(0, nThreadsGPU, [&](int i) {
      auto& queue = gpuQueues[i];
      auto& clusterer = gpuClusterers[i];
      while (eventCounter < nEvents) {
        int eventId = eventCounter.fetch_add(1);
        if (eventId >= nEvents)
          return;

        // Create the points host and device objects
        clue::PointsHost<2> h_points(queue, nPoints);
        clue::PointsDevice<2, Device> d_points(queue, nPoints);
        // clue::utils::generateRandomData<2>(h_points, 20, std::make_pair(-100.f, 100.f), 1.f);

        clusterer.make_clusters(h_points, d_points, FlatKernel{.5f}, queue, blocksize);
        alpaka::wait(queue);
      }
    });
  });
}

// int main(int argc, char* argv[]) {
//   auto min = std::stoi(argv[1]);
//   auto max = std::stoi(argv[2]);
//   auto nEvents = std::stoi(argv[3]);
//   auto nThreads = std::stoi(argv[4]);
//   auto range = max - min + 1;

//   std::string oFilename{"measures.csv"};
//   if (argc == 4) {
//     oFilename = argv[5];
//   }

//   TimeMeasures measures(range);
//   auto& sizes = measures.sizes;
//   auto& time_averages = measures.time_averages;
//   auto& time_stddevs = measures.time_stddevs;

//   auto avgIt = time_averages.begin();
//   auto stdIt = time_stddevs.begin();
//   auto sizeIt = sizes.begin();
//   const auto device = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
//   std::ranges::for_each(
//       std::views::iota(min) | std::views::take(range),
//       [nruns, &device, &sizeIt, &avgIt, &stdIt](auto i) -> void {
//         EventPool eventPool;
//         Queue queue(device);

//         const auto n_points = static_cast<std::size_t>(std::pow(2, i));

//         // Create the points host and device objects
//         clue::PointsHost<2> h_points(queue, n_points);
//         clue::PointsDevice<2, Device> d_points(queue, n_points);
//         clue::utils::generateRandomData<2>(h_points, 20, std::make_pair(-100.f, 100.f), 1.f);

//         auto start = std::chrono::high_resolution_clock::now();
//         auto end = std::chrono::high_resolution_clock::now();
//         std::vector<float> times(nruns);
//         for (auto i = 0; i < nruns; ++i) {
//           start = std::chrono::high_resolution_clock::now();
//           run(h_points, d_points, queue);
//           end = std::chrono::high_resolution_clock::now();
//           auto duration =
//               std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//           times[i] = duration;
//         }
//         *sizeIt++ = n_points;
//         *avgIt++ = mean(times);
//         *stdIt++ = stddev(times);
//       });

// #ifdef PYBIND11
//   auto figname = oFilename.substr(0, oFilename.find_last_of('.')) + ".pdf";
//   plot(measures, figname);
// #endif
//   to_csv(measures, oFilename);
// }

int main() {}
