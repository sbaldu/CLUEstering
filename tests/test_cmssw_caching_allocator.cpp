#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/utils/detail/get_cluster_properties.hpp"

#include "cmssw/AllocatorConfig.hpp"
#include "cmssw/CachingAllocator.hpp"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

  using CachingAllocator = cms::alpakatools::CachingAllocator<clue::Device, clue::Queue>;

  // Minimal wrapper adapting the CMSSW caching allocator to the allocator interface
  struct AllocatorWrapper {
    CachingAllocator* allocator;
    clue::Queue queue;

    void* allocate(std::size_t bytes, [[maybe_unused]] std::size_t align) {
      return allocator->allocate(bytes, queue);
    }

    void deallocate(void* ptr) { allocator->free(ptr); }
  };

  static_assert(clue::concepts::external_allocator<AllocatorWrapper>);

  // Check that two clusterings define the same partition of the points.
  // The cluster ids are assigned in parallel, so on multi-threaded backends they can
  // differ between runs, but there must be a one-to-one mapping between them.
  bool same_clusters(std::span<const int> lhs, std::span<const int> rhs) {
    if (lhs.size() != rhs.size())
      return false;

    std::unordered_map<int, int> lhs_to_rhs, rhs_to_lhs;
    for (auto i = 0u; i < lhs.size(); ++i) {
      const auto [it_lhs, new_lhs] = lhs_to_rhs.try_emplace(lhs[i], rhs[i]);
      const auto [it_rhs, new_rhs] = rhs_to_lhs.try_emplace(rhs[i], lhs[i]);
      if (it_lhs->second != rhs[i] || it_rhs->second != lhs[i])
        return false;
    }
    // outliers must be the same points in both clusterings
    return !lhs_to_rhs.contains(-1) || lhs_to_rhs.at(-1) == -1;
  }

  CachingAllocator make_caching_allocator(const clue::Device& device) {
    namespace config = cms::alpakatools::config;
    return CachingAllocator(device,
                            config::binGrowth,
                            config::minBin,
                            config::maxBin,
                            config::maxCachedBytes,
                            config::maxCachedFraction,
                            true,    // reuseSameQueueAllocations
                            false);  // debug
  }

}  // namespace

TEST_CASE("Test clustering with the CMSSW caching allocator") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/data_32768.csv";
  const float dc{1.3f}, rhoc{10.f}, outlier{1.3f};

  clue::PointsHost<2> h_points_default = clue::read_csv<2, float>(queue, test_file_path);
  clue::Clusterer<2> algo_default(queue, dc, rhoc, outlier);
  algo_default.make_clusters(queue, h_points_default);
  alpaka::wait(queue);

  // the caching allocator must outlive every buffer allocated through it
  CachingAllocator caching_allocator = make_caching_allocator(device);
  AllocatorWrapper allocator{&caching_allocator, queue};

  SUBCASE("Results match the default allocator") {
    clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
    clue::Clusterer<2, float, AllocatorWrapper> algo(queue, allocator, dc, rhoc, outlier);
    algo.make_clusters(queue, h_points);
    alpaka::wait(queue);

    CHECK(same_clusters(h_points.clusterIndexes(), h_points_default.clusterIndexes()));
  }

  SUBCASE("Results match the default allocator using device points") {
    clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
    clue::PointsDevice<2> d_points(queue, h_points.size(), allocator);
    clue::Clusterer<2, float, AllocatorWrapper> algo(queue, allocator, dc, rhoc, outlier);
    algo.make_clusters(queue, h_points, d_points);
    alpaka::wait(queue);

    CHECK(same_clusters(h_points.clusterIndexes(), h_points_default.clusterIndexes()));
  }

  SUBCASE("Internal buffers are allocated through the caching allocator") {
    {
      clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
      clue::Clusterer<2, float, AllocatorWrapper> algo(queue, allocator, dc, rhoc, outlier);
      algo.make_clusters(queue, h_points);
      alpaka::wait(queue);

      // the clusterer keeps its internal buffers alive between runs
      CHECK(caching_allocator.cacheStatus().live > 0);
    }
    alpaka::wait(queue);

    // once the clusterer is destroyed all the memory is returned to the cache
    const auto bytes = caching_allocator.cacheStatus();
    CHECK(bytes.live == 0);
    CHECK(bytes.free > 0);
  }

  SUBCASE("Cached memory is reused by a new clusterer") {
    {
      clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
      clue::Clusterer<2, float, AllocatorWrapper> algo(queue, allocator, dc, rhoc, outlier);
      algo.make_clusters(queue, h_points);
      alpaka::wait(queue);
    }
    alpaka::wait(queue);
    const auto first_run = caching_allocator.cacheStatus();

    {
      clue::PointsHost<2> h_points = clue::read_csv<2, float>(queue, test_file_path);
      clue::Clusterer<2, float, AllocatorWrapper> algo(queue, allocator, dc, rhoc, outlier);
      algo.make_clusters(queue, h_points);
      alpaka::wait(queue);

      CHECK(same_clusters(h_points.clusterIndexes(), h_points_default.clusterIndexes()));
    }
    alpaka::wait(queue);
    const auto second_run = caching_allocator.cacheStatus();

    // no new memory was requested for the second run
    CHECK(second_run.live == 0);
    CHECK(second_run.free == first_run.free);
  }
}

TEST_CASE("Test batched clustering with the CMSSW caching allocator") {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);

  CachingAllocator caching_allocator = make_caching_allocator(device);
  AllocatorWrapper allocator{&caching_allocator, queue};

  clue::PointsHost<2> h_points =
      clue::read_csv<2, float>(queue, std::string(TEST_DATA_DIR) + "/batched_data_1024.csv");
  clue::PointsDevice<2> d_points(queue, h_points.size(), allocator);

  const float dc{1.3f}, rhoc{10.f}, outlier{1.3f};
  clue::Clusterer<2, float, AllocatorWrapper> algo(queue, allocator, dc, rhoc, outlier);

  std::vector<uint32_t> event_sizes(10, 1024);
  algo.make_clusters(queue, h_points, d_points, event_sizes);
  alpaka::wait(queue);

  auto truth = clue::read_output<2, float>(
      queue, std::string(TEST_DATA_DIR) + "/truth_files/data_1024_truth.csv");
  auto truth_n_clusters = clue::detail::compute_nclusters(truth.clusterIndexes());
  auto n_clusters = clue::detail::compute_nclusters(h_points.clusterIndexes());
  CHECK(n_clusters == truth_n_clusters * 10);
}
