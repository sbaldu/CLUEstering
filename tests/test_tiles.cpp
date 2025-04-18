
#include "DataFormats/alpaka/temp.hpp"
#include "utility/read_csv.hpp"

#include <array>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test Build Tiles") {
  using namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE;

  const auto device = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
  Queue queue(device);

  const auto n_points = 1 << 10;
  const auto n_tiles = 1000;
  const std::array<std::array<float, 2>, 3> extremes{
      {-1.f, 2.f}, {0.f, 3.f}, {0.2f, 0.4f}};
  auto tiles = clue::buildTiles(queue, n_points, n_tiles, extremes);
}

TEST_CASE("Test execution with externally built tiles") {
  using namespace ALPAKA_ACCELERATOR_NAMESPACE;

  const auto device = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
  Queue queue(device);

  auto coords = read_csv<float, 2>("test_datasets/toyDetector.csv");
  const auto n_points = coords.size() / 3;
  std::vector<int> results(2 * n_points);

  PointsSoA<2> h_points(
      coords.data(), results.data(), PointInfo<2>{static_cast<uint32_t>(n_points)});
  PointsAlpaka<2> d_points(queue_, n_points);

  const auto n_tiles = 1000;
  const std::array<std::array<float, 2>, 3> extremes{{-235.44f, 242.24f},
                                                     {-243.1f, 243.14f}};
  auto tiles = buildTiles(queue, n_points, ntiles);

  const float dc{4.5f}, rhoc{2.5f}, outlier{4.5f};
  const int pPBin{128};
  CLUEAlgoAlpaka<2> algo(dc, rhoc, outlier, pPBin, queue_, tiles);

  const std::size_t block_size{256};
  algo.make_clusters(d_points, FlatKernel{.5f}, queue, block_size);
  auto clusters = algo.getClusters(h_points);

  CHECK(clusters.size() == );
  // auto ordered_clusters = clusters;
}

