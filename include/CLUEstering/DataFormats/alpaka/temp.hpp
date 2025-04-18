
#pragma once

#include "TilesAlpaka.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <ranges>

namespace clue {

  template <typename TQueue, uint8_t Ndim>
  TilesAlpaka<Ndim> buildTiles(
      TQueue& queue,
      uint32_t n_points,
      int32_t n_tiles,
      const std::array<std::array<float, 2>, Ndim>& coordinate_extremes) {
    const auto nPerDim = static_cast<int32_t>(std::ceil(std::pow(n_tiles, 1. / Ndim)));
    n_tiles = static_cast<int32_t>(std::pow(nPerDim, Ndim));
    TilesAlpaka<Ndim> tiles(queue, n_points, n_tiles);

    using Device = decltype(alpaka::getDev(queue));
	tiles->initialize(n_points, n_tiles, nPerDim, queue);

    std::array<float, Ndim> tiles_sizes;
    std::ranges::transform(
        coordinate_extremes, tiles_sizes.begin(), [](const auto& dimPair) -> float {
          return dimPair[1] - dimPair[0];
        });
    const auto device = alpaka::getDev(queue);
    alpaka::memcpy(queue, tiles->minMax(), coordinate_extremes);
    alpaka::memcpy(queue, tiles->tileSize(), tiles_sizes);
    alpaka::wait(queue);

    return tiles;
  }

}  // namespace clue
