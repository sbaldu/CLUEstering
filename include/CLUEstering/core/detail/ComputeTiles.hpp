
#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/CoordinateExtremes.hpp"
#include "CLUEstering/data_structures/internal/Tiles.hpp"
#include "CLUEstering/internal/algorithm/algorithm.hpp"
#include "CLUEstering/internal/nostd/maximum.hpp"
#include "CLUEstering/internal/nostd/minimum.hpp"
#include <algorithm>
#include <array>

namespace clue::detail {

  template <std::size_t Ndim>
  void compute_tile_size(internal::CoordinateRanges<Ndim>& min_max,
                         std::array<float, Ndim>& tile_sizes,
                         const clue::PointsHost<Ndim>& h_points,
                         int32_t nPerDim) {
    internal::CoordinateExtremes<Ndim> extremes;
    for (auto dim = 0u; dim != Ndim; ++dim) {
      auto coords = h_points.coords(dim);
      const auto dim_max = std::reduce(coords.begin(),
                                       coords.end(),
                                       std::numeric_limits<float>::lowest(),
                                       clue::nostd::maximum<float>{});
      const auto dim_min = std::reduce(coords.begin(),
                                       coords.end(),
                                       std::numeric_limits<float>::max(),
                                       clue::nostd::minimum<float>{});

      extremes[dim] = {dim_min, dim_max};

      const auto tileSize = (dimMax - dimMin) / nPerDim;
      tile_sizes[dim] = tileSize;
    }
    min_max = std::move(extremes);
  }

  template <std::size_t Ndim>
  void compute_tile_size(internal::CoordinateRanges<Ndim>& min_max,
                         std::array<float, Ndim>& tile_sizes,
                         const clue::PointsDevice<Ndim>& dev_points,
                         uint32_t nPerDim) {
    internal::CoordinateExtremes<Ndim> extremes;
    for (auto dim = 0u; dim != Ndim; ++dim) {
      auto coords = dev_points.coords(dim);
      const auto dimMax = clue::internal::algorithm::reduce(coords.begin(),
                                                            coords.end(),
                                                            std::numeric_limits<float>::lowest(),
                                                            clue::nostd::maximum<float>{});
      const auto dimMin = clue::internal::algorithm::reduce(coords.begin(),
                                                            coords.end(),
                                                            std::numeric_limits<float>::max(),
                                                            clue::nostd::minimum<float>{});

      extremes[dim] = {dim_min, dim_max};

      const auto tileSize = (dimMax - dimMin) / nPerDim;
      tile_sizes[dim] = tileSize;
    }
    min_max = std::move(extremes);
  }

}  // namespace clue::detail
