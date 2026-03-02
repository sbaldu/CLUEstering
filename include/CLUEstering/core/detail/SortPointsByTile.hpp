
#pragma once

#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/TilesView.hpp"
#include "CLUEstering/detail/concepts.hpp"

#include <alpaka/alpaka.hpp>
#include <cstddef>
#include <cstdint>

namespace clue::detail {


  template <clue::concepts::queue Queue, std::size_t Ndim, typename value_type>
  void sortPointsByTile(Queue& queue,
                        clue::PointsDevice<Ndim, value_type>& sorted_points,
                        clue::internal::TilesView<Ndim, value_type> tiles_view,
                        clue::PointsDevice<Ndim, value_type> dev_points) {
  }

}  // namespace clue::detail
