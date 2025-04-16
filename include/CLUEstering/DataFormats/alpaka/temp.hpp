
#pragma once

namespace clue {

  template <uint8_t Ndim>
  template <typename TQueue>
  void buildTiles(TQueue& queue, uint32_t n_points, int32_t n_tiles) {
	TilesAlpaka<Ndim> tiles(queue, n_points, ntiles);

    // check if tiles are large enough for current data
    if (!(alpaka::trait::GetExtents<clue::device_buffer<Device, uint32_t[]>>{}(
              d_tiles->indexes())[0u] >= npoints) or
        !(alpaka::trait::GetExtents<clue::device_buffer<Device, uint32_t[]>>{}(
              d_tiles->offsets())[0u] >= static_cast<uint32_t>(nTiles))) {
      d_tiles->initialize(npoints, nTiles, nPerDim, queue);
    } else {
      d_tiles->reset(npoints, nTiles, nPerDim, queue);
    }

    auto min_max = clue::make_host_buffer<CoordinateExtremes<Ndim>>(queue);
    auto tile_sizes = clue::make_host_buffer<float[Ndim]>(queue);

    const auto device = alpaka::getDev(queue);
    alpaka::memcpy(queue, d_tiles->minMax(), min_max);
    alpaka::memcpy(queue, d_tiles->tileSize(), tile_sizes);
    alpaka::memcpy(
        queue, d_tiles->wrapped(), clue::make_host_view(h_points.wrapped().data(), Ndim));
    alpaka::wait(queue);
  }

}  // namespace clue
