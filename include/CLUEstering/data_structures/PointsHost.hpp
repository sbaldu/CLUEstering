
#pragma once

#include "CLUEstering/data_structures/Common.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"

#include <alpaka/alpaka.hpp>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>

namespace clue {

  namespace concepts = detail::concepts;

  namespace soa::host {

    // No need to allocate temporary buffers on the host
    template <uint8_t Ndim>
    int32_t computeSoASize(int32_t n_points) {
      return ((Ndim + 1) * sizeof(float) + 2 * sizeof(int)) * n_points;
    }

    template <uint8_t Ndim>
    void partitionSoAView(PointsView* view, std::byte* buffer, int32_t n_points) {
      view->coords = reinterpret_cast<float*>(buffer);
      view->weight = reinterpret_cast<float*>(buffer + Ndim * n_points * sizeof(float));
      view->cluster_index = reinterpret_cast<int*>(buffer + (Ndim + 1) * n_points * sizeof(float));
      view->is_seed = reinterpret_cast<int*>(buffer + (Ndim + 2) * n_points * sizeof(float));
      view->n = n_points;
    }
    template <uint8_t Ndim, concepts::contiguous_raw_data... TBuffers>
      requires(sizeof...(TBuffers) == 4)
    void partitionSoAView(PointsView* view, int32_t n_points, TBuffers... buffer) {
      auto buffers_tuple = std::make_tuple(buffer...);
      // TODO: is reinterpret_cast necessary?
      view->coords = reinterpret_cast<float*>(std::get<0>(buffers_tuple));
      view->weight = reinterpret_cast<float*>(std::get<1>(buffers_tuple));
      view->cluster_index = reinterpret_cast<int*>(std::get<2>(buffers_tuple));
      view->is_seed = reinterpret_cast<int*>(std::get<3>(buffers_tuple));
      view->n = n_points;
    }
    template <uint8_t Ndim, concepts::contiguous_raw_data... TBuffers>
      requires(sizeof...(TBuffers) == 2)
    void partitionSoAView(PointsView* view, int32_t n_points, TBuffers... buffers) {
      auto buffers_tuple = std::make_tuple(buffers...);

      // TODO: is reinterpret_cast necessary?
      view->coords = reinterpret_cast<float*>(std::get<0>(buffers_tuple));
      view->weight = reinterpret_cast<float*>(std::get<0>(buffers_tuple) + Ndim * n_points);
      view->cluster_index = reinterpret_cast<int*>(std::get<1>(buffers_tuple));
      view->is_seed = reinterpret_cast<int*>(std::get<1>(buffers_tuple) + n_points);
      view->n = n_points;
    }
    template <uint8_t Ndim, std::ranges::contiguous_range... TBuffers>
      requires(sizeof...(TBuffers) == 4)
    void partitionSoAView(PointsView* view, int32_t n_points, TBuffers&&... buffers) {
      auto buffers_tuple = std::forward_as_tuple(std::forward<TBuffers>(buffers)...);
      // TODO: is reinterpret_cast necessary?
      view->coords = reinterpret_cast<float*>(std::get<0>(buffers_tuple).data());
      view->weight = reinterpret_cast<float*>(std::get<1>(buffers_tuple).data());
      view->cluster_index = reinterpret_cast<int*>(std::get<2>(buffers_tuple).data());
      view->is_seed = reinterpret_cast<int*>(std::get<3>(buffers_tuple).data());
      view->n = n_points;
    }
    template <uint8_t Ndim, std::ranges::contiguous_range... TBuffers>
      requires(sizeof...(TBuffers) == 2)
    void partitionSoAView(PointsView* view, int32_t n_points, TBuffers&&... buffers) {
      auto buffers_tuple = std::forward_as_tuple(std::forward<TBuffers>(buffers)...);
      // TODO: is reinterpret_cast necessary?
      view->coords = reinterpret_cast<float*>(std::get<0>(buffers_tuple).data());
      view->weight = reinterpret_cast<float*>(std::get<0>(buffers_tuple).data() + Ndim * n_points);
      view->cluster_index = reinterpret_cast<int*>(std::get<1>(buffers_tuple).data());
      view->is_seed = reinterpret_cast<int*>(std::get<1>(buffers_tuple).data() + n_points);
      view->n = n_points;
    }

  }  // namespace soa::host

  template <uint8_t Ndim>
  class PointsHost {
  public:
    template <concepts::queue TQueue>
    PointsHost(TQueue& queue, int32_t n_points)
        : m_buffer{make_host_buffer<std::byte[]>(queue, soa::host::computeSoASize<Ndim>(n_points))},
          m_view{make_host_buffer<PointsView>(queue)},
          m_size{n_points} {
      soa::host::partitionSoAView<Ndim>(m_view.data(), m_buffer->data(), n_points);
    }

    template <concepts::queue TQueue>
    PointsHost(TQueue& queue, int32_t n_points, std::span<std::byte> buffer)
        : m_view{make_host_buffer<PointsView>(queue)}, m_size{n_points} {
      assert(buffer.size() == soa::host::computeSoASize<Ndim>(n_points));

      soa::host::partitionSoAView<Ndim>(m_view.data(), buffer.data(), n_points);
    }

    template <concepts::queue TQueue, std::ranges::contiguous_range... TBuffers>
      requires(sizeof...(TBuffers) == 2 || sizeof...(TBuffers) == 4)
    PointsHost(TQueue& queue, int32_t n_points, TBuffers&&... buffers)
        : m_view{make_host_buffer<PointsView>(queue)}, m_size{n_points} {
      soa::host::partitionSoAView<Ndim>(
          m_view.data(), n_points, std::forward<TBuffers>(buffers)...);
    }

    template <concepts::queue TQueue, concepts::contiguous_raw_data... TBuffers>
      requires(sizeof...(TBuffers) == 2 || sizeof...(TBuffers) == 4)
    PointsHost(TQueue& queue, int32_t n_points, TBuffers... buffers)
        : m_view{make_host_buffer<PointsView>(queue)}, m_size{n_points} {
      soa::host::partitionSoAView<Ndim>(m_view.data(), n_points, buffers...);
    }

    PointsHost(const PointsHost&) = delete;
    PointsHost& operator=(const PointsHost&) = delete;
    PointsHost(PointsHost&&) = default;
    PointsHost& operator=(PointsHost&&) = default;
    ~PointsHost() = default;

    ALPAKA_FN_HOST int32_t size() const { return m_size; }

    ALPAKA_FN_HOST std::span<const float> coords() const {
      return std::span<const float>(m_view->coords, static_cast<std::size_t>(m_view->n * Ndim));
    }
    ALPAKA_FN_HOST std::span<float> coords() {
      return std::span<float>(m_view->coords, static_cast<std::size_t>(m_view->n * Ndim));
    }

    ALPAKA_FN_HOST std::span<const float> coords(size_t dim) const {
      return std::span<const float>(m_view->coords + dim * m_view->n,
                                    static_cast<std::size_t>(m_view->n));
    }
    ALPAKA_FN_HOST std::span<float> coords(size_t dim) {
      return std::span<float>(m_view->coords + dim * m_view->n,
                              static_cast<std::size_t>(m_view->n));
    }

    ALPAKA_FN_HOST std::span<const float> weights() const {
      return std::span<const float>(m_view->weight, static_cast<std::size_t>(m_view->n));
    }
    ALPAKA_FN_HOST std::span<float> weights() {
      return std::span<float>(m_view->weight, static_cast<std::size_t>(m_view->n));
    }

    ALPAKA_FN_HOST std::span<const int> clusterIndexes() const {
      return std::span<const int>(m_view->cluster_index, static_cast<std::size_t>(m_view->n));
    }
    ALPAKA_FN_HOST std::span<int> clusterIndexes() {
      return std::span<int>(m_view->cluster_index, static_cast<std::size_t>(m_view->n));
    }

    ALPAKA_FN_HOST std::span<const int> isSeed() const {
      return std::span<const int>(m_view->is_seed, static_cast<std::size_t>(m_view->n));
    }
    ALPAKA_FN_HOST std::span<int> isSeed() {
      return std::span<int>(m_view->is_seed, static_cast<std::size_t>(m_view->n));
    }

    ALPAKA_FN_HOST PointsView* view() { return m_view.data(); }
    ALPAKA_FN_HOST const PointsView* view() const { return m_view.data(); }

    template <concepts::queue _TQueue, uint8_t _Ndim, concepts::device _TDev>
    friend void copyToHost(_TQueue& queue,
                           PointsHost<_Ndim>& h_points,
                           const PointsDevice<_Ndim, _TDev>& d_points);
    template <concepts::queue _TQueue, uint8_t _Ndim, concepts::device _TDev>
    friend void copyToDevice(_TQueue& queue,
                             PointsDevice<_Ndim, _TDev>& d_points,
                             const PointsHost<_Ndim>& h_points);

  private:
    std::optional<host_buffer<std::byte[]>> m_buffer;
    host_buffer<PointsView> m_view;
    int32_t m_size;
  };

}  // namespace clue
