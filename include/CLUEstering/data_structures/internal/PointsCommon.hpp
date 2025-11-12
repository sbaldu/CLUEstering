
#pragma once

#include "CLUEstering/internal/alpaka/memory.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/detail/make_array.hpp"
#include "CLUEstering/internal/meta/apply.hpp"
#include <span>

namespace clue {

  namespace internal {

    template <typename TPoints>
    struct points_interface {
      ALPAKA_FN_HOST int32_t size() const { return static_cast<const TPoints*>(this)->m_size; }

      ALPAKA_FN_HOST auto coords(std::size_t dim) const {
        if (dim >= TPoints::Ndim_) {
          throw std::out_of_range("Dimension out of range in call to coords.");
        }
        auto& view = static_cast<const TPoints*>(this)->m_view;
        return std::span<const float>(view.m_coords[dim], view.m_size);
      }
      ALPAKA_FN_HOST auto coords(std::size_t dim) {
        if (dim >= TPoints::Ndim_) {
          throw std::out_of_range("Dimension out of range in call to coords.");
        }
        auto& view = static_cast<TPoints*>(this)->m_view;
        return std::span<float>(view.m_coords[dim], view.m_size);
      }

      ALPAKA_FN_HOST auto weights() const {
        auto& view = static_cast<const TPoints*>(this)->m_view;
        return std::span<const float>(view.m_weight, view.m_size);
      }
      ALPAKA_FN_HOST auto weights() {
        auto& view = static_cast<TPoints*>(this)->m_view;
        return std::span<float>(view.m_weight, view.m_size);
      }

      ALPAKA_FN_HOST auto clusterIndexes() const {
        assert(static_cast<const TPoints&>(*this).m_clustered &&
               "The points have not been clustered yet, so the cluster indexes cannot be accessed");
        auto& view = static_cast<const TPoints*>(this)->m_view;
        return std::span<const int>(view.m_cluster_index, view.m_size);
      }
      ALPAKA_FN_HOST auto clusterIndexes() {
        assert(static_cast<const TPoints&>(*this).m_clustered &&
               "The points have not been clustered yet, so the cluster indexes cannot be accessed");
        auto& view = static_cast<TPoints*>(this)->m_view;
        return std::span<int>(view.m_cluster_index, view.m_size);
      }

      ALPAKA_FN_HOST auto clustered() const {
        return static_cast<const TPoints&>(*this).m_clustered;
      }

      ALPAKA_FN_HOST const auto& view() const { return static_cast<const TPoints*>(this)->m_view; }
      ALPAKA_FN_HOST auto& view() { return static_cast<TPoints*>(this)->m_view; }
    };

  }  // namespace internal

  template <std::size_t Ndim>
  struct PointsView {
    std::array<float*, Ndim> m_coords;
    float* m_weight;
    int* m_cluster_index;
    int* m_is_seed;
    float* m_rho;
    int* m_nearest_higher;
    int32_t m_size;

    ALPAKA_FN_HOST_ACC auto operator[](int i) const {
      if (i == -1)
        return clue::nostd::make_array<float, Ndim>(std::numeric_limits<float>::max());

      std::array<float, Ndim> point;
      meta::apply<Ndim>([&]<std::size_t Dim>() { point[Dim] = m_coords[Dim][i]; });
      return point;
    }

    ALPAKA_FN_HOST_ACC auto coords() const {
      std::array<std::span<const float>, Ndim> ptrs;
      [&]<std::size_t... Dims>(std::index_sequence<Dims...>) -> void {
        ((ptrs[Dims] = std::span<const float>{m_coords[Dims], static_cast<std::size_t>(m_size)}),
         ...);
      }(std::make_index_sequence<Ndim>{});
      return ptrs;
    }
    ALPAKA_FN_HOST_ACC inline auto coords() {
      std::array<std::span<float>, Ndim> ptrs;
      [&]<std::size_t... Dims>(std::index_sequence<Dims...>) -> void {
        ((ptrs[Dims] = std::span<float>{m_coords[Dims], static_cast<std::size_t>(m_size)}), ...);
      }(std::make_index_sequence<Ndim>{});
      return ptrs;
    }

    ALPAKA_FN_HOST_ACC auto weight() const {
      return std::span<const float>{m_weight, static_cast<std::size_t>(m_size)};
    }
    ALPAKA_FN_HOST_ACC auto weight() {
      return std::span<float>{m_weight, static_cast<std::size_t>(m_size)};
    }

    ALPAKA_FN_HOST_ACC auto cluster_index() const {
      return std::span<const int>{m_cluster_index, static_cast<std::size_t>(m_size)};
    }
    ALPAKA_FN_HOST_ACC auto cluster_index() {
      return std::span<int>{m_cluster_index, static_cast<std::size_t>(m_size)};
    }

    ALPAKA_FN_HOST_ACC auto is_seed() const {
      return std::span<const int>{m_is_seed, static_cast<std::size_t>(m_size)};
    }
    ALPAKA_FN_HOST_ACC auto is_seed() {
      return std::span<int>{m_is_seed, static_cast<std::size_t>(m_size)};
    }

    ALPAKA_FN_HOST_ACC auto rho() const {
      return std::span<const float>{m_rho, static_cast<std::size_t>(m_size)};
    }
    ALPAKA_FN_HOST_ACC auto rho() {
      return std::span<float>{m_rho, static_cast<std::size_t>(m_size)};
    }

    ALPAKA_FN_HOST_ACC auto nearest_higher() const {
      return std::span<const int>{m_nearest_higher, static_cast<std::size_t>(m_size)};
    }
    ALPAKA_FN_HOST_ACC auto nearest_higher() {
      return std::span<int>{m_nearest_higher, static_cast<std::size_t>(m_size)};
    }
  };

  namespace concepts {

    template <typename T>
    concept contiguous_raw_data = std::is_array_v<T> || std::is_pointer_v<T>;

  }  // namespace concepts

  // TODO: implement for better cache use
  template <std::size_t Ndim>
  int32_t computeAlignSoASize(int32_t n_points);

  template <std::size_t Ndim>
  class PointsHost;
  template <std::size_t Ndim, concepts::device TDev>
  class PointsDevice;

  template <concepts::queue TQueue, std::size_t Ndim, concepts::device TDev>
  void copyToHost(TQueue& queue,
                  PointsHost<Ndim>& h_points,
                  const PointsDevice<Ndim, TDev>& d_points) {
    alpaka::memcpy(
        queue,
        make_host_view(h_points.m_view.cluster_index, h_points.size()),
        make_device_view(alpaka::getDev(queue), d_points.m_view.cluster_index, h_points.size()));
    h_points.mark_clustered();
  }
  template <concepts::queue TQueue, std::size_t Ndim, concepts::device TDev>
  void copyToDevice(TQueue& queue,
                    PointsDevice<Ndim, TDev>& d_points,
                    const PointsHost<Ndim>& h_points) {
    // TODO: copy each coordinate column separately
    alpaka::memcpy(
        queue,
        make_device_view(alpaka::getDev(queue), d_points.m_view.coords[0], Ndim * h_points.size()),
        make_host_view(h_points.m_view.coords[0], Ndim * h_points.size()));
    alpaka::memcpy(queue,
                   make_device_view(alpaka::getDev(queue), d_points.m_view.weight, h_points.size()),
                   make_host_view(h_points.m_view.weight, h_points.size()));
  }

}  // namespace clue
