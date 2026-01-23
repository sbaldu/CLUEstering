
#pragma once

#include "CLUEstering/internal/alpaka/memory.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/detail/make_array.hpp"
#include "CLUEstering/internal/meta/apply.hpp"
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
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
        auto& const_view = static_cast<const TPoints*>(this)->m_const_view;
        return std::span<const float>(const_view.coords[dim], const_view.n);
      }
      ALPAKA_FN_HOST auto coords(std::size_t dim) {
        if (dim >= TPoints::Ndim_) {
          throw std::out_of_range("Dimension out of range in call to coords.");
        }
        auto& view = static_cast<TPoints*>(this)->m_view;
        if (!view.has_value()) {
          throw std::logic_error("The input data passed to the points is read-only.");
        }
        return std::span<float>(view->coords[dim], view->n);
      }

      ALPAKA_FN_HOST auto weights() const {
        auto& const_view = static_cast<const TPoints*>(this)->m_const_view;
        return std::span<const float>(const_view.weight, const_view.n);
      }
      ALPAKA_FN_HOST auto weights() {
        auto& view = static_cast<TPoints*>(this)->m_view;
        return std::span<float>(view.weight, view.n);
      }

      ALPAKA_FN_HOST auto clusterIndexes() const {
        assert(static_cast<const TPoints&>(*this).m_clustered &&
               "The points have not been clustered yet, so the cluster indexes cannot be accessed");
        auto& const_view = static_cast<const TPoints*>(this)->m_const_view;
        return std::span<const int>(const_view.cluster_index, const_view.n);
      }

      ALPAKA_FN_HOST auto clustered() const {
        return static_cast<const TPoints&>(*this).m_clustered;
      }

      ALPAKA_FN_HOST const auto& view() const {
        return static_cast<const TPoints*>(this)->m_const_view;
      }
      ALPAKA_FN_HOST auto& view() { return static_cast<TPoints*>(this)->m_view.value(); }
    };

  }  // namespace internal

  namespace detail {

    template <std::size_t Ndim>
    struct InputView {
      std::array<const float*, Ndim> coords;
      const float* weight;
    };
    template <std::size_t Ndim>
    struct ConstInputView {
      std::array<float*, Ndim> coords;
      float* weight;
    };
    struct OutputView {
      int* cluster_index;
      int* is_seed;
      float* rho;
      int* nearest_higher;
      int32_t n;
    };

  }  // namespace detail

  template <std::size_t Ndim>
  struct ConstPointsView {
    std::array<const float*, Ndim> coords;
    const float* weight;
    int* cluster_index;
    int* is_seed;
    float* rho;
    int* nearest_higher;
    int32_t n;

    ALPAKA_FN_HOST_ACC auto operator[](int i) const {
      if (i == -1)
        return clue::nostd::make_array<float, Ndim + 1>(std::numeric_limits<float>::max());

      std::array<float, Ndim + 1> point;
      meta::apply<Ndim>([&]<std::size_t Dim>() { point[Dim] = coords[Dim][i]; });
      point[Ndim] = weight[i];
      return point;
    }
  };

  template <std::size_t Ndim>
  struct PointsView {
    std::array<float*, Ndim> coords;
    float* weight;
    std::int32_t* cluster_index;
    std::int32_t* is_seed;
    float* rho;
    std::int32_t* nearest_higher;
    std::int32_t n;

    ALPAKA_FN_HOST_ACC auto operator[](int index) const {
      if (index == -1)
        return clue::nostd::make_array<float, Ndim + 1>(std::numeric_limits<float>::max());

      std::array<float, Ndim + 1> point;
      meta::apply<Ndim>([&]<std::size_t Dim>() -> void { point[Dim] = coords[Dim][index]; });
      point[Ndim] = weight[index];
      return point;
    }
  };

  // TODO: implement for better cache use
  template <std::size_t Ndim>
  int32_t computeAlignSoASize(int32_t n_points);

}  // namespace clue
