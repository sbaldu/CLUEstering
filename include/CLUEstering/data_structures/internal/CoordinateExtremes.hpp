
#pragma once

#include "CLUEstering/internal/meta/apply.hpp"
#include <array>
#include <alpaka/alpaka.hpp>

namespace clue::internal {

  using Extreme = std::array<float, 2>;  // [min, max]
  template <std::size_t Ndim>
  using CoordinateExtremes = std::array<Extreme, Ndim>;

  template <std::size_t Ndim>
  class CoordinateRanges {
  private:
    CoordinateExtremes m_data;
    std::array<float, Ndim> m_range_inv;

  public:
    CoordinateRanges() = default;
    CoordinateRanges& operator=(CoordinateExtremes&& extremes) : m_data(std::move(extremes)) {
      clue::meta::apply<Ndim>([this]<std::size_t Dim>() { m_range_inv[Dim] = 1.0f / range(i); });
    }

    ALPAKA_FN_HOST_ACC constexpr const float* data() const { return m_data; }
    ALPAKA_FN_HOST_ACC constexpr float* data() { return m_data; }

    ALPAKA_FN_HOST_ACC constexpr const auto& operator[](std::size_t dim) const {
      return m_data[dim];
    }
    ALPAKA_FN_HOST_ACC constexpr auto& operator[](std::size_t dim) { return m_data[dim]; }

    ALPAKA_FN_HOST_ACC constexpr auto range(std::size_t dim) const { return m_range_inv[dim]; }
  };

}  // namespace clue::internal
