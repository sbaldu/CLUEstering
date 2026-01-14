

#pragma once

#include "CLUEstering/core/DistanceMetrics.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace clue {

  struct DistanceMetricTag {
  	virtual ~DistanceMetricTag() = default;
};

  class EuclideanMetricTag final : public DistanceMetricTag {
  public:
    EuclideanMetricTag() = default;

    template <std::size_t Ndim>
    operator clue::EuclideanMetric<Ndim>() const {
      return clue::EuclideanMetric<Ndim>{};
    }
  };

  class WeightedEuclideanTag final : public DistanceMetricTag {
  private:
    std::vector<float> m_weights;

  public:
    WeightedEuclideanTag(std::vector<float> weights) : m_weights(std::move(weights)) {}

    template <std::size_t Ndim>
    operator clue::WeightedEuclideanMetric<Ndim>() const {
      if (m_weights.size() != Ndim) {
        throw std::runtime_error("Weights size does not match the number of dimensions.");
      }
      std::array<float, Ndim> weights_array;
      std::copy_n(m_weights.begin(), Ndim, weights_array.begin());
      return clue::WeightedEuclideanMetric<Ndim>{std::move(weights_array)};
    }
  };

  class PeriodicEuclideanTag final : public DistanceMetricTag {
  private:
    std::vector<float> m_periods;

  public:
    PeriodicEuclideanTag(std::vector<float> periods) : m_periods(std::move(periods)) {}

    template <std::size_t Ndim>
    operator clue::PeriodicEuclideanMetric<Ndim>() const {
      if (m_periods.size() != Ndim) {
        throw std::runtime_error("Periods size does not match the number of dimensions.");
      }
      std::array<float, Ndim> periods_array;
      std::copy_n(m_periods.begin(), Ndim, periods_array.begin());
      return clue::PeriodicEuclideanMetric<Ndim>{std::move(periods_array)};
    }
  };

  class ManhattanTag final : public DistanceMetricTag {
  public:
    ManhattanTag() = default;

    template <std::size_t Ndim>
    operator clue::ManhattanMetric<Ndim>() const {
      return clue::ManhattanMetric<Ndim>{};
    }
  };

  class ChebyshevTag : public DistanceMetricTag {
  public:
    ChebyshevTag() = default;

    template <std::size_t Ndim>
    operator clue::ChebyshevMetric<Ndim>() const {
      return clue::ChebyshevMetric<Ndim>{};
    }
  };

  class WeightedChebyshevTag final : public DistanceMetricTag {
  private:
    std::vector<float> m_weights;

  public:
    WeightedChebyshevTag(std::vector<float> weights) : m_weights(std::move(weights)) {}

    template <std::size_t Ndim>
    operator clue::WeightedChebyshevMetric<Ndim>() const {
      if (m_weights.size() != Ndim) {
        throw std::runtime_error("Weights size does not match the number of dimensions.");
      }
      std::array<float, Ndim> weights_array;
      std::copy_n(m_weights.begin(), Ndim, weights_array.begin());
      return clue::WeightedChebyshevMetric<Ndim>{std::move(weights_array)};
    }
  };

  namespace internal {

    template <typename MetricTag>
    struct TagToMetric {
      using type = void;
    };

    template <>
    struct TagToMetric<EuclideanMetricTag> {
      template <std::size_t Ndim>
      using type = clue::EuclideanMetric<Ndim>;
    };

    template <>
    struct TagToMetric<WeightedEuclideanTag> {
      template <std::size_t Ndim>
      using type = clue::WeightedEuclideanMetric<Ndim>;
    };

    template <>
    struct TagToMetric<PeriodicEuclideanTag> {
      template <std::size_t Ndim>
      using type = clue::PeriodicEuclideanMetric<Ndim>;
    };

    template <>
    struct TagToMetric<ManhattanTag> {
      template <std::size_t Ndim>
      using type = clue::ManhattanMetric<Ndim>;
    };

    template <>
    struct TagToMetric<ChebyshevTag> {
      template <std::size_t Ndim>
      using type = clue::ChebyshevMetric<Ndim>;
    };
    template <>
    struct TagToMetric<WeightedChebyshevTag> {
      template <std::size_t Ndim>
      using type = clue::WeightedChebyshevMetric<Ndim>;
    };

  }  // namespace internal

}  // namespace clue
