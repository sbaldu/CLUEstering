
#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/alpaka.hpp>

namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE {

  class FlatKernel {
  private:
    float m_flat;

  public:
    FlatKernel(float flat) : m_flat{flat} {}

    ALPAKA_FN_HOST_ACC float operator()(const Acc1D&,
                                        float /*dist_ij*/,
                                        int point_id,
                                        int j) const;
  };

  class GaussianKernel {
  private:
    float m_gaus_avg;
    float m_gaus_std;
    float m_gaus_amplitude;

  public:
    GaussianKernel(float gaus_avg, float gaus_std, float gaus_amplitude)
        : m_gaus_avg{gaus_avg}, m_gaus_std{gaus_std}, m_gaus_amplitude{gaus_amplitude} {}

    ALPAKA_FN_HOST_ACC float operator()(const Acc1D& acc,
                                        float dist_ij,
                                        int point_id,
                                        int j) const;
  };

  class ExponentialKernel {
  private:
    float m_exp_avg;
    float m_exp_amplitude;

  public:
    ExponentialKernel(float exp_avg, float exp_amplitude)
        : m_exp_avg{exp_avg}, m_exp_amplitude{exp_amplitude} {}

    ALPAKA_FN_HOST_ACC float operator()(const Acc1D& acc,
                                        float dist_ij,
                                        int point_id,
                                        int j) const;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE
