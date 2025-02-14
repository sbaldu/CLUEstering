
#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/alpaka.hpp>

namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE {

  ALPAKA_FN_HOST_ACC float FlatKernel::operator()(const TAcc&,
                                                  float /*dist_ij*/,
                                                  int point_id,
                                                  int j) const {
    if (point_id == j) {
      return 1.f;
    } else {
      return m_flat;
    }
  }

  ALPAKA_FN_HOST_ACC float GaussianKernel::operator()(const Acc1D& acc,
                                                      float dist_ij,
                                                      int point_id,
                                                      int j) const {
    if (point_id == j) {
      return 1.f;
    } else {
      return (m_gaus_amplitude *
              alpaka::math::exp(acc,
                                -(dist_ij - m_gaus_avg) * (dist_ij - m_gaus_avg) /
                                    (2 * m_gaus_std * m_gaus_std)));
    }
  }

  ALPAKA_FN_HOST_ACC float ExponentialKernel::operator()(const Acc1D& acc,
                                                         float dist_ij,
                                                         int point_id,
                                                         int j) const {
    if (point_id == j) {
      return 1.f;
    } else {
      return (m_exp_amplitude * alpaka::math::exp(acc, -m_exp_avg * dist_ij));
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE
