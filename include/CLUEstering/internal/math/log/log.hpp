
#pragma once

#include "CLUEstering/internal/math/defines.hpp"
#include <concepts>
#include <alpaka/alpaka.hpp>

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED) && \
    !defined(ALPAKA_ACC_SYCL_ENABLED)
#include <cmath>
#endif

namespace clue::math {

  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline float log(float x) {
#if defined(CUDA_DEVICE_FN)
    return ::log(x);
#elif defined(HIP_DEVICE_FN)
    return ::log(x);
#elif defined(SYCL_DEVICE_FN)
    return sycl::log(x);
#else
    return std::log(x);
#endif
  }

  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline double log(double x) {
#if defined(CUDA_DEVICE_FN)
    return ::log(x);
#elif defined(HIP_DEVICE_FN)
    return ::log(x);
#elif defined(SYCL_DEVICE_FN)
    return sycl::log(x);
#else
    return std::log(x);
#endif
  }

  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline float logf(float x) { return log(x); }

  template <std::integral T>
  ALPAKA_FN_ACC MATH_FN_CONSTEXPR inline double log(T x) {
    return log(static_cast<double>(x));
  }

}  // namespace clue::math
