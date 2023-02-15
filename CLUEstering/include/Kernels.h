#ifndef kernels_h
#define kernels_h

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

using KernelType = std::function<float(float, int, int)>;

// Define the kernels used for the calculation of local density
class kernel {
protected:
  KernelType m_kernel_func;

public:
  kernel(KernelType kernel_function)
      : m_kernel_func{std::move(kernel_function)} {}

  kernel(float flat) {
    m_kernel_func = [=](float dist_ij, int point_id, int j) {
      if (point_id == j) {
        return 1.f;
      } else {
        return flat;
      }
    };
  }

  kernel(float gaus_avg, float gaus_std, float gaus_amplitude) {
    m_kernel_func = [=](float dist_ij, int point_id, int j) {
      return static_cast<float>(gaus_amplitude *
                                std::exp(-std::pow(dist_ij- gaus_avg, 2) /
                                         (2 * std::pow(gaus_std, 2))));
    };
  }

  kernel(float exp_avg, float exp_amplitude) {
    m_kernel_func = [=](float dist_ij, int point_id, int j) {
      return static_cast<float>(exp_amplitude * exp(-exp_avg * dist_ij));
    };
  }

  auto const operator()(float dist_ij, int point_id, int j) {
    return m_kernel_func(dist_ij, point_id, j);
  }
};

#endif
