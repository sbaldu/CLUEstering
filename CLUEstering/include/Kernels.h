#ifndef kernels_h
#define kernels_h

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

using KernelType = std::function<float(float, int, int)>;

// Define the kernels used for the calculation of local density
// The base class allows to define a generic kernel
class kernel {
protected:
  KernelType m_kernel_func;

public:
  kernel() = default;
  kernel(KernelType kernel_function)
      : m_kernel_func(kernel_function) {}

  virtual float applyKernel(float dist_ij, int point_id, int j) {
    return m_kernel_func(dist_ij, point_id, j);
  }
};

// Some derived classes are defined to simplify the use of the most common
// kernels
class flatKernel : public kernel {
private:
  float m_flat;

public:
  flatKernel(float flat) : m_flat(flat) {
    m_kernel_func = [=, this](float distij_, int id_, int j_) {
      if (id_ == j_) {
        return 1.f;
      } else {
        return m_flat;
      }
    };
  }
};

class gaussianKernel : public kernel {
private:
  float m_gaus_avg;
  float m_gaus_std;
  float m_gaus_amplitude;

public:
  gaussianKernel(float gaus_avg, float gaus_std, float gaus_amplitude)
      : m_gaus_avg(gaus_avg), m_gaus_std(gaus_std),
        m_gaus_amplitude(gaus_amplitude) {
    m_kernel_func = [=, this](float distij_, int id_, int j_) {
      if (id_ == j_) {
        return 1.f;
      } else {
        return static_cast<float>(
            m_gaus_amplitude *
            exp(-pow(distij_ - m_gaus_avg, 2) / (2 * pow(m_gaus_std, 2))));
      }
    };
  }
};

class exponentialKernel : public kernel {
private:
  float m_exp_avg;
  float m_exp_amplitude;

public:
  exponentialKernel(float exp_avg, float exp_amplitude)
      : m_exp_avg(exp_avg), m_exp_amplitude(exp_amplitude) {
    m_kernel_func = [=, this](float distij_, int id_, int j_) {
      if (id_ == j_) {
        return 1.f;
      } else {
        return static_cast<float>(m_exp_amplitude * exp(-m_exp_avg * distij_));
      }
    };
  }
};

#endif
