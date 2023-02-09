#ifndef kernels_h
#define kernels_h

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

class kernel {
protected:
  float m_dist_ij;
  int m_point_id;
  int m_j;

public:
  kernel(float dist_ij, int point_id, int j)
      : m_dist_ij(dist_ij), m_point_id(point_id), m_j(j) {}

  virtual float applyKernel(std::function<float(float, int, int)> func) {
    return func(m_dist_ij, m_point_id, m_j);
  }
};

class flatKernel : public kernel {
private:
  float m_flat;

public:
  flatKernel(float dist_ij, int point_id, int j, float flat)
      : kernel(dist_ij, point_id, j), m_flat(flat) {}
  float applyKernel() {
    return kernel::applyKernel([=, this](float distij_, int id_, int j_) {
      if (id_ == j_) {
        return 1.f;
      } else {
        return m_flat;
      }
    });
  }
};

class gaussianKernel : public kernel {
private:
  float m_gaus_avg;
  float m_gaus_std;
  float m_gaus_amplitude;

public:
  gaussianKernel(float dist_ij, int point_id, int j, float gaus_avg,
                 float gaus_std, float gaus_amplitude)
      : kernel(dist_ij, point_id, j), m_gaus_avg(gaus_avg),
        m_gaus_std(gaus_std), m_gaus_amplitude(gaus_amplitude) {}

  float applyKernel() {
    return kernel::applyKernel([=, this](float distij_, int id_, int j_) {
      if (id_ == j_) {
        return 1.f;
      } else {
        return static_cast<float>(
            m_gaus_amplitude *
            exp(-pow(distij_ - m_gaus_avg, 2) / (2 * pow(m_gaus_std, 2))));
      }
    });
  }
};

class exponentialKernel : public kernel {
private:
  float m_exp_avg;
  float m_exp_amplitude;

public:
  exponentialKernel(float dist_ij, int point_id, int j, float exp_avg,
                    float exp_amplitude)
      : kernel(dist_ij, point_id, j), m_exp_avg(exp_avg),
        m_exp_amplitude(exp_amplitude) {}

  float applyKernel() {
    return kernel::applyKernel([=, this](float distij_, int id_, int j_) {
      if (id_ == j_) {
        return 1.f;
      } else {
        return static_cast<float>(m_exp_amplitude * exp(-m_exp_avg * distij_));
      }
    });
  }
};

#endif
