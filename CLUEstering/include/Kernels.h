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

// Define the kernels used for the calculation of local density
/* class kernel{ */
/*    protected: */
/* 	  // Parameters of gaussian function */
/* 	  float m_gaus_avg; */
/* 	  float m_gaus_std; */
/* 	  float m_gaus_amplitude; */
/* 	  // Parameters of exponential function */
/* 	  float m_exp_avg; */
/* 	  float m_exp_amplitude; */
/* 	  // Parameter of flat function */
/* 	  float m_flat; */
/* 	  // Patameters of the polynomial */
/* 	  const uint8_t m_pol_grade; */
/* 	  std::vector<float> m_pol_coefficients; */
/*    public: */
/* 	  kernel(float gaus_avg, */
/* 			float gaus_std, */
/* 			float gaus_amplitude, */
/* 			float exp_avg, */
/* 			float exp_amplitude, */
/* 			const uint8_t pol_grade, */
/* 			std::vector<float> pol_coefficients) */
/* 		 : m_gaus_avg(gaus_avg), */
/* 		 m_gaus_std(gaus_std), */
/* 		 m_gaus_amplitude(gaus_amplitude), */
/* 		 m_exp_avg(exp_avg), */
/* 		 m_exp_amplitude(exp_amplitude), */
/* 		 m_pol_grade(pol_grade) { */
/* 			m_pol_coefficients = std::move(pol_coefficients); */
/* 		 } */

/* 	  virtual float applyKernel(float dist_ij, int point_id, int j) { */
/* 		 if(point_id == j) { */
/* 			return 1.f; */
/* 		 } else { */
/* 			float flat_component { m_flat }; */
/* 			double gaus_component { m_gaus_amplitude*exp(-pow(dist_ij -
 * m_gaus_avg,2)/(2*pow(m_gaus_std,2))) }; */
/* 			double exp_component {
 * m_exp_amplitude*exp(-m_exp_avg*dist_ij) }; */
/* double pol_component {}; */
/* for(int i {1}; i < m_pol_grade + 1; ++i) { */
/*    pol_component += m_pol_coefficients[i-1]*std::pow(dist_ij,i); */
/* } */

/* return flat_component + gaus_component + exp_component + pol_component; */
/* } */
/* } */
/* }; */
/* class gaussianKernel : public kernel{ */
/* public: */
/* gaussianKernel(float gaus_avg, float gaus_std, float gaus_amplitude) */
/* : kernel(gaus_avg, gaus_std, gaus_amplitude, 0., 0., 0, {}) {} */
/* }; */
/* class exponentialKernel : public kernel{}; */
/* class flatKernel : public kernel{}; */

#endif
