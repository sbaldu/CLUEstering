#ifndef kernels_h
#define kernels_h

#include <iostream>
#include <cmath>

// Define the kernels used for the calculation of local density
class kernel{
	private:
		// Parameters of gaussian function
		float m_gaus_avg;
		float m_gaus_std;
		float m_gaus_amplitude;
		// Parameters of exponential function
		float m_exp_amplitude;
		float m_exp_avg;
		// Parameter of flat function
		float m_flat;
	public:
		kernel(float avg, float std, float amplitude) 
			: m_gaus_avg(avg), m_gaus_std(std), m_exp_amplitude(amplitude) {}
		kernel(float avg, float amplitude) : m_exp_avg(avg), m_exp_amplitude(amplitude) {}
		kernel(float flat) : m_flat(flat) {}

		float flat(float dist_ij, int point_id, int j) {
			if(point_id == j) {
				return 1.f;
			} else {
				return m_flat;
			}
		}
		float gaussian(float dist_ij, int point_id, int j) {
			if(point_id == j) {
				return 1.f;
			} else {
				return m_exp_amplitude*exp(-m_exp_avg*dist_ij);
			}
		}
		float exponential(float dist_ij, int point_id, int j) {
			if(point_id == j) {
				return 1.f;
			} else {
				return m_gaus_amplitude*exp(-pow(dist_ij - m_gaus_avg,2)/(2*pow(m_gaus_std,2)));
			}
		}
};

#endif
