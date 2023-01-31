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
		float m_exp_avg;
		float m_exp_amplitude;
		// Parameter of flat function
		float m_flat;

		// Distinguish the type of kernel
		bool m_isGaus = false;
		bool m_isExp = false;
		bool m_isFlat = false;
	public:
		kernel(float avg, float std, float amplitude) 
			: m_gaus_avg(avg), m_gaus_std(std), m_gaus_amplitude(amplitude) {
				m_isGaus = true;
			}
		kernel(float avg, float amplitude) 
			: m_exp_avg(avg), m_exp_amplitude(amplitude) {
				m_isExp = true;
		}
		kernel(float flat) 
			: m_flat(flat) {
				m_isFlat = true;
		}

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
				return m_gaus_amplitude*exp(-pow(dist_ij - m_gaus_avg,2)/(2*pow(m_gaus_std,2)));
			}
		}
		float exponential(float dist_ij, int point_id, int j) {
			if(point_id == j) {
				return 1.f;
			} else {
				return m_exp_amplitude*exp(-m_exp_avg*dist_ij);
			}
		}

		float applyKernel(float dist_ij, int point_id, int j) {
			if(m_isGaus) {
				return gaussian(dist_ij, point_id, j);
			} else if(m_isExp) {
				return exponential(dist_ij, point_id, j);
			} else  { // assume that at least one is going to be true
				return flat(dist_ij, point_id, j);
			} 
		}
};

#endif
