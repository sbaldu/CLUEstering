#ifndef kernels_h
#define kernels_h

#include <iostream>
#include <cmath>

// Define the kernels used for the calculation of local density
class kernels{
	public:
		float flat(float dist_ij, int point_id, int j) {
			if(point_id == j) {
				return 1.f;
			} else {
				return 0.5f;
			}
		}
		float gaussian(float dist_ij) {}
		float exponential(float dist_ij) {}
};

#endif
