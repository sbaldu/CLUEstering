
#pragma once
#include <alpaka/alpaka.hpp>

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#define backend serial
#elif ALPAKA_ACC_GPU_CUDA_ENABLED
#define backend cuda
#endif
