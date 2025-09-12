
all:
	g++ -std=c++20 header.cpp -fPIC -shared -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED=ON -o libserial.so
	nvcc -x cu -std=c++20 header.cpp -Xcompiler -fPIC -shared -DALPAKA_ACC_GPU_CUDA_ENABLED=ON -o libcuda.so
