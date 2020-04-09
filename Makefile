CC        = g++
CFLAGS    = -std=c++11
NVCC      = nvcc
CUDA_INC  = -I/usr/local/cuda/include/
CUDA_LINK = -lcuda
CPU_SRC = nw.cpp testbatch.cpp
CPU_HDR = testbatch.hpp
GPU_SRC = needletail.cu testbatch.cpp
GPU_HDR = testbatch.hpp needletail_general.hpp needletail.cuh needletail_core.cuh cuda_error_check.cuh

gpu_nw: $(GPU_SRC) $(GPU_HDR)
  $(NVCC) $(CFLAGS) $(GPU_SRC) -o $@.o $(CUDA_LINK)

base_nw: $(CPU_SRC) $(CPU_HDR)
  $(CC) $(CFLAGS) $(CPU_SRC) -o $@.o $(CUDA_INC)

clean:
  rm -rf *.o