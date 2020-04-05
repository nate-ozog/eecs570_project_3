CC        = g++
CFLAGS    = -std=c++11
NVCC      = nvcc
CUDA_INC  = -I/usr/local/cuda/include/
CUDA_LINK = -lcuda
BASE_MAIN = nw.cpp
BASE_DEPS = nw.cpp nw_general.h
GPU_SRC = nw.cu testbatch.cpp
GPU_HDR = nw.cu nw_general.h xs.cuh xs_core.cuh testbatch.hpp cuda_error_check.cuh

all: gpu_nw base_nw gpu_nw_debug

gpu_nw: $(GPU_SRC) $(GPU_HDR)
	$(NVCC) $(CFLAGS) $(GPU_SRC) -o $@.o $(CUDA_LINK)

base_nw: $(BASE_DEPS)
	$(CC) $(CFLAGS) $(BASE_MAIN) -o $@.o $(CUDA_INC)

clean:
	rm -rf *.o