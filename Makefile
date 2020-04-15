CC         = g++
CFLAGS     = -std=c++11
NVCC       = nvcc
NVCC_FLAGS = --default-stream per-thread -arch=sm_60 -rdc=true -O2 -Xptxas -O2 ${CFLAGS}
CUDA_INC   = -I/usr/local/cuda/include/
CUDA_LINK  = -lcuda
CPU_SRC    = nw.cpp testbatch.cpp
CPU_HDR    = testbatch.hpp
GPU_SRC    = needletail.cu poolman.cpp testbatch.cpp
GPU_HDR    = cuda_error_check.cuh needletail_general.hpp needletail_kernels.cuh needletail_threads.cuh poolman.hpp pools.hpp testbatch.hpp

all: gpu_nw base_nw batchgen

gpu_nw: $(GPU_SRC) $(GPU_HDR)
	$(NVCC) $(NVCC_FLAGS) $(GPU_SRC) -o $@.o $(CUDA_LINK)

base_nw: $(CPU_SRC) $(CPU_HDR)
	$(CC) $(CFLAGS) $(CPU_SRC) -o $@.o $(CUDA_INC)

batchgen: batchgen.cpp aligngen.cpp aligngen.hpp
	$(CC) -std=c++11 batchgen.cpp aligngen.cpp -o $@.o

clean:
	rm -rf *.o
