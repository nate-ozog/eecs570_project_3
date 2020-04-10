CC         = g++
CFLAGS     = -std=c++11
NVCC       = nvcc
NVCC_FLAGS = -arch=sm_60 -rdc=true --default-stream per-thread ${CFLAGS}
CUDA_INC   = -I/usr/local/cuda/include/
CUDA_LINK  = -lcuda
CPU_SRC    = nw.cpp testbatch.cpp
CPU_HDR    = testbatch.hpp
GPU_SRC    = nw.cu testbatch.cpp
GPU_HDR    = nw_general.hpp testbatch.hpp xs.cuh xs_core.cuh cuda_error_check.cuh

gpu_nw: $(GPU_SRC) $(GPU_HDR)
  $(NVCC) $(NVCC_FLAGS) $(GPU_SRC) -o $@.o $(CUDA_LINK)

base_nw: $(CPU_SRC) $(CPU_HDR)
  $(CC) $(CFLAGS) $(CPU_SRC) -o $@.o $(CUDA_INC)

batchgen: batchgen.cpp aligngen.cpp aligngen.hpp
  $(CC) -std=c++11 batchgen.cpp aligngen.cpp -o $@.o

clean:
  rm -rf *.o