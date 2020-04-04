CC        = g++
CFLAGS    = -std=c++11
NVCC      = nvcc
NVCC_FLAGS= -arch=sm_61 -rdc=true ${CFLAGS}
CUDA_INC  = -I/usr/local/cuda/include/
CUDA_LINK = -lcuda
BASE_MAIN = nw.cpp
BASE_DEPS = nw.cpp nw_general.h
CUDA_MAIN = nw.cu
CUDA_DEPS = nw.cu nw_general.h xs.h xs_core.h

all: gpu_nw base_nw gpu_nw_debug

gpu_nw: $(CUDA_DEPS)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_MAIN) -o $@.o $(CUDA_LINK)

base_nw: $(BASE_DEPS)
	$(CC) $(CFLAGS) $(BASE_MAIN) -o $@.o $(CUDA_INC)

gpu_nw_debug: $(CUDA_DEPS)
	$(NVCC) $(NVCC_FLAGS) -G -g $(CUDA_MAIN) -o $@.o $(CUDA_LINK)

clean:
	rm -rf *.o
