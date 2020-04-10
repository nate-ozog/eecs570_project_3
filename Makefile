CC        = g++
CFLAGS    = -std=c++11
NVCC      = nvcc
CUDA_INC  = -I/usr/local/cuda/include/
CUDA_LINK = -lcuda
CPU_SRC = nw.cpp testbatch.cpp
CPU_HDR = nw_general.h testbatch.hpp
GPU_SRC = nw.cu testbatch.cpp
GPU_HDR = nw_general.h testbatch.hpp xs.cuh xs_core.cuh cuda_error_check.cuh

all: gpu_nw base_nw gpu_nw_debug

gpu_nw: $(GPU_SRC) $(GPU_HDR)
	$(NVCC) $(CFLAGS) $(GPU_SRC) -o $@.o $(CUDA_LINK)

base_nw: $(CPU_SRC) $(CPU_HDR)
	$(CC) $(CFLAGS) $(CPU_SRC) -o $@.o $(CUDA_INC)

aligngen_test: aligngen_test.cpp aligngen.cpp aligngen.h
	$(CC) -std=c++11 aligngen_test.cpp aligngen.cpp -o $@.o

testbatch_test: testbatch_test.cpp testbatch.cpp testbatch.h
	$(CC) -std=c++11 testbatch_test.cpp testbatch.cpp -o $@.o

batchgen: batchgen.cpp aligngen.cpp aligngen.h
	$(CC) -std=c++11 batchgen.cpp aligngen.cpp -o $@.o

parasail_nw: parasail_nw.cpp testbatch.cpp testbatch.hpp ./external/parasail-master/build/libparasail.a
	$(CC) -std=c++11 parasail_nw.cpp testbatch.cpp -L./external/parasail-master/build/ -lparasail -pthread -o parasail_nw.o

clean:
	rm -rf *.o
