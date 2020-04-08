#ifndef CUDA_ERROR_CHECK_CUH
#define CUDA_ERROR_CHECK_CUH

void cuda_error_check(cudaError_t e) {
  if (e != cudaSuccess) {
    std::cerr << "CUDA FAILURE: " << cudaGetErrorString(e) << std::endl;
    exit(0);
  }
}

#endif