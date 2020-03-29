#ifndef XS_CUH
#define XS_CUH

#include <memory>

#include <mutex>
#include <list>
#include <utility>
// #include <unordered_map>

#include "nw_general.h"
#include "xs_core.cuh"
#include "cuda_error_check.cuh"

int * xs_man(
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind
) {
  uint64_t num_GPU_mem_bytes = 3 * (tlen + 1) * sizeof(int);
  num_GPU_mem_bytes += (tlen + 1) * (qlen + 1) * sizeof(int);
  num_GPU_mem_bytes += tlen * sizeof(char);
  num_GPU_mem_bytes += qlen * sizeof(char);
  // Malloc memory for our program.
  void * GPU_mem = NULL;
  cuda_error_check( cudaMalloc((void **) & GPU_mem, num_GPU_mem_bytes) );
  // Create a stream.
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  int * mat = xs_t_geq_q_man(t, q, tlen, qlen, mis_or_ind, GPU_mem, &stream);
  cudaStreamSynchronize(stream);
  cudaFree(GPU_mem);

  // // TEMP: UNCOMMENT FOR MATRIX PRINTING!
  // for (int i = 0; i <= qlen; ++i) {
  //   for (int j = 0; j <= tlen; ++j)
  //     std::cout << std::setfill(' ') << std::setw(5)
  //       << mat[(tlen+1) * i + j] << " ";
  //   std::cout << std::endl;
  // }

  return mat;
}

/* threading API starts here */

struct MatFuture {
  // Arguments
  char * t;
  char * q;
  uint32_t tlen;
  uint32_t qlen;
  signed char mis_or_ind;
  
  // Results
  int* mat;
  bool ready;

  MatFuture(char* _t, char* _q, uint32_t _tlen, uint32_t _qlen, signed char _mis_or_ind) {
    assert(_t != nullptr);
    assert(_q != nullptr);
    assert(_tlen != 0);
    assert(_qlen != 0);

    t = _t;
    q = _q;
    tlen = _tlen;
    qlen = _qlen;
    mis_or_ind = _mis_or_ind;

    mat = nullptr;
    ready = false;
  }

  bool is_ready() { return ready; }
};


std::unique_ptr<MatFuture> xs_man_batch (
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind
);

void start_queing_thread();
void stop_queing_thread();

/* API stops here */

// NOTE: This could be a heap over the memory request as well, but this is fine for now
std::list<std::pair<size_t, MatFuture*>> waiting_list;

std::mutex list_mutex;

bool stop_spinning = false;

std::unique_ptr<MatFuture> xs_man_batch (
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind
) {
  size_t num_GPU_mem_bytes = 3 * (tlen + 1) * sizeof(int);
  num_GPU_mem_bytes += (tlen + 1) * (qlen + 1) * sizeof(int);
  num_GPU_mem_bytes += tlen * sizeof(char);
  num_GPU_mem_bytes += qlen * sizeof(char);

  std::unique_ptr<MatFuture> future;
  future.reset(new MatFuture(t, q, tlen, qlen, mis_or_ind));
  
  {
    const std::lock_guard<std::mutex> lock(list_mutex);
    waiting_list.push_back({num_GPU_mem_bytes,future.get()});
  }

  return std::move(future);
}

void spinning_func_t() {

  auto kernel_launch_t = [&](MatFuture* future) {
    uint64_t num_GPU_mem_bytes = 3 * (future->tlen + 1) * sizeof(int);
    num_GPU_mem_bytes += (future->tlen + 1) * (future->qlen + 1) * sizeof(int);
    num_GPU_mem_bytes += future->tlen * sizeof(char);
    num_GPU_mem_bytes += future->qlen * sizeof(char);
    // Malloc memory for our program.
    void * GPU_mem = NULL;
    cuda_error_check( cudaMalloc((void **) & GPU_mem, num_GPU_mem_bytes) );
    // Create a stream.
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    future->mat = xs_t_geq_q_man(future->t, future->q, future->tlen, future->qlen, future->mis_or_ind, GPU_mem, &stream);
    cudaStreamSynchronize(stream);
    cudaFree(GPU_mem);
    future->ready = true;
  
    // // TEMP: UNCOMMENT FOR MATRIX PRINTING!
    // for (int i = 0; i <= qlen; ++i) {
    //   for (int j = 0; j <= tlen; ++j)
    //     std::cout << std::setfill(' ') << std::setw(5)
    //       << mat[(tlen+1) * i + j] << " ";
    //   std::cout << std::endl;
    // }
  };

  while( !stop_spinning ) {
    if( waiting_list.empty() ) continue;

    auto launch_meta = waiting_list.front();
    const size_t num_GPU_mem_bytes = launch_meta.first;

    size_t free = 0, total = 0;
    while (free < num_GPU_mem_bytes) {
      cudaError_t err = cudaMemGetInfo(&free, &total);
      cuda_error_check(err);
    }

    // NOTE: Assuming that no other process is requesting memory concurrently.
    std::thread t(kernel_launch_t, launch_meta.second);
    t.detach();

    {
      std::lock_guard<std::mutex> guard(list_mutex);
      waiting_list.pop_front();
    }
  }

  assert( waiting_list.empty() );
}

void start_queing_thread() {
  std::thread t(spinning_func_t);
  t.detach();
}

void stop_queing_thread() {
  stop_spinning = true;
}

#endif
