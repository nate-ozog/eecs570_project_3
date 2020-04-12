#ifndef _POOLMAN_H_
#define _POOLMAN_H_

#include <list>
#include <map>
#include <unordered_map>
#include <iostream>
#include <stdint.h>

class PoolMan
{
public:
  PoolMan() {};
  PoolMan( void *pool, uint64_t size, uint32_t align_pow );

  bool init( void *pool, uint64_t size, uint32_t align_pow );

  void *malloc( uint64_t size );
  void  free  ( void    *ptr  );

  uint64_t get_free_bytes();
  uint32_t get_free_count();
  uint64_t get_max_malloc_bytes();
  uint32_t get_alloc_count();

  void print_pool();

private:
  uint64_t  free_bytes_; // Number of bytes available in the pool
  uint32_t  align_pow_;  // 2^(align_pow_) is the minimum alignment for mallocs

  // Maps a segment size to an address
  // Using an ordered map for efficient searching
  // Using a multimap because multiple segments can have the same size
  std::multimap<uint64_t,uintptr_t> free_size_;

  // Maps an address to a segment size
  // Using an ordered map for efficient searching
  // Using a normal map because addresses are unique
  std::map<uintptr_t,uint64_t> free_addr_;

  // Maps an address to an allocation size
  std::unordered_map<void*,uint64_t> allocs_;
};

#endif