#include "poolman.h"

using namespace std;

PoolMan::PoolMan( uint32_t align_pow ) {
  align_pow_   = align_pow;
  free_bytes_  = 0;
  total_bytes_ = 0;
  peak_bytes_  = 0;
}

void PoolMan::add_pool( void *pool, uint64_t size ) {
  uintptr_t temp;
  uintptr_t base = (uintptr_t) pool;

  // Get base % 2^align_pow
  temp = base & ( ( 1 << align_pow_ ) - 1 );

  // If the pool isn't aligned, round up to the nearest aligned value
  // Update the pool size to reflect this change
  if ( temp != 0 ) {
    base += ( 1 << align_pow_ ) - temp;
    size -= ( 1 << align_pow_ ) - temp;
  }

  // Align the size by rounding down to the nearest aligned value
  size &= ~( ( 1 << align_pow_ ) - 1 );

  // Add the (aligned) pool to the free maps to start using it
  // TODO? Merge address-adjacent pools
  free_size_.insert( pair<uint64_t,uintptr_t>( size, base ) );
  free_addr_[base]  = size;
  free_bytes_      += size;
  total_bytes_     += size;
}

void *PoolMan::malloc( uint64_t size ) {
  void *result = NULL;
  multimap<uint64_t,uintptr_t>::iterator it;
  uint64_t  segment_size;
  uintptr_t segment_addr;
  uint64_t  temp;

  // Return NULL if no bytes requested
  if ( size == 0 )
    return result;

  // Round size up to the nearest aligned value
  temp = size & ( ( 1 << align_pow_ ) - 1 );
  if ( temp != 0 ) {
    size += ( 1 << align_pow_ ) - temp;
  }

  // Find the best fit
  it = free_size_.lower_bound( size );

  // If a segment was found
  if ( it != free_size_.end() ) {

    // Get the allocation starting address and record the allocation
    result = (void*) it->second;
    allocs_[result] = size;
    free_bytes_ -= size;

    // Remove this segment from the maps
    segment_size = it->first;
    segment_addr = it->second;
    free_addr_.erase( segment_addr );
    free_size_.erase( it );

    // If the segment isn't empty, add the remaining space to the maps
    if ( segment_size > size ) {
      free_size_.insert( pair<uint64_t,uintptr_t>( segment_size - size, segment_addr + size ) );
      free_addr_[ segment_addr + size ] = segment_size - size;
    }
  }

  // Update the peak counter, if necessary
  if ( total_bytes_ - free_bytes_ > peak_bytes_ )
    peak_bytes_ = total_bytes_ - free_bytes_;

  // Dump some stats and exit if returning NULL
  // TODO: Have the caller wait for space instead of this
  if ( result == NULL ) {
    cerr << "PoolMan::malloc returning NULL\n";
    cerr << " Requested Size B: " << size << "\n";
    cerr << "           Free B: " << free_bytes_ << "\n";
    cerr << "          Total B: " << total_bytes_ << "\n";
    cerr << "Max Avail. Seg. B: " << get_max_malloc_bytes() << "\n";
    cerr << "      Alloc Count: " << get_alloc_count() << "\n";
    cerr << "       Free Count: " << get_free_count()  << "\n";
    exit(-2);
  }

  return result;
}

void PoolMan::free( void *ptr ) {
  uint64_t  size;
  uintptr_t addr;
  bool      merged = false;
  map<uintptr_t,uint64_t>::iterator      ita;
  multimap<uint64_t,uintptr_t>::iterator its;

  // Don't try to free invalid addresses
  if ( allocs_.count( ptr ) == 0 )
    return;

  // Get this allocation's size and address
  size = allocs_[ptr];
  addr = (uintptr_t) ptr;
  free_bytes_ += size;

  // Erase this allocation
  allocs_.erase( ptr );

  // Get the first address greater-than [or equal to] this one
  ita = free_addr_.lower_bound( addr );

  // If this segment ends at the start of the next, merge the segments
  if ( ita != free_addr_.end() && ( addr + size ) == ita->first ) {

    // Get the target segment's iterator in the size multimap
    for ( its = free_size_.find( ita->second ); its != free_size_.end(); its++ )
      if ( its->second == ita->first )
        break;

    size += ita->second;
    free_addr_.erase( ita );
    free_size_.erase( its );

    // Re-get the first address greater-than or equal to this one
    ita = free_addr_.lower_bound( addr );
  }

  // If the first address greater-than or equal to this one isn't the first address,
  if ( ita != free_addr_.begin() ) {

    // Go back one address to get the address just before this one
    ita--;

    // If the previous segment ends at this address, merge the segments
    if ( ita->first + ita->second == addr ) {

      // Get the target segment's iterator in the size multimap
      for ( its = free_size_.find( ita->second ); its != free_size_.end(); its++ )
        if ( its->second == ita->first )
          break;

      size += ita->second;
      addr  = ita->first;
      free_addr_.erase( ita );
      free_size_.erase( its );
    }
  }

  // Add this segment to the free maps
  free_addr_[addr] = size;
  free_size_.insert( pair<uint64_t,uintptr_t>( size, addr ) );
}

uint64_t PoolMan::get_free_bytes()  { return free_bytes_;       }

uint64_t PoolMan::get_total_bytes() { return total_bytes_;      }

uint32_t PoolMan::get_free_count()  { return free_size_.size(); }

uint64_t PoolMan::get_max_malloc_bytes() {
  if ( !free_size_.empty() )
    return ( --free_size_.end() )->first;
  else
    return 0;
}

uint32_t PoolMan::get_alloc_count() { return allocs_.size();    }

uint64_t PoolMan::get_peak_bytes()  { return peak_bytes_;       }

void PoolMan::print_pool() {

  cout << "Allocs\n";
  for ( auto it = allocs_.begin(); it != allocs_.end(); it++ ) {
    cout << (uintptr_t) it->first << " " << it->second << "\n"; 
  }

  cout << "\n";

  cout << "Free map (size)\n";
  for ( auto it = free_size_.begin(); it != free_size_.end(); it++ ) {
    cout << it->first << " " << it->second << "\n";
  }

  cout << "\n";

  cout << "Free map (addr)\n";
  for ( auto it = free_addr_.begin(); it != free_addr_.end(); it++ ) {
    cout << it->first << " " << it->second << "\n";
  }
}