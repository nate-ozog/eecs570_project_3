#ifndef ALIGNGEN_HPP
#define ALIGNGEN_HPP

#include <string>
#include <random>
#include <chrono>

class AlignGen
{
public:
  AlignGen();
  AlignGen( unsigned int initial_seed );

  void         set_target( std::string new_target );
  std::string  get_target();
  void         set_seed( unsigned int new_seed );
  unsigned int get_seed();

  std::string gen_target( unsigned int len_lower, unsigned int len_upper, float gc_content = 0.5f );
  std::string gen_read(   unsigned int len_lower, unsigned int len_upper, float swap_prob );

private:
  std::string                target;
  std::default_random_engine generator;
  unsigned int               seed;
};

#endif