#include "aligngen.hpp"

using namespace std;

#define RANDOM_UINT(a,b) uniform_int_distribution<unsigned int>{a,b}(generator)
#define RANDOM_FLOAT uniform_real_distribution<float>{}(generator)

AlignGen::AlignGen() {
  seed = (unsigned int)chrono::high_resolution_clock::now().time_since_epoch().count();
  generator.seed( seed );
}
AlignGen::AlignGen( unsigned int initial_seed ) {
  seed = initial_seed;
  generator.seed( initial_seed );
}

// new_target should only contain characters from the set {'A', 'T', 'C', 'G'}
void AlignGen::set_target( string new_target ) {
  target = new_target;
}
string AlignGen::get_target() {
  return target;
}
void AlignGen::set_seed( unsigned int new_seed ) {
  seed = new_seed;
  generator.seed( new_seed );
}
unsigned int AlignGen::get_seed() {
  return seed;
}

// Generates a target sequence string with the properties:
// - Length uniformly distributed on the interval [len_lower, len_upper]
// - Containing characters from the set {'A', 'T', 'C', 'G'}
// - P(A) = P(T) and P(C) = P(G) (Chargaff's 2nd rule?)
// - P(G or C) = gc_content
// gc_content should be on the interval [0.0, 1.0]
string AlignGen::gen_target( unsigned int len_lower, unsigned int len_upper, float gc_content ) {
  unsigned int len = RANDOM_UINT( len_lower, len_upper );
  string       new_target;

  new_target.reserve(len);
  for ( unsigned int i = 0; i < len; i++ ) {
    if ( RANDOM_FLOAT >= gc_content )
      new_target += ( RANDOM_FLOAT >= 0.5f ) ? 'A' : 'T';
    else
      new_target += ( RANDOM_FLOAT >= 0.5f ) ? 'G' : 'C';
  }

  target = new_target;
  return new_target;
}

string AlignGen::gen_read( unsigned int len_lower, unsigned int len_upper, float swap_prob ) {
  uniform_int_distribution<unsigned int> new_base_dist( 0, 2 );
  unsigned int len_read   = RANDOM_UINT( len_lower, len_upper );
  unsigned int len_target = target.length();

  if ( len_read > len_target )
    len_read = len_target;

  unsigned int start = RANDOM_UINT( 0, len_target - len_read );
  string       read = target.substr( start, len_read );

  // TODO?: Deletion pass
  // - Read n extra characters in prior step
  // - Remove n characters randomly from the read string

  // Swap pass:
  // - Each character has 'swap_prob' odds of being switched to another base
  // - The new base is chosen uniformly (would more flexibility be useful?)
  // Idea: dependent odds for more "realistic" runs of errors
  // Idea: more flexible (e.g. per-base) odds
  for ( int i = 0; i < len_read; i++ ) {
    if ( RANDOM_FLOAT < swap_prob ) {
      switch (read[i]) {
        case 'A':
          read[i] = "TCG"[ RANDOM_UINT( 0, 2 ) ];
          break;
        case 'T':
          read[i] = "ACG"[ RANDOM_UINT( 0, 2 ) ];
          break;
        case 'C':
          read[i] = "ATG"[ RANDOM_UINT( 0, 2 ) ];
          break;
        case 'G':
          read[i] = "ATC"[ RANDOM_UINT( 0, 2 ) ];
          break;
      }
    }
  }

  return read;
}