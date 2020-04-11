#include <iostream>
#include <fstream>
#include <ostream>
#include <getopt.h>
#include "aligngen.hpp"

using namespace std;

void print_help() {
  cerr << "Options [optional]:\n";
  cerr << "[--help,   -h]               Print this options listing\n";
  cerr << "[--output, -o] <filename>    Output file, outputs to stdout if omitted\n";
  cerr << " --tc,     -c  <int count>   Target count\n";
  cerr << " --qpt,    -p  <int count>   Queries per target\n";
  cerr << "[--tl,     -t] <int length>  Target length (lower limit if upper limit specified)\n";
  cerr << "[--tu,     -T] <int length>  Target length upper limit\n";
  cerr << " --ql,     -q  <int length>  Query  length (lower limit if upper limit specified)\n";
  cerr << "[--qu,     -Q] <int length>  Query  length upper limit\n";
  cerr << "[--tgc,    -g] <float ratio> Target gc content      (default: 0.5)\n";
  cerr << "[--qsp,    -s] <float prob>  Query swap probability (default: 0.0625)\n";
  cerr << "[--seed,   -r] <int seed>    Seed for RNG           (default: from time)\n";
  cerr << endl;
  return;
}

int main( int argc, char* argv[] ) {
  AlignGen ag;
  string target;
  ofstream fout;
  ostream *out = &cout;

  string filename          = "";

  int   target_count       = -1;
  int   queries_per_target = -1;

  int   target_len_lower   = -1;
  int   target_len_upper   = -1;
  float target_gc_content  = 0.5f;

  int   query_len_lower    = -1;
  int   query_len_upper    = -1;
  float query_swap_prob    = 0.0625f;

  struct option long_opt[] = {
    { "help",   no_argument,       NULL, 'h' }, // Print help
    { "output", required_argument, NULL, 'o' }, // Output file (print to stdout if not specified)
    { "tc",     required_argument, NULL, 'c' }, // Target count                                   (required)
    { "qpt",    required_argument, NULL, 'p' }, // Queries per target                             (required)
    { "tl",     required_argument, NULL, 't' }, // Target length (lower limit if upper specified) (required)
    { "tu",     required_argument, NULL, 'T' }, // Target length upper limit
    { "ql",     required_argument, NULL, 'q' }, // Query length (lower limit if upper specified)  (required)
    { "qu",     required_argument, NULL, 'Q' }, // Query length upper limit
    { "tgc",    required_argument, NULL, 'g' }, // Target gc content      (default is 0.5   )
    { "qsp",    required_argument, NULL, 's' }, // Query swap probability (default is 0.0625)
    { "seed",   required_argument, NULL, 'r' }, // RNG seed               (default from time)
    { NULL,     0,                 NULL, 0   }
  };
  char short_opt[] = "ho:c:p:t:T:q:Q:g:s:r:";
  int  opt;
  bool bad_opt = false;

  // Collect options
  while ( ( opt = getopt_long( argc, argv, short_opt, long_opt, NULL ) ) != -1 ) {
    switch( opt ) {

    case 'h':
      print_help();
      return 1;
      break;

    case 'o':
      filename = optarg;
      break;
    case 'c':
      target_count       = atoi( optarg );
      break;
    case 'p':
      queries_per_target = atoi( optarg );
      break;
    case 't':
      target_len_lower   = atoi( optarg );
      break;
    case 'T':
      target_len_upper   = atoi( optarg );
      break;
    case 'q':
      query_len_lower    = atoi( optarg );
      break;
    case 'Q':
      query_len_upper    = atoi( optarg );
      break;
    case 'g':
      target_gc_content  = atoi( optarg );
      break;
    case 's':
      query_swap_prob    = atof( optarg );
      break;
    case 'r':
      ag.set_seed( (unsigned int) atoi( optarg ) );
      break;

    default:
      return 3;
      break;
    }
  }

  // Validate options
  if ( target_count       <= 0 ) {
    cerr << "Target count must be > 0\n";
    bad_opt = true;
  }
  if ( queries_per_target <= 0 ) {
    cerr << "Queries per target must be > 0\n";
    bad_opt = true;
  }
  if ( target_len_lower   <= 0 ) {
    cerr << "Target [lower] length must be > 0\n";
    bad_opt = true;
  }
  if ( query_len_lower    <= 0 ) {
    cerr << "Query [lower] length must be > 0\n";
    bad_opt = true;
  }
  if ( target_gc_content < 0.0f || target_gc_content > 1.0f ) {
    cerr << "Target gc content must be between 0.0 and 1.0 (inclusive)\n";
    bad_opt = true;
  }
  if ( query_swap_prob < 0.0f || query_swap_prob > 1.0f ) {
    cerr << "Query swap probability must be between 0.0 and 1.0 (inclusive)\n";
    bad_opt = true;
  }
  if ( bad_opt ) {
    cerr << endl;
    return 4;
  }

  // Open the output file
  if ( filename.length() ) {
    fout.open( filename );
    if ( !fout.is_open() ) {
      cerr << "Unable to open " << filename << endl;
      return 5;
    }
    out = &fout;
  }

  // Correct upper-limit lengths
  if ( target_len_upper < target_len_lower )
    target_len_upper = target_len_lower;
  if ( query_len_upper < query_len_lower )
    query_len_upper = query_len_lower;

  // Write the active options to a label
  // TODO? add a comment field separate from labels to carry these into the results?
  *out << "'\nbatchgen options:";
  *out << "\ntc=  " << target_count;
  *out << "\nqpt= " << queries_per_target;
  *out << "\ntl=  " << target_len_lower;
  *out << "\ntu=  " << target_len_upper;
  *out << "\nql=  " << query_len_lower;
  *out << "\nqu=  " << query_len_upper;
  *out << "\ntgc= " << target_gc_content;
  *out << "\nqsp= " << query_swap_prob;
  *out << "\nseed=" << ag.get_seed();
  *out << "\n'\n"   << endl;

  // Generate and output the batch
  for ( int t = 0; t < target_count; t++ ) {

    // Generate a new target
    target = ag.gen_target( target_len_lower, target_len_upper, target_gc_content );

    // Write the target index and length as a label
    *out << "'" << "t" << t << ",l" << target.length() << "_'\n";

    // Write the target
    *out << "<" << target << ">\n\n";

    // Generate and write queries for this target
    for ( int q = 0; q < queries_per_target; q++ ) {
      if ( q != 0 )
        *out << "<>{";
      else
        *out << "  {";
      *out << ag.gen_read( query_len_lower, query_len_upper, query_swap_prob ) << "}\n";
    }
    *out << endl;
  }

  // Close the output file
  if ( fout.is_open() )
    fout.close();

  return 0;
}