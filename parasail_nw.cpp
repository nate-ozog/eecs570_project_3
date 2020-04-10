#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <pthread.h>
#include <vector>
#include "testbatch.hpp"
#include "external/parasail-master/parasail.h"

using namespace std;


// Global since all threads share this data
TestBatch          tb;
signed char        gap_score_in;
parasail_matrix_t *matrix;


void *compute_alignments( void *arg ) {
  Test_t                test;
  parasail_result_t    *result;
  parasail_traceback_t *traceback;

  chrono::high_resolution_clock::time_point start, end;


  // Loop through all the tests in the batch
  while ( tb.next_test( test ) ) {

    //cout << "On test " << test.id << "...\n";

    start = chrono::high_resolution_clock::now();

    // Perform the alignment (32-bit scores)
    result = parasail_nw_trace_striped_32(
      test.s2, test.s2_len,
      test.s1, test.s1_len,
      -gap_score_in, -gap_score_in,
      matrix
      );

    end = chrono::high_resolution_clock::now();

    // Backtrack to get the aligned strings
    traceback = parasail_result_get_traceback(
      result,
      test.s2, test.s2_len,
      test.s1, test.s1_len,
      matrix,
      ' ', '+', '-'
      );

    // Record the alignment results
    tb.log_result( test.id, traceback->ref, traceback->query, result->score,
                   chrono::duration_cast<chrono::microseconds>( end - start ).count() );

    // Free parasail objects
    parasail_result_free( result );
    parasail_traceback_free( traceback );
  }

}


int main( int argc, char *argv[] ) {
  Test_t            test;
  ifstream          fin;
  signed char       matrix_in[16];
  int               worker_count = 0;
  vector<pthread_t> workers;

  chrono::high_resolution_clock::time_point start, end;
  chrono::microseconds batch_time;


  // Verify argument count
  if ( argc != 5 && argc != 6 ) {
    cerr << "Usage: " << argv[0] << " <matrix filename> <gap score> <batch filename> <result filename> [worker threads]" << endl;
    return 1;
  }

  // Open the matrix file
  fin.open( argv[1] );
  if ( !fin.is_open() ) {
    cerr << "An error occurred while opening \"" << argv[1] << "\"" << endl;
    return 2;
  }

  // Initialize a parasail matrix
  matrix = parasail_matrix_create( "ATGC", 1, -1 );

  // Read the matrix and close the file
  for ( int i = 0; i < 16; i++ ) {
    int temp = 0;
    if ( fin.eof() ) {
      cerr << "Incomplete matrix in \"" << argv[1] << "\"" << endl;
      fin.close();
      return 3;
    }
    fin >> temp;
    matrix_in[i] = (signed char) temp;
    parasail_matrix_set_value( matrix, i / 4, i % 4, (int) matrix_in[i] );
  }
  fin.close();

  // Get the gap score
  gap_score_in = atoi( argv[2] );

  // Get the worker thread count
  if ( argc == 6 ) {
    worker_count = atoi( argv[5] );
    if ( worker_count < 0 ) {
      cerr << "Worker thread count must be >= 0" << endl;
      return 4;
    }
    workers.resize( worker_count );
  }

  // Load the test batch
  if ( !tb.load_batch( argv[3] ) ) {
    cerr << "An error occurred while loading the batch. Exiting..." << endl;
    return 5;
  }

  // Start timing the batch alignment
  start = chrono::high_resolution_clock::now();

  // Don't launch threads if 0 workers are requested
  if ( worker_count == 0 ) {
    compute_alignments( NULL );
  }
  else {
    // Launch parallel worker threads
    for ( int i = 0; i < worker_count; i++ )
      pthread_create( &workers[i], NULL, compute_alignments, NULL );

    // Join the parallel worker threads
    for ( int i = 0; i < worker_count; i++ )
      pthread_join( workers[i], NULL );
  }
  
  // Stop timing the batch alignment
  end = chrono::high_resolution_clock::now();

  // Set batch-wide values
  tb.set_matrix( matrix_in );
  tb.set_gapscore( gap_score_in );
  tb.set_time( chrono::duration_cast<chrono::microseconds>( end - start ).count() );

  // Output the batch results
  tb.save_results( argv[4] );

  // Free the parasail matrix
  parasail_matrix_free( matrix );

  return 0;
}
