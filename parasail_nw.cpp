#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include "testbatch.hpp"
#include "external/parasail-master/parasail.h"

using namespace std;

int main( int argc, char *argv[] ) {
  TestBatch tb;
  Test_t test;

  ifstream fin;
  signed char matrix_in[16];
  signed char gap_score_in;

  parasail_matrix_t    *matrix;
  parasail_result_t    *result;
  parasail_traceback_t *traceback;

  chrono::high_resolution_clock::time_point start, end;
  chrono::microseconds alignment_time;

  // Verify argument count
  if ( argc != 5 ) {
    cerr << "Usage: " << argv[0] << " <matrix filename> <gap score> <batch filename> <result filename>" << endl;
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

  // Load the test batch
  if ( !tb.load_batch( argv[3] ) ) {
    cerr << "An error occurred while loading the batch. Exiting..." << endl;
    return 4;
  }

  // Loop through all the tests in the batch
  while ( tb.next_test( test ) ) {

    //cout << "On test " << test.id << "...\n";

    start = chrono::high_resolution_clock::now();

    // Perform the alignment (32-bit scores)
    /*
    result = parasail_nw_trace_striped_32(
      test.s2, test.s2_len,
      test.s1, test.s1_len,
      2, 1,
      matrix
      );
    */
    result = parasail_nw_trace_striped_8(
      test.s2, test.s2_len,
      test.s1, test.s1_len,
      -gap_score_in, -gap_score_in,
      matrix
      );

    end = chrono::high_resolution_clock::now();
    alignment_time += chrono::duration_cast<chrono::microseconds>( end - start );

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

  // Set batch-wide values
  tb.set_matrix( matrix_in );
  tb.set_gapscore( gap_score_in );
  tb.set_time( (unsigned long) alignment_time.count() );

  // Output the batch results
  tb.save_results( argv[4] );

  // Free parasail matrix
  parasail_matrix_free( matrix );

  return 0;
}