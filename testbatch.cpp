#include "testbatch.hpp"
#include <fstream>
#include <ctype.h>
#include <algorithm>
#include <iostream>

using namespace std;


enum Parse_State { NONE, IN_LABEL, IN_S1, IN_S2 };

static void open_error( const char *filename )
{
  cerr << "An error occurred while opening \"" << filename << "\"" << endl;
}

static void char_error( const char *filename, const int line_num, const char c )
{
  cerr << filename << ":" << line_num << ": unexpected character \'" << c << "\'" << endl;
}


TestBatch::TestBatch( const char * const filename ) {
  load_batch( filename );
}

bool TestBatch::load_batch( const char * const filename ) {
  ifstream fin( filename );
  Parse_State state    = NONE;
  string      line;
  int         line_num = 0;

  string      label, s1, s2;
  int         str_start_line;
  int         label_offset = 0;
  bool        have_s1 = false;
  bool        have_s2 = false;

  // All fields of this struct are initialized to 0
  InternalTest_t new_test = {};

  // Verify that the file was opened
  if ( !fin.is_open() ) {
    open_error( filename );
    return false;
  }

  // Loop over lines in the file
  while ( getline( fin, line ) ) {
    line_num++;

    // Remove all whitespace in the current line
    line.erase( remove_if( line.begin(), line.end(), ::isspace ), line.end() );

    // Loop over characters in the line
    for ( auto c : line ) {
      switch ( state ) {

      // NONE state: not yet parsing a label, s1, or s2 string
      case NONE:
        // If both s1 and s2 have been defined, create a test
        if ( have_s1 && have_s2 ) {
          // For each new string (label, s1, s2), add the string
          // to the 'test_strings' vector. The new test references
          // these strings via their index in the vector.
          if ( label.length() ) {
            test_strings.push_back( label );
            new_test.label_base_idx = test_strings.size() - 1;
          }
          if ( s1.length() ) {
            test_strings.push_back( s1 );
            new_test.s1_idx = test_strings.size() - 1;
          }
          if ( s2.length() ) {
            test_strings.push_back( s2 );
            new_test.s2_idx = test_strings.size() - 1;
          }

          // Assign the label offset to this test, then increment the counter
          new_test.label_offset = label_offset;
          label_offset++;

          // Add this test to the collection of tests
          tests.push_back( new_test );

          // Reset state variables and continue parsing
          label   = "";
          have_s1 = false;
          have_s2 = false;
        }

        switch ( c ) {
        // ' - denotes the start of label
        case '\'':
          label          = "";
          label_offset   = 0;
          str_start_line = line_num;
          state          = IN_LABEL;
          break;
        // < - denotes the start of s1
        case '<':
          s1             = "";
          str_start_line = line_num;
          state          = IN_S1;
          break;
        // { - denotes the start of s2
        case '{':
          s2             = "";
          str_start_line = line_num;
          state          = IN_S2;
          break;
        // All other characters are a syntax error
        default:
          char_error( filename, line_num, c );
          fin.close();
          return false;
          break;
        }
        break; // NONE

      // IN_LABEL state: currently parsing the contents of a label
      case IN_LABEL:
        // ' - denotes the end of a label
        if ( c == '\'' )
          state = NONE;
        else
          label += c;
        break; // IN_LABEL

      // IN_S1 state: currently parsing an s1 string
      case IN_S1:
        c = toupper( c );
        switch ( c ) {
        // Only allowed to use AaTtCcGg
        case 'A':
        case 'T':
        case 'C':
        case 'G':
          s1 += c;
          break;
        // > - denotes the end of an s1
        case '>':
          state = NONE;
          have_s1 = true;
          break;
        // All other characters are a syntax error
        default:
          char_error( filename, line_num, c );
          fin.close();
          return false;
          break;
        }
        break; // IN_S1

      // IN_S2 state: currently parsing an s2 string
      case IN_S2:
        c = toupper( c );
        switch ( c ) {
        // Only allowed to use AaTtCcGg
        case 'A':
        case 'T':
        case 'C':
        case 'G':
          s2 += c;
          break;
        // } - denotes the end of an s2
        case '}':
          state = NONE;
          have_s2 = true;
          break;
        // All other characters are a syntax error
        default:
          char_error( filename, line_num, c );
          fin.close();
          return false;
          break;
        }
        break; // IN_S2

      // How did you get here?
      default:
        cerr << "Unhandled Parse_State enum!" << endl;
        break;
      }
    }
  }

  // Close the input file
  fin.close();

  // Handle any remaining parsing activities
  switch ( state ) {
  case NONE:
    // If both s1 and s2 have been defined, create a test (see copy above for comments)
    if ( have_s1 && have_s2 ) {
      if ( label.length() ) {
        test_strings.push_back( label );
        new_test.label_base_idx = test_strings.size() - 1;
      }
      if ( s1.length() ) {
        test_strings.push_back( s1 );
        new_test.s1_idx = test_strings.size() - 1;
      }
      if ( s2.length() ) {
        test_strings.push_back( s2 );
        new_test.s2_idx = test_strings.size() - 1;
      }
      new_test.label_offset = label_offset;
      tests.push_back( new_test );
    }
    break;
  case IN_LABEL:
    cerr << filename << ":" << line_num << ": missing end \' for label started on line " << str_start_line << endl;
    return false;
    break;
  case IN_S1:
    cerr << filename << ":" << line_num << ": missing > for S1 started on line " << str_start_line << endl;
    return false;
    break;
  case IN_S2:
    cerr << filename << ":" << line_num << ": missing } for S2 started on line " << str_start_line << endl;
    return false;
    break;
  default:
    break;
  }

  // DEBUG: print strings
  //for ( auto i : test_strings )
  //  cout << i << endl;

  // Pre-allocate vector entries for the result strings (2 per test)
  result_strings.resize( tests.size() * 2, "" );

  // Reference the result strings by index
  for ( int i = 0; i < tests.size(); i++ ) {
    tests[i].s1_align_idx = i * 2;
    tests[i].s2_align_idx = i * 2 + 1;
  }

  // Batch imported from file without any errors
  return true;
}

void TestBatch::clear() {
  // Clear all stored strings
  tests.clear();
  test_strings.clear();
  result_strings.clear();

  // Re-initialize member variables
  test_strings.push_back("");
  next_idx = 0;
  for ( int i = 0; i < 16; i++ )
    matrix[i] = 0;
  gap_score = 0;
}

int TestBatch::count() {
  return tests.size();
}

bool TestBatch::next_test( Test_t &out ) {
  // Atomically fetch the next test's index and increment the counter
  int curr_idx = __atomic_fetch_add( &next_idx, 1, __ATOMIC_RELAXED );

  // If at the end of the test vector, no more tests available
  if ( curr_idx >= tests.size() )
    return false;

  out.id     = curr_idx;
  out.s1     = test_strings[ tests[curr_idx].s1_idx ].c_str();
  out.s1_len = test_strings[ tests[curr_idx].s1_idx ].length();
  out.s2     = test_strings[ tests[curr_idx].s2_idx ].c_str();
  out.s2_len = test_strings[ tests[curr_idx].s2_idx ].length();

  return true;
}

bool TestBatch::set_next( const int id ) {
  if ( id >= tests.size() )
    return false;
  next_idx = id;
  return true;
};

bool TestBatch::get_test( const int id, Test_t &out ) {
  if ( id >= tests.size() )
    return false;

  out.id     = id;
  out.s1     = test_strings[ tests[id].s1_idx ].c_str();
  out.s1_len = test_strings[ tests[id].s1_idx ].length();
  out.s2     = test_strings[ tests[id].s2_idx ].c_str();
  out.s2_len = test_strings[ tests[id].s2_idx ].length();

  return true;
}

bool TestBatch::get_label( const int id, string &out ) {
  if ( id >= tests.size() )
    return false;

  out = test_strings[ tests[id].label_base_idx ] + to_string( tests[id].label_offset );

  return true;
}

bool TestBatch::log_result( const int id,    const char * const s1_align, const char * const s2_align,
                            const int score, const unsigned long time ) {
  if ( id >= tests.size() )
    return false;

  // Keep a copy of both aligned strings
  result_strings[ tests[id].s1_align_idx ] = s1_align;
  result_strings[ tests[id].s2_align_idx ] = s2_align;

  // Update the test score and time
  tests[id].score = score;
  tests[id].time  = time;

  return true;
}

void TestBatch::set_matrix( const signed char * const new_matrix ) {
  for ( int i = 0; i < 16; i++ )
    matrix[i] = new_matrix[i];
}

void TestBatch::set_gapscore( const signed char new_gap_score ) {
  gap_score = new_gap_score;
}

void TestBatch::set_time( const unsigned long new_time ) {
  time = new_time;
}

bool TestBatch::save_results( const char * const filename ) {
  ofstream fout( filename );
  string s;

  // Verify that the file was opened
  if ( !fout.is_open() ) {
    open_error( filename );
    return false;
  }

  // Write the batch time
  fout << "t= " << time << "\n\n";

  // Write the similarity matrix
  fout << "m= ";
  for ( int i = 0; i < 15; i++ ) {
    fout << (int)matrix[i] << ",";
  }
  fout << (int)matrix[15] << "\n";

  // Write the gap score
  fout << "g= " << (int)gap_score << "\n\n";

  // Write the results for each test
  for ( int i = 0; i < tests.size(); i++ ) {
    get_label( i, s );

    // Write the full label, aligned strings, and alignment score
    fout << "\'" << s << "\'\n";
    fout << "<" << result_strings[ tests[i].s1_align_idx ] << ">\n";
    fout << "{" << result_strings[ tests[i].s2_align_idx ] << "}\n";
    fout << "s= " << tests[i].score << "\n";
    fout << "t= " << tests[i].time  << "\n\n";
  }

  fout.close();
  return true;
}
