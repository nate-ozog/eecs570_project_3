#ifndef TESTBATCH_HPP
#define TESTBATCH_HPP

#include <vector>
#include <string>

// Existing Test_t instances may become invalid after 'load_batch' calls
struct Test_t {
int          id;
const char * s1;
int          s1_len;
const char * s2;
int          s2_len;
};

class TestBatch
{
public:

  TestBatch() {};
  TestBatch( const char * const filename );

  bool load_batch( const char * const filename );
  void clear     ();

  int  count    ();
  bool next_test( Test_t &out );
  bool set_next ( const int id );
  bool get_test ( const int id, Test_t      &out );
  bool get_label( const int id, std::string &out );

  bool log_result  ( const int id,    const char * const s1_align, const char * const s2_align,
                     const int score, const unsigned long time );
  void set_matrix  ( const signed char * const new_matrix );
  void set_gapscore( const signed char new_gap_score );
  void set_time    ( const unsigned long new_time );
  bool save_results( const char * const filename );

private:

  struct InternalTest_t {
    int label_base_idx;
    int label_offset;
    int s1_idx;
    int s2_idx;
    int s1_align_idx;
    int s2_align_idx;
    int score;
    unsigned long time;
  };

  std::vector<InternalTest_t> tests;
  std::vector<std::string>    test_strings = {""}; // String 0 is expected to exist and be empty
  std::vector<std::string>    result_strings;

  int next_idx __attribute__((aligned(64))) = 0;

  signed char   matrix[16] = {0};
  signed char   gap_score  = 0;
  unsigned long time       = 0;
};

#endif