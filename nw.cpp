#include "nw_general.h"

int * nw_scoring(
  signed char * s,
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind
) {
  int * mat = new int [(qlen + 1) * (tlen + 1)];
  mat[0] = 0;
  for (uint32_t i = 1; i <= qlen; ++i)
    mat[(tlen + 1) * i] = mis_or_ind * i;
  for (uint32_t i = 1; i <= tlen; ++i)
    mat[i] = mis_or_ind * i;
  for (uint32_t i = 1; i <= qlen; ++i) {
    for (uint32_t j = 1; j <= tlen; ++j) {
      int match = mat[(tlen+1) * (i-1) + (j-1)] + nw_get_sim(s, q[i-1], t[j-1]);
      int del = mat[(tlen+1) * (i-1) + j] + mis_or_ind;
      int ins = mat[(tlen+1) * i + (j-1)] + mis_or_ind;
      int cell = match > del ? match : del;
      cell = cell > ins ? cell : ins;
      mat[(tlen+1) * i + j] = cell;
    }
  }

  // // TEMP: UNCOMMENT FOR MATRIX PRINTING!
  // for (int i = 0; i <= qlen; ++i) {
  //   for (int j = 0; j <= tlen; ++j)
  //     std::cout << std::setfill(' ') << std::setw(5)
  //       << mat[(tlen+1) * i + j] << " ";
  //   std::cout << std::endl;
  // }

  return mat;
}

int main() {
  // Input variables.
  std::string input_line;
  uint32_t tlen = 0;
  uint32_t qlen = 0;
  char * t = NULL;
  char * q = NULL;
  signed char * s = NULL;

  // Read in similarity matrix file.
  std::string sim_file = "datasets/similarity.txt";
  std::ifstream sim_file_stream(sim_file);
  s = new signed char[16];
  unsigned char sim_cnt = 0;
  while (std::getline(sim_file_stream, input_line)) {
    s[sim_cnt] = std::stoi(input_line);
    ++sim_cnt;
  }

  // Prepare our time recording.
  auto start = std::chrono::high_resolution_clock::now();
  auto finish = std::chrono::high_resolution_clock::now();
  auto runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);

  // Run through test file.
  for (uint32_t i = 0; i < NUM_TEST_FILES; ++i) {

    // Read in file.
    std::string test_file = "datasets/" + std::to_string(i) + ".txt";
    std::ifstream test_file_stream(test_file);
    uint32_t test_cnt = 0;
    while (std::getline(test_file_stream, input_line)) {
      if (test_cnt == 0) {
        tlen = std::stoll(input_line);
        t = new char [tlen + 1];
      }
      if (test_cnt == 1) {
        qlen = std::stoll(input_line);
        q = new char [qlen + 1];
      }
      if (test_cnt == 2)
        strcpy(t, input_line.c_str());
      if (test_cnt == 3)
        strcpy(q, input_line.c_str());
      ++test_cnt;
    }

    // Run matrix computation and time runtime.
    start = std::chrono::high_resolution_clock::now();
    int * nw_score_mat = nw_scoring(s, t, q, tlen, qlen, GAP_SCORE);
    finish = std::chrono::high_resolution_clock::now();
    runtime += std::chrono::duration_cast<std::chrono::microseconds>(finish - start);

	// Debug print score matrix as pointer matrix.
	/* print_score_as_ptr_mat(nw_score_mat, s, t, q, tlen, qlen, GAP_SCORE); */

    // Backtrack through matrix.
    nw_backtrack(nw_score_mat, s, t, q, tlen, qlen, GAP_SCORE);

    // Clean up memory
    delete [] nw_score_mat;
    delete [] q;
    delete [] t;
  }

  // Clean up similarity matrix memory.
  delete [] s;
  // Print out runtime and kill program.
  std::cerr << "Base Runtime: " << runtime.count() << " us" << std::endl;
  return 0;
}
