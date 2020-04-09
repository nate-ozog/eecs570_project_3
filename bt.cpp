#include <assert.h>
#include <bits/stdc++.h>

uint32_t ij_to_z(uint32_t i, uint32_t j, uint32_t qlen, uint32_t tlen) {
  uint32_t z = 0;
  uint32_t jpi = j + i;
  if (jpi <= qlen)
    z = jpi * (jpi + 1) / 2 + i;
  else if (jpi <= tlen)
    z = qlen * (qlen + 1) / 2 + (qlen + 1) * (jpi - qlen) + i;
  else
    z = (tlen + 1) * (qlen + 1)
      - (tlen + qlen + 1 - jpi) * (tlen + qlen + 2 - jpi) / 2
      + i - (jpi - tlen);
  // printf("%d\n", z);
  return z;
}

void test_3x5() {
  // Row 0.
  assert(ij_to_z(0,0,3,5) == 0);
  assert(ij_to_z(0,1,3,5) == 1);
  assert(ij_to_z(1,0,3,5) == 2);
  assert(ij_to_z(0,2,3,5) == 3);
  assert(ij_to_z(1,1,3,5) == 4);
  assert(ij_to_z(2,0,3,5) == 5);
  // Row 1.
  assert(ij_to_z(0,3,3,5) == 6);
  assert(ij_to_z(1,2,3,5) == 7);
  assert(ij_to_z(2,1,3,5) == 8);
  assert(ij_to_z(3,0,3,5) == 9);
  assert(ij_to_z(0,4,3,5) == 10);
  assert(ij_to_z(1,3,3,5) == 11);
  // Row 2.
  assert(ij_to_z(2,2,3,5) == 12);
  assert(ij_to_z(3,1,3,5) == 13);
  assert(ij_to_z(0,5,3,5) == 14);
  assert(ij_to_z(1,4,3,5) == 15);
  assert(ij_to_z(2,3,3,5) == 16);
  assert(ij_to_z(3,2,3,5) == 17);
  // Row 3.
  assert(ij_to_z(1,5,3,5) == 18);
  assert(ij_to_z(2,4,3,5) == 19);
  assert(ij_to_z(3,3,3,5) == 20);
  assert(ij_to_z(2,5,3,5) == 21);
  assert(ij_to_z(3,4,3,5) == 22);
  assert(ij_to_z(3,5,3,5) == 23);
}

void test_3x3() {
  // Row 0.
  assert(ij_to_z(0,0,3,3) == 0);
  assert(ij_to_z(0,1,3,3) == 1);
  assert(ij_to_z(1,0,3,3) == 2);
  assert(ij_to_z(0,2,3,3) == 3);
  // Row 1.
  assert(ij_to_z(1,1,3,3) == 4);
  assert(ij_to_z(2,0,3,3) == 5);
  assert(ij_to_z(0,3,3,3) == 6);
  assert(ij_to_z(1,2,3,3) == 7);
  // Row 2.
  assert(ij_to_z(2,1,3,3) == 8);
  assert(ij_to_z(3,0,3,3) == 9);
  assert(ij_to_z(1,3,3,3) == 10);
  assert(ij_to_z(2,2,3,3) == 11);
  // Row 3.
  assert(ij_to_z(3,1,3,3) == 12);
  assert(ij_to_z(2,3,3,3) == 13);
  assert(ij_to_z(3,2,3,3) == 14);
  assert(ij_to_z(3,3,3,3) == 15);
}

void test_1x5() {
  // Row 0.
  assert(ij_to_z(0,0,1,5) == 0);
  assert(ij_to_z(0,1,1,5) == 1);
  assert(ij_to_z(1,0,1,5) == 2);
  assert(ij_to_z(0,2,1,5) == 3);
  assert(ij_to_z(1,1,1,5) == 4);
  assert(ij_to_z(0,3,1,5) == 5);
  // Row 1.
  assert(ij_to_z(1,2,1,5) == 6);
  assert(ij_to_z(0,4,1,5) == 7);
  assert(ij_to_z(1,3,1,5) == 8);
  assert(ij_to_z(0,5,1,5) == 9);
  assert(ij_to_z(1,4,1,5) == 10);
  assert(ij_to_z(1,5,1,5) == 11);
}

void test_1x1() {
  // Row 0.
  assert(ij_to_z(0,0,1,1) == 0);
  assert(ij_to_z(0,1,1,1) == 1);
  // Row 1.
  assert(ij_to_z(1,0,1,1) == 2);
  assert(ij_to_z(1,1,1,1) == 3);
}

void test_2x2() {
  // Row 0.
  assert(ij_to_z(0,0,2,2) == 0);
  assert(ij_to_z(0,1,2,2) == 1);
  assert(ij_to_z(1,0,2,2) == 2);
  // Row 1.
  assert(ij_to_z(0,2,2,2) == 3);
  assert(ij_to_z(1,1,2,2) == 4);
  assert(ij_to_z(2,0,2,2) == 5);
  // Row 2.
  assert(ij_to_z(1,2,2,2) == 6);
  assert(ij_to_z(2,1,2,2) == 7);
  assert(ij_to_z(2,2,2,2) == 8);
}

int main() {
  test_3x5();
  test_3x3();
  test_1x5();
  test_1x1();
  test_2x2();
  return 0;
}