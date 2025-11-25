// Copyright 2021 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Author: Matheus Cavalcante <matheusd@iis.ee.ethz.ch>
//         Basile Bougenot <bbougenot@student.ethz.ch>
//         Matteo Perotti <mperotti@iis.ee.ethz.ch>

#include "macros/vector/float_macros.h"
#include "macros/vector/vector_macros.h"

// Simple random test with similar values
void TEST_CASE1(void) {
  VSET(16, e16, m1);
  //              -4628.000,   5116.000, -9928.000,   9392.000, -140.875,
  //              6112.000,   2598.000,   3210.000,   528.000, -3298.000,
  //              -3674.000,   368.250,   1712.000, -8584.000, -2080.000,
  //              4336.000
  VLOAD_16(v2, 0xec85, 0x6cff, 0xf0d9, 0x7096, 0xd867, 0x6df8, 0x6913, 0x6a45,
           0x6020, 0xea71, 0xeb2d, 0x5dc1, 0x66b0, 0xf031, 0xe810, 0x6c3c);
  asm volatile("vfexpf.v v3, v2");
  asm volatile("vfexps.v v3, v2");
  asm volatile("vfcoshf.v v3, v2");
  asm volatile("vfcoshs.v v3, v2");
  asm volatile("vftanhf.v v3, v2");
  asm volatile("vftanhs.v v3, v2");
  asm volatile("vflog.v v3, v2");
  asm volatile("vfsin.v v3, v2");
  asm volatile("vfcos.v v3, v2");
  asm volatile("vfrsqrt.v v3, v2");
}
int main(void) {
//   INIT_CHECK();
  enable_vec();
  enable_fp();
  // Change RM to RTZ since there are issues with FDIV + RNE in fpnew
  // Update: there are issues also with RTZ...
  CHANGE_RM(RM_RTZ);

  TEST_CASE1();
  

//   EXIT_CHECK();
}
