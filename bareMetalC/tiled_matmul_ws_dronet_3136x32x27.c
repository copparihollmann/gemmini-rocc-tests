// Canonical chipyard LOOP_WS test for dronet's failing unaligned shape
// matmul_3136x32x27 (M=3136, N=32, K=27). K=27 is NOT a multiple of DIM=16
// (padK=5). Our IREE LOOP_WS codegen crashes with mcause=1 Instruction
// Access fault on this shape on FireSim Shuttle. If THIS canonical test
// also crashes on the same bitstream, the bug is in libgemmini/RTL for
// unaligned LOOP_WS. If it passes, the bug is in our codegen.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define CHECK_RESULT 1
#define NO_BIAS 1
#define FULL_BIAS_WIDTH 1

#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#endif

#define MAT_DIM_I 3136
#define MAT_DIM_K 27
#define MAT_DIM_J 32

void full_matmul(elem_t A[MAT_DIM_I][MAT_DIM_K], elem_t B[MAT_DIM_K][MAT_DIM_J], ACC_T D[MAT_DIM_I][MAT_DIM_J], full_t C_full[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t r = 0; r < MAT_DIM_I; r++)
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < MAT_DIM_K; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}

int full_is_equal(elem_t x[MAT_DIM_I][MAT_DIM_J], elem_t y[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i)
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      if (x[i][j] != y[i][j])
        return 0;
  return 1;
}

void full_matscale(full_t full[MAT_DIM_I][MAT_DIM_J], elem_t out[MAT_DIM_I][MAT_DIM_J], acc_scale_t scale) {
  for (size_t r = 0; r < MAT_DIM_I; r++)
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      full_t scaled = ACC_SCALE(full[r][c], scale);
#ifndef ELEM_T_IS_FLOAT
      full_t elem = scaled > elem_t_max ? elem_t_max : (scaled < elem_t_min ? elem_t_min : scaled);
      out[r][c] = elem;
#else
      out[r][c] = scaled;
#endif
    }
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    printf("MAT_DIM_I: %d\n", MAT_DIM_I);
    printf("MAT_DIM_J: %d\n", MAT_DIM_J);
    printf("MAT_DIM_K: %d\n", MAT_DIM_K);

    gemmini_flush(0);

    static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(1);
    static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(1);
    static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(1);
    static ACC_T full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);

    static full_t gold_full[MAT_DIM_I][MAT_DIM_J];
    static elem_t gold[MAT_DIM_I][MAT_DIM_J];

    // Deterministic init for reproducibility.
    for (size_t i = 0; i < MAT_DIM_I; ++i)
      for (size_t j = 0; j < MAT_DIM_K; ++j)
        full_A[i][j] = (i + j) & 0x1;
    for (size_t i = 0; i < MAT_DIM_K; ++i)
      for (size_t j = 0; j < MAT_DIM_J; ++j)
        full_B[i][j] = (i * 3 + j) & 0x1;
    for (size_t i = 0; i < MAT_DIM_I; ++i)
      for (size_t j = 0; j < MAT_DIM_J; ++j)
        full_D[i][j] = NO_BIAS ? 0 : 1;

    printf("Starting gemmini matmul %dx%dx%d\n", MAT_DIM_I, MAT_DIM_J, MAT_DIM_K);
    unsigned long start = read_cycles();

    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)full_A, (elem_t*)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false,
            false, !FULL_BIAS_WIDTH,
            0,
            WS);

    unsigned long end = read_cycles();
    printf("Cycles taken: %lu\n", end-start);

    full_matmul(full_A, full_B, full_D, gold_full);
    full_matscale(gold_full, gold, ACC_SCALE_IDENTITY);

    if (!full_is_equal(full_C, gold)) {
      printf("MISMATCH\n");
      // Print first 3 rows for diagnosis.
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < MAT_DIM_J; ++j)
          printf("%d ", full_C[i][j]);
        printf(" | gold: ");
        for (size_t j = 0; j < MAT_DIM_J; ++j)
          printf("%d ", gold[i][j]);
        printf("\n");
      }
      exit(1);
    }

    printf("PASS\n");
  exit(0);
}
