// Standalone bare-metal repro for the dronet conv1 matmul hang
// observed via the IREE/Zephyr merlin Gemmini plugin path on
// FireSimGemminiAndOPUShuttleConfig.
//
// Calls gemmini.h's reference tiled_matmul_auto with the EXACT
// shape that hangs through the plugin emission:
//   M = 3136, N = 32, K = 27   (= dronet conv1 after im2col)
//
// If THIS hangs, the bug is in the Gemmini RTL / FireSim Shuttle
// integration. If THIS works, the bug is definitively in the IREE
// plugin emission (likely a missing inter-iter fence the C-ref
// macros emit but our MLIR codegen does not).
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

// override Makefile's -DPRINT_TILE=0 so we can see what tile_I/J/K
// the C autotile picks for our 3136x32x27 shape.
#undef PRINT_TILE
#define PRINT_TILE 1

#include "include/gemmini_testutils.h"

#define ACTIVATION NO_ACTIVATION

#define NO_BIAS 0
#define REPEATING_BIAS 1

#define A_TRANSPOSE 0
#define B_TRANSPOSE 0

#define MAT_DIM_I 3136
#define MAT_DIM_K 27
#define MAT_DIM_J 32

#if A_TRANSPOSE==0
#define A_STRIDE MAT_DIM_K
#else
#define A_STRIDE MAT_DIM_I
#endif

#if B_TRANSPOSE==0
#define B_STRIDE MAT_DIM_J
#else
#define B_STRIDE MAT_DIM_K
#endif

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

#if A_TRANSPOSE==0
    static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(1);
#else
    static elem_t full_A[MAT_DIM_K][MAT_DIM_I] row_align(1);
#endif

#if B_TRANSPOSE==0
    static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(1);
#else
    static elem_t full_B[MAT_DIM_J][MAT_DIM_K] row_align(1);
#endif

    static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(1);
    static acc_t full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);

    printf("dronet conv1 repro: tiled_matmul_auto(%d, %d, %d)\n",
           MAT_DIM_I, MAT_DIM_J, MAT_DIM_K);
    printf("starting...\n");

    uint64_t start = read_cycles();

    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)full_A, (elem_t*)full_B,
            NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
            A_STRIDE, B_STRIDE, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
            A_TRANSPOSE, B_TRANSPOSE,
            false, false,
            0,
            WS);

    gemmini_fence();

    uint64_t end = read_cycles();
    printf("DONE. Cycles: %llu\n", (unsigned long long)(end - start));
    printf(">>> DRONET CONV1 REPRO COMPLETED SUCCESSFULLY <<<\n");

    exit(0);
}
