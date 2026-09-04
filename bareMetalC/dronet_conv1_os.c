// Reproduce the OS-path hang with the regular gemmini-rocc-tests library.
// Same shape as dronet_conv1_repro.c (M=3136, J=32, K=27) but explicitly
// uses OS (output-stationary) dataflow which forces `sp_tiled_matmul_os`
// instead of `sp_tiled_matmul_ws`/LOOP_WS.
//
// If this hangs in bareMetal, the OS path with multi-iter outer loop has
// a real bug in gemmini-rocc-tests (independent of IREE/Zephyr/plugin).
// If this PASSES, the bug is something specific to our plugin emission.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define ACTIVATION NO_ACTIVATION
#define NO_BIAS 0
#define REPEATING_BIAS 1
#define A_TRANSPOSE 0
#define B_TRANSPOSE 0

#define MAT_DIM_I 3136
#define MAT_DIM_K 27
#define MAT_DIM_J 32

#define A_STRIDE MAT_DIM_K
#define B_STRIDE MAT_DIM_J

static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(1);
static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(1);
static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(1);
static acc_t  full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif
    gemmini_flush(0);

    printf("dronet conv1 OS-path repro: tiled_matmul_auto(%d, %d, %d, OS)\n",
           MAT_DIM_I, MAT_DIM_J, MAT_DIM_K);
    printf("starting...\n");

    uint64_t start = read_cycles();

    // dataflow = OS (last arg). This forces sp_tiled_matmul_os in the
    // inner of tiled_matmul_outer.
    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)full_A, (elem_t*)full_B,
            NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
            A_STRIDE, B_STRIDE, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
            A_TRANSPOSE, B_TRANSPOSE,
            false, false,
            0,
            OS);   /* <--- the critical change vs dronet_conv1_repro */

    gemmini_fence();
    uint64_t end = read_cycles();
    printf("DONE OS-path. Cycles: %llu\n", (unsigned long long)(end - start));
    printf(">>> DRONET CONV1 OS-PATH COMPLETED <<<\n");
    exit(0);
}
