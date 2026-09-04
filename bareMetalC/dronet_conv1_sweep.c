// M-sweep diagnostic for the dronet_conv1 matmul shape. Runs
// `tiled_matmul_auto(M, 32, 27)` at increasing M values and prints
// cycles per call. If a particular M hangs the Gemmini RoCC engine
// on FireSimGemminiAndOPUShuttleConfig, that's the boundary we use
// to bisect the IREE-side bug.
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

#define MAT_DIM_K 27
#define MAT_DIM_J 32
#define MAT_DIM_I_MAX 3136

// Static buffers sized for the largest M.
static elem_t full_A[MAT_DIM_I_MAX][MAT_DIM_K] row_align(1);
static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(1);
static elem_t full_C[MAT_DIM_I_MAX][MAT_DIM_J] row_align(1);
static acc_t full_D[MAT_DIM_I_MAX][MAT_DIM_J] row_align_acc(1);

static void run_one(size_t dim_I) {
    printf("--- BEGIN M=%zu ---\n", dim_I);
    uint64_t start = read_cycles();

    tiled_matmul_auto(dim_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)full_A, (elem_t*)full_B,
            NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
            A_TRANSPOSE, B_TRANSPOSE,
            false, false,
            0,
            WS);

    gemmini_fence();
    uint64_t end = read_cycles();
    printf("DONE M=%zu cycles=%llu\n", dim_I,
           (unsigned long long)(end - start));
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif
    gemmini_flush(0);

    printf("dronet_conv1 M-sweep\n");

    run_one(16);
    run_one(32);
    run_one(64);
    run_one(128);
    run_one(256);
    run_one(512);
    run_one(1024);
    run_one(2048);
    run_one(3136);

    printf(">>> SWEEP COMPLETED <<<\n");
    exit(0);
}
