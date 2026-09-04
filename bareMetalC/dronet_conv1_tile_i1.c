// bareMetal repro that forces 196 outer iters of OS-path
// sp_tiled_matmul_os, exactly mirroring our merlin plugin's emission
// shape (tileI=1, tileJ=2, tileK=2 over dronet conv1 3136x32x27).
//
// If THIS hangs: the HW genuinely doesn't tolerate 196 outer-iter OS
// sequences; fix is to raise tile_I in the plugin's autotile.
// If THIS passes: the HW is fine with 196 outer iters; bug is purely
// in our plugin's emission state surrounding the inner call.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

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

#define A_STRIDE MAT_DIM_K
#define B_STRIDE MAT_DIM_J

// Match merlin plugin's clamped autotile choice for this shape:
//   I0 = 3136 / (1*16) = 196 outer-I iters
//   J0 =   32 / (2*16) =   1 outer-J iter
//   K0 =   32 / (2*16) =   1 outer-K iter (K=27 pads to 32 = 2*DIM)
#define TILE_I 1
#define TILE_J 2
#define TILE_K 2

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

    printf("dronet conv1 tile_I=1 repro: tiled_matmul(%d, %d, %d, "
           "tile_I=%d, tile_J=%d, tile_K=%d, OS)\n",
           MAT_DIM_I, MAT_DIM_J, MAT_DIM_K, TILE_I, TILE_J, TILE_K);
    printf("starting (196 outer iters of sp_tiled_matmul_os)...\n");

    uint64_t start = read_cycles();

    tiled_matmul(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)full_A, (elem_t*)full_B,
            NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
            A_STRIDE, B_STRIDE, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
            TILE_I, TILE_J, TILE_K,
            A_TRANSPOSE, B_TRANSPOSE,
            false, false,
            0,
            OS);

    gemmini_fence();
    uint64_t end = read_cycles();
    printf("DONE tile_I=1 OS. Cycles: %llu\n",
           (unsigned long long)(end - start));
    printf(">>> DRONET CONV1 TILE_I=1 COMPLETED <<<\n");
    exit(0);
}
