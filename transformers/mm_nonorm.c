// Isolation micro-test A: a single Gemmini matmul with NO normalization.
// Should complete quickly on GemminiRocketConfig (norm HW absent is irrelevant).
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#define I 16
#define J 16
#define K 64

static elem_t A[I][K] row_align(1);
static elem_t B[K][J] row_align(1);
static elem_t C[I][J] row_align(1);

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall failed"); exit(1); }
#endif
    gemmini_flush(0);

    printf("mm_nonorm: start matmul (NO_ACTIVATION)\n");
    uint64_t start = read_cycles();
    tiled_matmul_auto(I, J, K,
        (elem_t*)A, (elem_t*)B,
        NULL, (elem_t*)C,
        K, J, 0, J,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false, false, false, 0, WS);
    gemmini_fence();
    uint64_t end = read_cycles();
    printf("mm_nonorm: DONE cycles=%llu\n", end - start);
    exit(0);
}
