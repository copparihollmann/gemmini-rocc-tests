// Isolation micro-test B: a single Gemmini matmul with SOFTMAX activation,
// exactly the pattern BERT attention uses (transformer.c line ~70).
// On GemminiRocketConfig (has_normalizations=false) this exercises the norm path.
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
static elem_t B[J][K] row_align(1);   // transpose_B
static elem_t C[I][J] row_align(1);

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall failed"); exit(1); }
#endif
    gemmini_flush(0);

    printf("mm_softmax: start matmul (SOFTMAX)\n");
    uint64_t start = read_cycles();
    tiled_matmul_auto(I, J, K,
        (elem_t*)A, (elem_t*)B,
        NULL, (elem_t*)C,
        K, K, 0, J,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        SOFTMAX, ACC_SCALE_IDENTITY, 0,
        false, false, true, false, false, 0, WS);
    gemmini_fence();
    uint64_t end = read_cycles();
    printf("mm_softmax: DONE cycles=%llu\n", end - start);
    exit(0);
}
