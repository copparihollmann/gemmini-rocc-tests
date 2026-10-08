// Region-cycle probe for the WS INT8 Gemmini M2 overlap model (ModeLIR).
// Measures the RAW per-tile mvin and mvout DMA latency (unoverlapped), fenced so each
// bracket isolates its phase. Compute is measured elsewhere (bit-exact io_busy floor),
// so this probe only needs the two DMA rates the overlap roofline is missing.
// Prints CSV lines: "PROBE,<phase>,<ntiles>,<cycles>" for a few tile counts so the
// per-tile latency = slope (amortizes the fixed issue/fence overhead).
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define MAXT 32                       // max tiles measured in one bracket (fits scratchpad)
static elem_t A[MAXT][DIM][DIM] row_align(1);
static elem_t Cout[MAXT][DIM][DIM] row_align(1);

static const int tile_counts[] = {1, 2, 4, 8, 16, 32};

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall failed"); exit(1); }
#endif
    gemmini_flush(0);
    for (size_t t = 0; t < MAXT; ++t)
        for (size_t i = 0; i < DIM; ++i)
            for (size_t j = 0; j < DIM; ++j) { A[t][i][j] = (i*3+j+t)&0x7f; Cout[t][i][j] = 0; }

    gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, false, false);
    gemmini_config_ld(DIM * sizeof(elem_t));
    gemmini_config_st(DIM * sizeof(elem_t));
    gemmini_fence();

    printf("PROBE_BEGIN\n");

    // MVIN: DRAM -> scratchpad, per DIM×DIM tile, fenced.
    for (size_t s = 0; s < sizeof(tile_counts)/sizeof(tile_counts[0]); ++s) {
        int nt = tile_counts[s];
        gemmini_fence();
        uint64_t t0 = read_cycles();
        for (int n = 0; n < nt; ++n)
            gemmini_mvin(A[n], n * DIM);
        gemmini_fence();
        uint64_t t1 = read_cycles();
        printf("PROBE,mvin,%d,%lu\n", nt, (unsigned long)(t1 - t0));
    }

    // MVOUT: scratchpad -> DRAM, per DIM×DIM tile, fenced. (scratchpad already holds the mvin'd tiles.)
    for (size_t s = 0; s < sizeof(tile_counts)/sizeof(tile_counts[0]); ++s) {
        int nt = tile_counts[s];
        gemmini_fence();
        uint64_t t0 = read_cycles();
        for (int n = 0; n < nt; ++n)
            gemmini_mvout(Cout[n], n * DIM);
        gemmini_fence();
        uint64_t t1 = read_cycles();
        printf("PROBE,mvout,%d,%lu\n", nt, (unsigned long)(t1 - t0));
    }

    printf("PROBE_END\n");
    exit(0);
}
