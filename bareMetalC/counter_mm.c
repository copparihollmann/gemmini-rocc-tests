// Counter-instrumented gemmini_sweep CLONE — ModeLIR gemmini INT8 M2 whole-kernel overlap grounding.
// Byte-identical to the known-good gemmini_sweep.c matmul (same square MAXD buffers, same init, same
// tiled_matmul_auto(WS) call that generated calibration/data/gemmini_rtl_sweep_verilator.csv), PLUS the 7
// MAIN_* Gemmini perf counters read as per-shape deltas at idle. The MAIN_* counters decompose mesh-active
// cycles by which queues overlap: LD/ST/EX (only that queue) and LD_ST/LD_EX/ST_EX/LD_ST_EX (concurrent =
// HIDDEN dma). total(read_cycles) - sum = idle/host. Also prints whole-kernel cycles as a cross-check.
// COUNTER DISCIPLINE: counter_read is a synchronous RoCC read (core stalls for the response), so reads at
// idle (post-fence) cannot race the matmul config; configure ONCE, no per-shape counter_reset.
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define MAXD 256
#define NO_BIAS 1
#define FULL_BIAS_WIDTH 1
static elem_t A[MAXD*MAXD] row_align(1);
static elem_t B[MAXD*MAXD] row_align(1);
static elem_t C[MAXD*MAXD] row_align(1);

// regime-spanning subset of the gemmini_sweep shapes (square, tall-M, tall-N, deep-K, memory-bound tall)
static const int shapes[][3] = {
    {16,16,16},{32,32,32},{64,64,64},{128,128,128},
    {128,32,32},{32,128,32},{32,32,128},{256,32,32},{128,128,256},{64,64,128},
};
static const int codes[7] = {
    MAIN_LD_CYCLES, MAIN_ST_CYCLES, MAIN_EX_CYCLES,
    MAIN_LD_ST_CYCLES, MAIN_LD_EX_CYCLES, MAIN_ST_EX_CYCLES, MAIN_LD_ST_EX_CYCLES,
};

static unsigned long run_shape(int I, int J, int K) {   // byte-identical to gemmini_sweep.c run_shape
    unsigned long start = read_cycles();
    tiled_matmul_auto(I, J, K, A, B, NULL, C, K, J, J, J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false, false, false, false, !FULL_BIAS_WIDTH, 0, WS);
    unsigned long end = read_cycles();
    return (unsigned long)(end - start);
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall failed"); exit(1); }
#endif
    gemmini_flush(0);
    for (size_t i = 0; i < MAXD*MAXD; ++i) { A[i] = (i*3+1)&0x7f; B[i] = (i*5+2)&0x7f; }

    for (int c = 0; c < 7; c++) counter_configure(c, codes[c]);
    gemmini_fence();

    printf("CTR_BEGIN\n");
    for (size_t s = 0; s < sizeof(shapes)/sizeof(shapes[0]); ++s) {
        int I = shapes[s][0], J = shapes[s][1], K = shapes[s][2];
        gemmini_fence();
        uint32_t pre[7];
        for (int c = 0; c < 7; c++) pre[c] = counter_read(c);
        unsigned long cyc = run_shape(I, J, K);
        uint32_t v[7];
        for (int c = 0; c < 7; c++) v[c] = counter_read(c) - pre[c];
        printf("CTR,%d,%d,%d,%lu,%u,%u,%u,%u,%u,%u,%u\n", I, J, K, cyc,
               v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
    }
    printf("CTR_END\n");
    exit(0);
}
