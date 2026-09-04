// Calibration sweep: measure Gemmini matmul-region cycles for many shapes in ONE boot.
// Cycles only (no golden check). Data feeds modeling/calibration.
//
// Extended sweep (Phase 3): ~43 warm steady-state shapes + 2 cold-start controlled
// probes. The cold probes run as the FIRST matmuls in the boot (cold icache + cold
// Gemmini config) and are re-run warm in the main sweep, so the ~1041-cy warmup bias
// can be measured directly (COLD vs warm delta for the same shape). All main "SWEEP"
// rows therefore execute in a consistent warm/steady-state regime.
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

// flat max-size buffers; per-shape we pass actual dims+strides. Data is arbitrary
// (we measure cycles, not correctness), so garbage contents are fine.
static elem_t A[MAXD*MAXD] row_align(1);
static elem_t B[MAXD*MAXD] row_align(1);
static elem_t C[MAXD*MAXD] row_align(1);

// Main WARM sweep shapes. First 14 = original Phase-2 sweep (order preserved).
// Remainder = Phase-3 additions: denser mid squares, >128 squares, tall/wide/deep-K
// rectangles, mixed rectangles, and conv-as-GEMM (im2col) shapes.
static const int shapes[][3] = {
  // -- original 14 (order preserved for provenance) --
  {16,16,16},{32,32,32},{48,48,48},{64,64,64},{80,80,80},{96,96,96},{112,112,112},{128,128,128},
  {32,32,128},{128,32,32},{32,128,32},{128,128,64},{64,64,128},{48,96,64},
  // -- denser mid squares (16..128) --
  {24,24,24},{40,40,40},{56,56,56},{72,72,72},{88,88,88},{104,104,104},{120,120,120},
  // -- larger squares (>128, baremetal-memory limit test) --
  {144,144,144},{160,160,160},{192,192,192},
  // -- tall (M >> N,K) --
  {128,16,16},{128,64,64},{160,32,32},{256,32,32},
  // -- wide (N >> M,K) --
  {16,128,16},{64,128,64},{32,160,32},{32,256,32},
  // -- deep-K (K >> M,N) --
  {32,32,64},{16,16,128},{64,64,192},{48,48,128},{32,32,256},
  // -- mixed rectangles --
  {96,48,64},{112,64,80},{80,112,48},
  // -- conv-as-GEMM (im2col: M=out_h*out_w, K=Cin*kh*kw, N=Cout) --
  {64,144,32},   // 3x3 conv, Cin=16 Cout=32, 8x8 out
  {36,72,64},    // 3x3 conv, Cin=8  Cout=64, 6x6 out
  {16,64,128}    // 1x1 conv, Cin=64 Cout=128, 4x4 out
};

// Cold-start controlled probes: run FIRST in the boot (cold), also present in the
// warm sweep above, so warmup bias = COLD(shape) - warm(shape).
static const int cold_shapes[][3] = {
  {64,64,64},{32,32,32}
};

static unsigned long run_shape(int I, int J, int K) {
    unsigned long start = read_cycles();
    tiled_matmul_auto(I, J, K,
            A, B, NULL, C,
            K, J, J, J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false,
            false, !FULL_BIAS_WIDTH,
            0,
            WS);
    unsigned long end = read_cycles();
    return (unsigned long)(end - start);
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall failed"); exit(1); }
#endif
    gemmini_flush(0);
    for (size_t i = 0; i < MAXD*MAXD; ++i) { A[i] = (i*3+1)&0x7f; B[i] = (i*5+2)&0x7f; }

    printf("SWEEP_BEGIN\n");

    // Cold-start controlled probes (first matmuls in the boot => cold).
    int nc = sizeof(cold_shapes)/sizeof(cold_shapes[0]);
    for (int s = 0; s < nc; s++) {
        int I = cold_shapes[s][0], J = cold_shapes[s][1], K = cold_shapes[s][2];
        unsigned long cyc = run_shape(I, J, K);
        printf("COLD,%d,%d,%d,%lu\n", I, J, K, cyc);
    }

    // Warm steady-state sweep.
    int n = sizeof(shapes)/sizeof(shapes[0]);
    for (int s = 0; s < n; s++) {
        int I = shapes[s][0], J = shapes[s][1], K = shapes[s][2];
        unsigned long cyc = run_shape(I, J, K);
        printf("SWEEP,%d,%d,%d,%lu\n", I, J, K, cyc);
    }
    printf("SWEEP_END\n");
    exit(0);
}
