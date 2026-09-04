// Held-out ladder-validation sweep (G2 capstone).
// Reproduces the EXACT calibration boot from calibration/data/gemmini_sweep.c
// (cold-start probes first, then the full 43-shape warm ramp) so the machine
// reaches the same warm/steady-state regime as the calibration set, THEN runs
// 3 held-out shapes NOT present in gemmini_rtl_sweep.csv, tagged HELD,M,N,K,cyc.
//
// Same region definition as the calibration set: read_cycles() brackets one
// tiled_matmul_auto (WS, INT8), cycles only, no golden check. This makes the
// HELD measurements apples-to-apples with the calibration rtl_cycles.
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

// Warm ramp: identical to the calibration sweep's warm shape list (43 shapes).
// Present only to drive the machine into the same steady-state regime; the
// held-out measurements come from held_shapes[] below.
static const int shapes[][3] = {
  {16,16,16},{32,32,32},{48,48,48},{64,64,64},{80,80,80},{96,96,96},{112,112,112},{128,128,128},
  {32,32,128},{128,32,32},{32,128,32},{128,128,64},{64,64,128},{48,96,64},
  {24,24,24},{40,40,40},{56,56,56},{72,72,72},{88,88,88},{104,104,104},{120,120,120},
  {144,144,144},{160,160,160},{192,192,192},
  {128,16,16},{128,64,64},{160,32,32},{256,32,32},
  {16,128,16},{64,128,64},{32,160,32},{32,256,32},
  {32,32,64},{16,16,128},{64,64,192},{48,48,128},{32,32,256},
  {96,48,64},{112,64,80},{80,112,48},
  {64,144,32},{36,72,64},{16,64,128}
};

// Cold-start controlled probes: run FIRST in the boot (cold) to absorb the
// one-shot ~550-600cy warmup penalty, exactly as the calibration boot did.
static const int cold_shapes[][3] = {
  {64,64,64},{32,32,32}
};

// HELD-OUT ladder-validation shapes (min-dim >= 32, absent from calibration CSV).
static const int held_shapes[][3] = {
  {96,160,224},{208,208,208},{128,256,64}
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

    // Warm ramp (same shapes as the calibration warm sweep) => reach steady state.
    int n = sizeof(shapes)/sizeof(shapes[0]);
    for (int s = 0; s < n; s++) {
        int I = shapes[s][0], J = shapes[s][1], K = shapes[s][2];
        unsigned long cyc = run_shape(I, J, K);
        printf("SWEEP,%d,%d,%d,%lu\n", I, J, K, cyc);
    }

    // Held-out ladder-validation shapes, measured deep in the warm regime.
    int nh = sizeof(held_shapes)/sizeof(held_shapes[0]);
    for (int s = 0; s < nh; s++) {
        int I = held_shapes[s][0], J = held_shapes[s][1], K = held_shapes[s][2];
        unsigned long cyc = run_shape(I, J, K);
        printf("HELD,%d,%d,%d,%lu\n", I, J, K, cyc);
    }
    printf("SWEEP_END\n");
    exit(0);
}
