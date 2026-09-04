// Phase-bracketed manual WS INT8 matmul (ModeLIR gemmini M2 overlap grounding).
// Measures config / mvin / compute / mvout as SEQUENTIAL fenced regions (each fully
// realized, no double-buffer overlap) for the square tile sweep, so the exposed-DMA of
// the OVERLAPPED auto-kernel (the oracle) can be grounded as a function of the measured
// unoverlapped phase costs. Cycles are data-independent, so buffers hold garbage.
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define GARBAGE_ADDR ((uint32_t)(-1))
#define MAXTILE 8                                  // up to 8x8x8 tiles = 128^3
static elem_t Adram[MAXTILE*MAXTILE][DIM][DIM] row_align(1);
static elem_t Bdram[MAXTILE*MAXTILE][DIM][DIM] row_align(1);
static elem_t Cdram[MAXTILE*MAXTILE][DIM][DIM] row_align(1);

// square tile counts tI=tJ=tK=t → M=N=K=16t
static const int tvals[] = {1, 2, 3, 4, 5, 6, 7, 8};

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall failed"); exit(1); }
#endif
    gemmini_flush(0);
    for (size_t x = 0; x < MAXTILE*MAXTILE; ++x)
        for (size_t i = 0; i < DIM; ++i)
            for (size_t j = 0; j < DIM; ++j) { Adram[x][i][j]=(i+j+x)&0x7f; Bdram[x][i][j]=(i*2+x)&0x7f; }

    printf("PHASE_BEGIN\n");

    for (size_t s = 0; s < sizeof(tvals)/sizeof(tvals[0]); ++s) {
        int t = tvals[s];
        int tI=t, tJ=t, tK=t;
        // scratchpad layout: A tiles [0..tI*tK), B tiles after, C in accumulator
        uint32_t a_base = 0;
        uint32_t b_base = (uint32_t)(tI*tK*DIM);
        uint32_t acc = (1u << (ADDR_LEN - 1));      // accumulator region bit

        // --- config ---
        gemmini_fence();
        uint64_t c0 = read_cycles();
        gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, false, false);
        gemmini_config_ld(DIM * sizeof(elem_t));
        gemmini_config_st(DIM * sizeof(elem_t));
        gemmini_fence();
        uint64_t c1 = read_cycles();

        // --- mvin: A (tI*tK) + B (tK*tJ) tiles ---
        for (int i = 0; i < tI; i++)
            for (int k = 0; k < tK; k++)
                gemmini_mvin(Adram[i*tK+k], a_base + (uint32_t)(i*tK+k)*DIM);
        for (int k = 0; k < tK; k++)
            for (int j = 0; j < tJ; j++)
                gemmini_mvin(Bdram[k*tJ+j], b_base + (uint32_t)(k*tJ+j)*DIM);
        gemmini_fence();
        uint64_t c2 = read_cycles();

        // --- compute: for each output tile (i,j), accumulate over k (WS: preload B, stream A) ---
        for (int i = 0; i < tI; i++)
            for (int j = 0; j < tJ; j++) {
                uint32_t c_addr = acc | (uint32_t)(i*tJ+j)*DIM;
                for (int k = 0; k < tK; k++) {
                    uint32_t b_addr = b_base + (uint32_t)(k*tJ+j)*DIM;
                    uint32_t a_addr = a_base + (uint32_t)(i*tK+k)*DIM;
                    uint32_t out = (k == 0) ? c_addr : (c_addr | (uint32_t)(1u<<30)); // accumulate for k>0
                    gemmini_preload(b_addr, out);
                    if (k == 0) { gemmini_compute_preloaded(a_addr, GARBAGE_ADDR); }
                    else        { gemmini_compute_accumulated(a_addr, GARBAGE_ADDR); }
                }
            }
        gemmini_fence();
        uint64_t c3 = read_cycles();

        // --- mvout: C tiles (tI*tJ) from accumulator to DRAM ---
        for (int i = 0; i < tI; i++)
            for (int j = 0; j < tJ; j++)
                gemmini_mvout(Cdram[i*tJ+j], (acc | (uint32_t)(i*tJ+j)*DIM));
        gemmini_fence();
        uint64_t c4 = read_cycles();

        printf("PHASE,%d,%lu,%lu,%lu,%lu\n", 16*t,
               (unsigned long)(c1-c0), (unsigned long)(c2-c1),
               (unsigned long)(c3-c2), (unsigned long)(c4-c3));
    }
    printf("PHASE_END\n");
    exit(0);
}
