// INT8 accelerator DRAM-byte probe (gemmini_sweep clone + DMA-byte counters) — M3 byte-model validation.
// Byte-identical to the known-good gemmini_sweep.c matmul, PLUS RDMA_BYTES_REC / WDMA_BYTES_SENT hardware
// counters, which measure the accelerator's OWN DRAM read/write bytes (DMA.scala total_bytes_read/sent) and
// EXCLUDE host instruction fetch + CPU readback — the exact M3 (accelerator data-movement) quantity, unlike
// the full-SoC SimDRAM counter. If operand footprint (M*K + K*N read, M*N written) matches these, the byte
// model is validated and the same oracle definition transfers to the MX corpus.
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define MAXD 256
#define FULL_BIAS_WIDTH 1
static elem_t A[MAXD*MAXD] row_align(1);
static elem_t B[MAXD*MAXD] row_align(1);
static elem_t C[MAXD*MAXD] row_align(1);

static const int shapes[][3] = {
    {16,16,16},{32,32,32},{64,64,64},{128,128,128},
    {128,32,32},{32,32,128},{256,32,32},{128,128,256},
};

static void run_shape(int I, int J, int K) {
    tiled_matmul_auto(I, J, K, A, B, NULL, C, K, J, J, J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false, false, false, false, !FULL_BIAS_WIDTH, 0, WS);
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall failed"); exit(1); }
#endif
    gemmini_flush(0);
    for (size_t i = 0; i < MAXD*MAXD; ++i) { A[i] = (i*3+1)&0x7f; B[i] = (i*5+2)&0x7f; }

    counter_configure(0, RDMA_BYTES_REC);
    counter_configure(1, WDMA_BYTES_SENT);
    counter_configure(2, RDMA_ACTIVE_CYCLE);
    counter_configure(3, WDMA_ACTIVE_CYCLE);
    gemmini_fence();

    printf("DMA_BEGIN\n");
    for (size_t s = 0; s < sizeof(shapes)/sizeof(shapes[0]); ++s) {
        int I = shapes[s][0], J = shapes[s][1], K = shapes[s][2];
        gemmini_fence();
        uint32_t p0 = counter_read(0), p1 = counter_read(1), p2 = counter_read(2), p3 = counter_read(3);
        run_shape(I, J, K);
        gemmini_fence();
        uint32_t rb = counter_read(0)-p0, wb = counter_read(1)-p1, ra = counter_read(2)-p2, wa = counter_read(3)-p3;
        printf("DMA,%d,%d,%d,%u,%u,%u,%u\n", I, J, K, rb, wb, ra, wa);
    }
    printf("DMA_END\n");
    exit(0);
}
