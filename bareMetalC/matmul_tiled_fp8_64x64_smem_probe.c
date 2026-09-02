// INSTRUMENTED probe of matmul_tiled_fp8_64x64_smem_mvout.c — the DMA-mvout variant (no CPU readback).
// Same region brackets as the base probe so the two runs isolate readback-loop vs mvout-DMA cost.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_64x64.h"

#define GEMMINI_SF_MEM 0x40088000
#define GEMMINI_SF_MEM_A (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B GEMMINI_SF_MEM
#define SMEM 0x40000000
#define DIM 16
#define GEMMINI_CTRL 0x40084000
#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)
#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1); \
    *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2); \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) = (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}
#define ADDR_LEN 32
#define BF16_PER_WORD 4
#define OUT_COLS (MATMUL_M / BF16_PER_WORD)
typedef uint8_t  elem_t;
typedef uint8_t  welem_t;
typedef uint64_t out_t;

void load_scale_factors(volatile uint64_t *sf_mem, uint8_t *scale_factors, int INDIM, int K) {
  for (size_t k = 0; k < K/32; k++)
    for (size_t i = 0; i < INDIM / 8; i++)
        sf_mem[k*INDIM/8 + i] = ((uint64_t*) scale_factors)[k * INDIM/8 + i];
}

int main() {
  static out_t C_hw[MATMUL_M][OUT_COLS];
  uint32_t scale_factors[512] = {0};
  memset(C_hw, 0, sizeof(C_hw));
  int tiles_I = MATMUL_M / DIM, tiles_J = MATMUL_N / DIM, tiles_K = MATMUL_K / DIM;
  uint32_t a_base = 0;
  uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;

  uint64_t t0 = read_cycles();
  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
  uint64_t t1 = read_cycles();

  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_A, (uint8_t *) &A_scales_row, MATMUL_M, MATMUL_K);
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_B, (uint8_t *) &B_scales_col, MATMUL_N, MATMUL_K);
  uint64_t t2 = read_cycles();

  gemmini_config_ld(MATMUL_M * sizeof(elem_t));
  for (int i = 0; i < tiles_I; i++)
    for (int k = 0; k < tiles_K; k++) {
      elem_t *dram_ptr = ((elem_t*)A_in) + i * DIM * MATMUL_M + k * DIM;
      gemmini_extended_mvin((void *) dram_ptr, a_base + (i * tiles_K + k) * DIM, DIM, DIM);
    }
  for (int j = 0; j < tiles_J; j++)
    for (int k = 0; k < tiles_K; k++) {
      elem_t *dram_ptr = ((elem_t*)B_in) + j * DIM * MATMUL_M + k * DIM;
      gemmini_extended_mvin((void *) dram_ptr, b_base + (j * tiles_K + k) * DIM, DIM, DIM);
    }
  uint64_t t3 = read_cycles();

  int SPAD_DEST = 0;
  gemmini_config_st(DIM * sizeof(elem_t));
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
      0, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
  gemmini_fence();
  uint64_t t4 = read_cycles();

  for (int i = 0; i < tiles_I; i++)
    for (int j = 0; j < 2*tiles_J; j++) {
      uint32_t spad_addr = SPAD_DEST + (i * 2*tiles_J + j) * DIM;
      out_t *dram_ptr = C_hw + i*4*tiles_J + j*2;
      gemmini_mvout((void *) dram_ptr, spad_addr);
    }
  gemmini_fence();
  uint64_t t5 = read_cycles();

  int mvout_cmds = tiles_I * 2 * tiles_J;
  printf("PROBE M=%d N=%d K=%d mvout_cmds=%d\n", MATMUL_M, MATMUL_N, MATMUL_K, mvout_cmds);
  printf("PROBE_config %lu\n",   (unsigned long)(t1 - t0));
  printf("PROBE_scale %lu\n",    (unsigned long)(t2 - t1));
  printf("PROBE_mvin %lu\n",     (unsigned long)(t3 - t2));
  printf("PROBE_compute %lu\n",  (unsigned long)(t4 - t3));
  printf("PROBE_mvout %lu\n",    (unsigned long)(t5 - t4));
  printf("PROBE_total_from_t0 %lu\n", (unsigned long)(t5 - t0));

  volatile uint64_t sink = 0;
  for (int i = 0; i < MATMUL_M; i++) for (int j = 0; j < OUT_COLS; j++) sink ^= C_hw[i][j];
  printf("PROBE_sink %lx\n", (unsigned long)sink);
  return 0;
}
