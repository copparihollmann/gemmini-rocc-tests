#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif

#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_128x128x256.h"

#define GEMMINI_SF_MEM 0x40088000
#define GEMMINI_SF_MEM_A (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B GEMMINI_SF_MEM
#define SMEM 0x40000000

#define DIM 16

#define GEMMINI_CTRL 0x40084000
#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)
#define GEMMINI_BUSY_ADDR (GEMMINI_CTRL + 0x20)

#ifndef SPIKE_SIM
#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1); \
    *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2); \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) = (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}
#endif

#define ADDR_LEN 32

// BF16 values packed 4 per uint64_t output word
#define BF16_PER_WORD 4
#define OUT_COLS (MATMUL_M / BF16_PER_WORD)

typedef uint8_t  elem_t;   // fp8:e4m3 input elements
typedef uint8_t  welem_t;  // fp8:e4m3 weight elements
typedef uint64_t out_t;    // 4x bf16 packed per word

// ---- Scale factor loader ----
void load_scale_factors(volatile uint64_t *sf_mem, uint8_t *scale_factors, int INDIM, int K) {
  for (size_t k = 0; k < K/32; k++) {
    for (size_t i = 0; i < INDIM / 8; i++) {
        sf_mem[k*INDIM/8 + i] = ((uint64_t*) scale_factors)[k * INDIM/8 + i];
    }
  }
}

#ifndef SPIKE_SIM
#undef gemmini_fence
#define gemmini_fence() { while (*((volatile uint32_t *) GEMMINI_BUSY_ADDR)) asm volatile ("nop"); }
#endif

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall");
    return 1;
  }
#endif

  // ---- Output buffer ----
  static out_t C_hw[MATMUL_M][OUT_COLS];
  uint32_t scale_factors[512] = {0};
  memset(C_hw, 0, sizeof(C_hw));

  // ---- Tile dimensions ----
  int tiles_I = MATMUL_M / DIM;   // 8
  int tiles_J = MATMUL_N / DIM;   // 8
  int tiles_K = MATMUL_K / DIM;   // 16

  // ---- DOUBLE-BUFFER over the K dimension ----
  // Split tiles_K into two halves. Each half lives in its own scratchpad
  // bank region so the Load and Execute controllers can run concurrently:
  //   half 0 loads -> EX(half 0) runs while half 1 loads -> EX(half 1) accumulates.
  // The 2-deep loop_ws_spad slot queue (concurrent_loops == 2 in RTL) lets
  // the second loop's mvins overlap the first loop's compute when they touch
  // disjoint banks.
  int K_half = tiles_K / 2;       // 8 tiles each

  // Per-half footprints (rows of the 16-wide scratchpad):
  //   A half: tiles_I * K_half * DIM = 8*8*16 = 1024 rows
  //   B half: K_half * tiles_J * DIM = 8*8*16 = 1024 rows
  uint32_t A_half_rows = tiles_I * K_half * DIM;   // 1024
  uint32_t B_half_rows = K_half * tiles_J * DIM;   // 1024

  // A region grows up from 0; two halves back-to-back: [0,1024), [1024,2048)
  uint32_t a_base[2];
  a_base[0] = 0;
  a_base[1] = A_half_rows;

  // B region grows down from top of spad; two halves back-to-back below it:
  //   bank top = BANK_NUM*BANK_ROWS = 8192
  //   half 0 occupies [8192-1024, 8192) = [7168,8192)
  //   half 1 occupies [8192-2048, 8192-1024) = [6144,7168)
  // The EX iterator derives the physical B start as (B_arg - K_half*tiles_J*DIM),
  // so B_arg is the TOP of each half's region.
  uint32_t b_top[2];
  b_top[0] = (uint32_t)(BANK_NUM * BANK_ROWS);          // 8192
  b_top[1] = (uint32_t)(BANK_NUM * BANK_ROWS) - B_half_rows; // 7168

  int SPAD_DEST = 0;

  // ---- Gemmini setup ----
  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
  uint64_t _mmt0 = read_cycles();

#ifdef SPIKE_SIM
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
#else
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_A, (uint8_t *) &A_scales_row, MATMUL_M, MATMUL_K);
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_B, (uint8_t *) &B_scales_col, MATMUL_N, MATMUL_K);
#endif

  // ---- Helper: mvin A/B for one K-half into its bank region ----
  // (inlined below so config_ld ordering is explicit)

  gemmini_config_st(DIM * sizeof(elem_t));
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, tiles_K);

  // ===================== HALF 0 =====================
  // MVIN A (half 0): K tiles [0, K_half)
  gemmini_config_ld(MATMUL_K * sizeof(elem_t));
  for (int i = 0; i < tiles_I; i++) {
    for (int k = 0; k < K_half; k += 2) {
      int batch = (K_half - k < 2) ? (K_half - k) : 2;
      elem_t *dram_ptr = ((elem_t*)A_in) + i * DIM * MATMUL_K + (0 * K_half + k) * DIM;
      uint32_t sp_addr = a_base[0] + (i * K_half + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, batch * DIM, DIM);
    }
  }
  // MVIN B (half 0): K tiles [0, K_half)
  gemmini_config_ld(MATMUL_N * sizeof(elem_t));
  uint32_t b_base0 = b_top[0] - B_half_rows; // physical start of half-0 B region
  for (int k = 0; k < K_half; k++) {
    for (int j = 0; j < tiles_J; j += 2) {
      int batch = (tiles_J - j < 2) ? (tiles_J - j) : 2;
      elem_t *dram_ptr = ((elem_t*)B_in) + (0 * K_half + k) * DIM * MATMUL_N + j * DIM;
      uint32_t sp_addr = b_base0 + (k * tiles_J + j) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, batch * DIM, DIM);
    }
  }

  // Compute half 0 (no fence after the mvins above -> the issue below overlaps
  // with the half-1 mvins issued right after it). ex_accumulate=false: this
  // half WRITES C (k==0 of this loop overwrites the accumulator tiles).
  gemmini_loop_ws_spad(
      tiles_I, tiles_J, K_half,
      0, 0, 0,
      a_base[0],
      b_top[0],
      0,
      SPAD_DEST,
      false, false,
      false, false, false,   // full_C, low_D, ex_accumulate=false
      NO_ACTIVATION,
      0, 0,
      false,
      0x38);

  // ===================== HALF 1 =====================
  // MVIN A (half 1): K tiles [K_half, tiles_K). Issued WITHOUT a fence so the
  // Load controller runs these while EX(half 0) is still draining.
  gemmini_config_ld(MATMUL_K * sizeof(elem_t));
  for (int i = 0; i < tiles_I; i++) {
    for (int k = 0; k < K_half; k += 2) {
      int batch = (K_half - k < 2) ? (K_half - k) : 2;
      elem_t *dram_ptr = ((elem_t*)A_in) + i * DIM * MATMUL_K + (1 * K_half + k) * DIM;
      uint32_t sp_addr = a_base[1] + (i * K_half + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, batch * DIM, DIM);
    }
  }
  // MVIN B (half 1): K tiles [K_half, tiles_K)
  gemmini_config_ld(MATMUL_N * sizeof(elem_t));
  uint32_t b_base1 = b_top[1] - B_half_rows; // physical start of half-1 B region
  for (int k = 0; k < K_half; k++) {
    for (int j = 0; j < tiles_J; j += 2) {
      int batch = (tiles_J - j < 2) ? (tiles_J - j) : 2;
      elem_t *dram_ptr = ((elem_t*)B_in) + (1 * K_half + k) * DIM * MATMUL_N + j * DIM;
      uint32_t sp_addr = b_base1 + (k * tiles_J + j) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, batch * DIM, DIM);
    }
  }

  // Compute half 1: ex_accumulate=TRUE so its k==0 step ACCUMULATES onto the
  // C tiles written by half 0 (C = sum over all K of A*B).
  gemmini_loop_ws_spad(
      tiles_I, tiles_J, K_half,
      0, 0, 0,
      a_base[1],
      b_top[1],
      0,
      SPAD_DEST,
      false, false,
      false, false, true,    // full_C, low_D, ex_accumulate=true
      NO_ACTIVATION,
      0, 0,
      false,
      0x38);

  gemmini_fence();

  uint64_t _mmt1 = read_cycles();
  printf("MMCYCLES %lu\n", (unsigned long)(_mmt1 - _mmt0));
#ifdef SPIKE_SIM
  gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N);
#else
  printf("Moving out:\n");
  for (int i = 0; i < tiles_J*tiles_I*2; i++) {
      gemmini_mvout((void*)((uint64_t*) C_hw + i*2*DIM), SPAD_DEST + i*DIM);
  }
  gemmini_fence();
#endif

  // ---- Elementwise check against C_out_bf16 ----
  int errors = 0;

  for (int i = 0; i < MATMUL_M; i++) {
    for (int j = 0; j < OUT_COLS; j++) {
      uint64_t got = C_hw[i][j];

      uint64_t exp = ((uint64_t)C_out_bf16[i][j*4 + 3] << 48) |
                     ((uint64_t)C_out_bf16[i][j*4 + 2] << 32) |
                     ((uint64_t)C_out_bf16[i][j*4 + 1] << 16) |
                     ((uint64_t)C_out_bf16[i][j*4 + 0]);

      if (got != exp) {
        for (int lane = 0; lane < BF16_PER_WORD; lane++) {
          uint16_t got_bf16 = (got >> (lane * 16)) & 0xFFFF;
          uint16_t exp_bf16 = C_out_bf16[i][j * BF16_PER_WORD + lane];
          if (got_bf16 != exp_bf16) {
            printf("MISMATCH @(%d,%d) HW=0x%04x EXP=0x%04x\n",
                   i, j * BF16_PER_WORD + lane, got_bf16, exp_bf16);
            errors++;
          }
        }
      }
    }
  }

  if (errors == 0) {
    printf("fp8 WS matmul test PASSED (no mismatches).\n");
  } else {
    printf("fp8 WS matmul test FAILED with %d mismatches.\n", errors);
  }

#ifndef BAREMETAL
  exit(errors != 0);
#else
  return errors != 0;
#endif
}
