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
//        printf("loading: %lx\n", ((uint64_t*) scale_factors)[k * INDIM/8 + i]);
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
  int tiles_I = MATMUL_M / DIM;
  int tiles_J = MATMUL_N / DIM;
  int tiles_K = MATMUL_K / DIM;

  uint32_t a_base = 0;
  uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
  uint32_t acc_addr = (1u << (ADDR_LEN - 1));

  // ---- Gemmini setup ----
  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);

#ifdef SPIKE_SIM
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
#else
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_A, (uint8_t *) &A_scales_row, MATMUL_M, MATMUL_K);
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_B, (uint8_t *) &B_scales_col, MATMUL_N, MATMUL_K);
#endif

  // ==== CYCLE MEASUREMENT START (accelerator region: mvin + compute + store) ====
  uint64_t _mmt0 = read_cycles();

  // ---- MVIN A: tile (i,k) -> a_base + (i*tiles_K + k)*DIM ----
  gemmini_config_ld(MATMUL_K * sizeof(elem_t));

  for (int i = 0; i < tiles_I; i++) {
    for (int k = 0; k < tiles_K; k++) {
      elem_t *dram_ptr = ((elem_t*)A_in) + i * DIM * MATMUL_K + k * DIM;
      uint32_t sp_addr = a_base + (i * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    }
  }

  gemmini_config_ld(MATMUL_N * sizeof(elem_t));

  // ---- MVIN B: tile (k,j) -> b_base + (j*tiles_K + k)*DIM ----
  for (int k = 0; k < tiles_K; k++) {
    for (int j = 0; j < tiles_J; j++) {
      elem_t *dram_ptr = ((elem_t*)B_in) + k * DIM * MATMUL_N + j * DIM;
      uint32_t sp_addr = b_base + (k * tiles_J + j) * DIM;
//      printf("j: %d, k: %d, sp_addr: %d \n", j, k, sp_addr);
//      printf("first elem: %x \n", dram_ptr[0]);
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, DIM, DIM);
    }
  }

  int SPAD_DEST = 0;

  gemmini_config_st(DIM * sizeof(elem_t));
  // fine-grained MX: same as gemmini_mxquant_config_mvout(scale_factors, tiles_I,J,K, 0,0,1)
  // but with rs1[63]=1 -> selects the carried per-tile scale rows (operand-derived,
  // order-independent) over the loop-order counter FSM in ScaleFactorMem.
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
    ((uint64_t)1 << 63) |
    ((uint64_t)(tiles_K) << 51) | ((uint64_t)(tiles_J) << 42) | ((uint64_t)(tiles_I) << 33) |
    (uint64_t)((uintptr_t)scale_factors),
    (uint64_t)1 & 0xFFFF, CONFIG_SCALE_MEM);

  // ---- Compute: FINE-GRAINED preload/compute (replaces FSM Execute) ----
  // Transcribed from LoopMatmul Execute FSM: loops k-outer, j-mid, i-inner.
  //   A tile (i,k) @ a_base + (i*tiles_K + k)*DIM
  //   B tile (k,j) @ b_base + (k*tiles_J + j)*DIM   (b_base = BANK_NUM*BANK_ROWS - tiles_K*tiles_J*DIM)
  //   C tile (i,j) acc-offset (i*tiles_J + j)*(DIM/4)  [fp8 packed bf16 output, stride DIM/4]
  // PRELOAD: load B into array only at i==0 (weight-stationary reuse across i); else GARBAGE.
  // COMPUTE: FLIP (compute_preloaded) at i==0, STAY (compute_accumulated) for i>0.
  // accumulate bit in C set when k!=0 (K-reduction).
  const uint32_t C_ACC_BIT   = 1u << (ADDR_LEN - 1);  // 0x80000000 : address is accumulator
  const uint32_t C_ACCUM_BIT = 1u << (ADDR_LEN - 2);  // 0x40000000 : accumulate (vs overwrite)
  // REORDERED: j-outer / k-mid / i-inner. The counter-FSM scale addressing CANNOT tolerate
  // this (it assumes k-outer/j-mid order), but the carried per-tile rows make it correct.
  for (int j = 0; j < tiles_J; j++) {
    for (int k = 0; k < tiles_K; k++) {
      for (int i = 0; i < tiles_I; i++) {
        uint32_t a_addr = a_base + (i * tiles_K + k) * DIM;   // (i,k)
        uint32_t b_addr = b_base + (k * tiles_J + j) * DIM;   // (k,j)
        uint32_t c_off  = (i * tiles_J + j) * (DIM / 4);      // fp8 packed stride
        // carried per-tile scale rows == what the counter FSM read_row_addr would produce
        // for this (i,j,k): act = tiles_I*(k>>1)+i, w = tiles_J*(k>>1)+j. Stashed in the
        // c-addr garbage band [25:14] (act_row=[19:14], w_row=[25:20], 6 bits each).
        uint32_t group   = (uint32_t)(k >> 1);
        uint32_t act_row = (uint32_t)tiles_I * group + (uint32_t)i;
        uint32_t w_row   = (uint32_t)tiles_J * group + (uint32_t)j;
        uint32_t c_addr  = C_ACC_BIT | ((k != 0) ? C_ACCUM_BIT : 0u) | c_off
                           | (act_row << 14) | (w_row << 20);
        if (i == 0) {
          gemmini_preload(b_addr, c_addr);
          gemmini_compute_preloaded(a_addr, GARBAGE_ADDR);
        } else {
          gemmini_preload(GARBAGE_ADDR, c_addr);
          gemmini_compute_accumulated(a_addr, GARBAGE_ADDR);
        }
      }
    }
  }

  gemmini_fence();

  // ---- Store: FSM store-only (skips=0x78: mark lda/ldb/ldd/ex done, run only ST) ----
  // Step A isolates compute correctness; Step B replaces this with explicit mvout_spad.
  gemmini_loop_ws_spad(
      tiles_I, tiles_J, tiles_K,
      0, 0, 0,
      a_base,
      BANK_NUM * BANK_ROWS,
      0,
      SPAD_DEST,
      false, false,
      false, false, false,
      NO_ACTIVATION,
      0, 0,
      false,
      0x78);

  gemmini_fence();

  // ==== CYCLE MEASUREMENT END ====
  uint64_t _mmt1 = read_cycles();
  printf("MMCYCLES %lu\n", (unsigned long)(_mmt1 - _mmt0));

//  uint64_t* smem_addr = ((uint64_t*)SMEM) + SPAD_DEST * 2 + 448 / 8;
//  printf("SMEM at address %x = %lx \n", ((uint64_t*)SMEM), *((uint64_t*)SMEM));
//  printf("SMEM at address %x = %lx \n", smem_addr, *smem_addr);
//  printf("SMEM at address %x = %lx \n", smem_addr + 2, *(smem_addr+2));


#ifdef SPIKE_SIM
  gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N);
#else
  printf("Moving out:\n");
  for (int i = 0; i < tiles_J*tiles_I*2; i++) {
      gemmini_mvout((void*)((uint64_t*) C_hw + i*2*DIM), SPAD_DEST + i*DIM);
  }
  gemmini_fence();
#endif

  // ---- Debug print tile (0,0) ----
//  printf("=== Tile (0,0) - acc_addr=0x%08x ===\n", acc_addr);
//  for (int i = 0; i < DIM; i++) {
//    for (int j = 0; j < DIM; j++) {
//      printf("C_hw[%d][%d] = 0x%016lx\n", i, j, (unsigned long)C_hw[i][j]);
//    }
//  }

  // ---- Elementwise check against C_out_bf16 ----
  int errors = 0;

  for (int i = 0; i < MATMUL_M; i++) {
    for (int j = 0; j < OUT_COLS; j++) {
      uint64_t got = C_hw[i][j];

      // Pack 4 consecutive bf16 golden values into expected uint64_t
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