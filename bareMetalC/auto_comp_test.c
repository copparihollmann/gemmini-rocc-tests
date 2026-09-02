#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif
#include "include/gemmini_testutils.h"
#include "include/mxgen_fp8_256x256x512.h"

#define DIM 16
#define BLK 64
#define BLK_OUTC (BLK / 4)
#define BLOCKS_M (MATMUL_M / BLK)
#define BLOCKS_N (MATMUL_N / BLK)
#define BLOCKS_K (MATMUL_K / BLK)
#define BF16_PER_WORD 4
#define OUT_COLS (MATMUL_N / BF16_PER_WORD)
// MATMUL_M/K/N are PADDED to BLK (M,N) / DIM (K); REAL_* are the true dims. The block loop
// covers padded blocks; the last M/N block is partial and uses loop_ws pad to skip the rest.
#define REAL_M 256
#define REAL_N 512
typedef uint8_t  elem_t;
typedef uint64_t out_t;
#ifndef fence
#define fence() gemmini_fence()
#endif

// All DMA-visible buffers at FIXED DRAM addresses so the capture and the
// optimization harness use bit-identical addresses regardless of binary layout
// (the spike MX model's behavior depends on buffer addresses).
#define A_blk          ((elem_t (*)[MATMUL_K])      0xA0000000UL)
#define B_blk          ((elem_t (*)[MATMUL_K])      0xA0100000UL)
#define A_s            ((uint8_t (*)[BLK])          0xA0200000UL)
#define B_s            ((uint8_t (*)[BLK])          0xA0280000UL)
#define C_blk          ((out_t (*)[BLK_OUTC])       0xA0300000UL)
#define scale_factors  ((uint32_t *)                0xA0400000UL)
#define C_acc          ((float (*)[MATMUL_N])       0xA0800000UL)
#define C_hw           ((out_t (*)[OUT_COLS])       0xA1000000UL)  // bf16-packed row-major output
#define SCALE_FACTORS_BYTES (1 << 22)
#define A_S_BYTES   ((MATMUL_K / 32) * BLK)
#define B_S_BYTES   ((MATMUL_K / 32) * BLK)   // requantizer scale stream; advances per launch -- keep large + zeroed

static inline float bf16f(uint16_t b) {
  union { uint32_t u; float f; } v; v.u = ((uint32_t)b) << 16; return v.f;
}
static inline uint32_t fbits(float f) {
  union { uint32_t u; float f; } v; v.f = f; return v.u;
}

#define OUTPUT_MATRIX_NAME C_hw
// GOLD_FILE: /scratch/agustin/projects/autocomp/harnesses/real-pi05-ffn/gold0.txt
int full_is_equal(out_t x[MATMUL_M][OUT_COLS], const out_t y[MATMUL_M][OUT_COLS]) {
  (void)x; (void)y; return 1;  // correctness checked vs GOLD_FILE by the runner
}
static out_t gold[1][1];

#define REPEAT_TEST_ITERS 1
#define RUN_BASELINE_CODE 0

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); return 1; }
#endif
  memset(C_hw, 0, MATMUL_M * OUT_COLS * sizeof(out_t));
  memset(scale_factors, 0, SCALE_FACTORS_BYTES);
  // SUBSTITUTE HERE
    int generated_implementation_start_cycle;
int generated_implementation_end_cycle;
    gemmini_flush(0);
    fence();
    generated_implementation_start_cycle = read_cycles();
    {
    
  // Pre-format scale factor matrices to eliminate CPU-side staging in the hot loop
  static uint8_t A_scales_perf[BLOCKS_M][MATMUL_K / 32][BLK];
  static uint8_t B_scales_perf[BLOCKS_N][MATMUL_K / 32][BLK];

  // One-time pre-formatting before main loops start
  for (int i0 = 0; i0 < BLOCKS_M; i0++) {
    for (int g = 0; g < MATMUL_K / 32; g++) {
      int mrows = REAL_M - i0 * BLK;
      if (mrows > BLK) mrows = BLK;
      if (mrows <= 0) continue;
      
      // Only copy actual data, handle partial blocks correctly
      for (int i = 0; i < mrows; i++) {
        A_scales_perf[i0][g][i] = A_scales_row[g][i0 * BLK + i];
      }
      // Pad with zeros for partial blocks if needed
      for (int i = mrows; i < BLK; i++) {
        A_scales_perf[i0][g][i] = 0;
      }
    }
  }

  for (int j0 = 0; j0 < BLOCKS_N; j0++) {
    for (int g = 0; g < MATMUL_K / 32; g++) {
      int ncols = REAL_N - j0 * BLK;
      if (ncols > BLK) ncols = BLK;
      if (ncols <= 0) continue;
      
      // Only copy actual data, handle partial blocks correctly
      for (int j = 0; j < ncols; j++) {
        B_scales_perf[j0][g][j] = B_scales_col[g][j0 * BLK + j];
      }
      // Pad with zeros for partial blocks if needed
      for (int j = ncols; j < BLK; j++) {
        B_scales_perf[j0][g][j] = 0;
      }
    }
  }

  // Hoist invariant Gemmini execution configuration outside all loops
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);

  const elem_t *A_row_base = (const elem_t*)A_in;
  const uint8_t *A_scale_ptr = &A_scales_perf[0][0][0];
  uint64_t (*C_row_block_base)[MATMUL_N/BF16_PER_WORD] = &C_hw[0];

  for (int i0 = 0; i0 < BLOCKS_M; i0++) {
    int mrows = REAL_M - i0 * BLK; 
    if (mrows > BLK) mrows = BLK; 
    if (mrows <= 0) continue;

    int tiles = BLK / DIM, tiles_K = MATMUL_K / DIM;
    uint32_t a_base = 0;

    // Load stationary A scale factors once per row block
    gemmini_mx_load_scales((uint64_t)A_scale_ptr, A_S_BYTES, 0);

    // Keep A tiles stationary in the lower half of the scratchpad
    gemmini_config_ld(MATMUL_K * sizeof(elem_t)); // A rows: stride K
    const elem_t *A_tile_ptr = A_row_base;
    for (int i = 0; i < tiles; i++) {
      const elem_t *A_tile_ptr_inner = A_tile_ptr;
      for (int k = 0; k < tiles_K; k += 2) {
        gemmini_extended_mvin((void*)A_tile_ptr_inner,
                              a_base + (i * tiles_K + k) * DIM, 2 * DIM, DIM);
        A_tile_ptr_inner += 2 * DIM;
      }
      A_tile_ptr += DIM * MATMUL_K;
    }

    const elem_t *B_col_base = (const elem_t*)B_in;
    const uint8_t *B_scale_ptr = &B_scales_perf[0][0][0];
    int C_col_offset_words = 0;

    for (int j0 = 0; j0 < BLOCKS_N; j0++) {
      int ncols = REAL_N - j0 * BLK; 
      if (ncols > BLK) ncols = BLK; 
      if (ncols <= 0) continue;

      uint32_t b_base = BANK_NUM * BANK_ROWS - tiles * tiles_K * DIM;
      int SPAD_DEST = 128;
      int pad_i = BLK - mrows, pad_j = BLK - ncols;

      // Flush to synchronize state
      gemmini_flush(0);

      // Load dynamic B scale factors
      gemmini_mx_load_scales((uint64_t)B_scale_ptr, B_S_BYTES, 1);

      // Load dynamic B tiles into the upper half of the scratchpad
      gemmini_config_ld(MATMUL_N * sizeof(elem_t)); // B rows: stride N
      const elem_t *B_tile_ptr = B_col_base;
      for (int k = 0; k < tiles_K; k++) {
        const elem_t *B_tile_ptr_inner = B_tile_ptr;
        for (int j = 0; j < tiles; j += 2) {
          gemmini_extended_mvin((void*)B_tile_ptr_inner,
                                b_base + (k * tiles + j) * DIM, 2 * DIM, DIM);
          B_tile_ptr_inner += 2 * DIM;
        }
        B_tile_ptr += DIM * MATMUL_N;
      }

      gemmini_config_st(BLK_OUTC * sizeof(out_t));
      gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles, tiles, tiles_K, 0, 0, 1);
      
      // Perform computation utilizing the stationary A tiles and new B tiles
      gemmini_loop_ws_spad(tiles, tiles, tiles_K, pad_i, pad_j, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                           SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);

      // Drain results to DRAM using strength-reduced pointer arithmetic
      uint64_t *C_drain_ptr = (uint64_t*)((char*)C_row_block_base + C_col_offset_words * sizeof(uint64_t));
      uint32_t smem_offset = SPAD_DEST * 16;
      for (int r = 0; r < mrows; r++) {
        gemmini_mx_read_smem(C_drain_ptr, smem_offset, ncols);
        C_drain_ptr += (MATMUL_N / BF16_PER_WORD);
        smem_offset += BLK;
      }
      gemmini_fence();

      B_col_base += BLK;
      B_scale_ptr += (MATMUL_K / 32) * BLK;
      C_col_offset_words += BLK / BF16_PER_WORD;
    }

    A_row_base += BLK * MATMUL_K;
    A_scale_ptr += (MATMUL_K / 32) * BLK;
    C_row_block_base += BLK;
  }

    }
    fence();
    generated_implementation_end_cycle = read_cycles();
    printf("Generated implementation latency: %d cycles\n", generated_implementation_end_cycle - generated_implementation_start_cycle);

  // SUBSTITUTE END
  printf("GOLD_BEGIN\n");
  for (int i = 0; i < REAL_M; i++)                 // dump only the REAL sub-block: padding writes by a
    for (int j = 0; j < REAL_N / BF16_PER_WORD; j++)  // candidate can't cause a false mismatch
      printf("%08x\n%08x\n", (unsigned)(C_hw[i][j] >> 32), (unsigned)(C_hw[i][j] & 0xffffffffu));
  printf("GOLD_END\n");
  printf("Correct result\n");
#ifndef BAREMETAL
  exit(0);
#else
  return 0;
#endif
}