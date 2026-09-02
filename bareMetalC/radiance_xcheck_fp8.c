// Cross-check harness: run REAL spike libgemmini's MX FP8 matmul over the exact
// operands the radiance `gemm_mxgemmini` kernel uses, and print the bf16 result.
//
// Purpose: prove cyclotron's MX-Gemmini functional co-model is bit-identical to spike
// on a real kernel's data. Cyclotron drives the accelerator through the Muon MMIO path
// and DMAs A/B itself; here we hand spike the same operands via explicit mvin, using the
// same tile mapping (A tile (i,k) -> a_base+(i*TK+k)*DIM, B tile (k,j) -> b_base+(k*TJ+j)*DIM)
// that spike's mx_loop_ws_spad reads back.
//
// Build (from gemmini-rocc-tests/build_spike/bareMetalC), run under
//   spike --extension=gemmini radiance_xcheck_fp8-baremetal
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"

#define DIM 16
#define ADDR_LEN 32
#define MATMUL_M 64
#define MATMUL_N 64
#define MATMUL_K 64
#define MATMUL_GK (MATMUL_K / 32)
#define MATMUL_GN (MATMUL_N / 32)

// The radiance kernel's data: A_in[M][K], B_in[K][N], A_scales_row[GK][M],
// B_scales_col[GK][N], and the (python-model) C_out_bf16 reference.
#include "mxgemm.data.fp8.m64n64k64.h"

typedef uint8_t elem_t_fp8;

int main() {
  static uint16_t C_hw[MATMUL_M][MATMUL_N];
  uint32_t scale_factors[512] = {0};
  memset(C_hw, 0, sizeof(C_hw));

  const int tiles_I = MATMUL_M / DIM;
  const int tiles_J = MATMUL_N / DIM;
  const int tiles_K = MATMUL_K / DIM;

  const uint32_t a_base = 0;
  const uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
  const int SPAD_DEST = 256;

  gemmini_flush(0);
  // fp8 in / fp8 weights / full (bf16) out  -> same as the passing fp8 test
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0,
                              false, 0, 0, 3, 0);

  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);

  // ---- MVIN A: tile (i,k) -> a_base + (i*tiles_K + k)*DIM ; row stride = K ----
  gemmini_config_ld(MATMUL_K * sizeof(elem_t_fp8));
  for (int i = 0; i < tiles_I; i++)
    for (int k = 0; k < tiles_K; k++) {
      elem_t_fp8 *dram_ptr = ((elem_t_fp8 *)A_in) + (i * DIM) * MATMUL_K + k * DIM;
      uint32_t sp_addr = a_base + (i * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *)dram_ptr, sp_addr, DIM, DIM);
    }

  // ---- MVIN B: tile (k,j) -> b_base + (k*tiles_J + j)*DIM ; row stride = N ----
  gemmini_config_ld(MATMUL_N * sizeof(elem_t_fp8));
  for (int k = 0; k < tiles_K; k++)
    for (int j = 0; j < tiles_J; j++) {
      elem_t_fp8 *dram_ptr = ((elem_t_fp8 *)B_in) + (k * DIM) * MATMUL_N + j * DIM;
      uint32_t sp_addr = b_base + (k * tiles_J + j) * DIM;
      gemmini_extended_mvin((void *)dram_ptr, sp_addr, DIM, DIM);
    }

  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);

  // compute + store only (loads already done by mvin): skips = 0x38
  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base,
                       BANK_NUM * BANK_ROWS, 0, SPAD_DEST, false, false, false, false,
                       false, NO_ACTIVATION, 0, 0, false, 0x38);
  gemmini_fence();

  gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * DIM, MATMUL_M * MATMUL_N);

  printf("SPIKE_MX_BEGIN\n");
  for (int m = 0; m < MATMUL_M; m++) {
    for (int n = 0; n < MATMUL_N; n++) printf("%04x\n", C_hw[m][n]);
  }
  printf("SPIKE_MX_END\n");
  return 0;
}
