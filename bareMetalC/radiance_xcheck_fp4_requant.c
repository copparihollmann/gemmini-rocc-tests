// FP4 (e2m1) variant of radiance_xcheck_fp8.c: run REAL spike's MX matmul over the
// chipyard-mx fp4 128x128 operands and print the bf16 result, so cyclotron's co-model
// and autocomp's mx_golden can be diffed against it.
//
// Sub-byte packing: A_in_hw is [M/2][K] (2 e2m1 codes per byte, packed along M),
// B_in is [K][N/2] (packed along N). PE tiles are 32x32, so tiles_I = M/DIM/2.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "include/gemmini_testutils.h"

#define DIM 16
#define ADDR_LEN 32
#include "include/matmul_fp4_128x128.h"

typedef uint8_t elem_t_p;

int main() {
  static uint16_t C_hw[MATMUL_M][MATMUL_N];
  uint32_t scale_factors[512] = {0};
  memset(C_hw, 0, sizeof(C_hw));

  const int tiles_I = MATMUL_M / DIM / 2;   // 32-wide PE tiles
  const int tiles_J = MATMUL_N / DIM / 2;
  const int tiles_K = MATMUL_K / DIM;

  const uint32_t a_base = 0;
  const uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
  const int SPAD_DEST = 128;

  gemmini_flush(0);
  // act_fmt = wgt_fmt = 2 (fp4), out = 3 (full/bf16)
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0,
                              false, 2, 2, 2, 0);

  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);

  gemmini_config_ld(MATMUL_K * sizeof(elem_t_p));           // A packed rows: K bytes
  for (int i = 0; i < tiles_I; i++)
    for (int k = 0; k < tiles_K; k++) {
      elem_t_p *dram_ptr = ((elem_t_p *)A_in_hw) + (i * DIM) * MATMUL_K + k * DIM;
      gemmini_extended_mvin((void *)dram_ptr, a_base + (i * tiles_K + k) * DIM, DIM, DIM);
    }

  gemmini_config_ld(MATMUL_N * sizeof(elem_t_p) / 2);       // B packed rows: N/2 bytes
  for (int k = 0; k < tiles_K; k++)
    for (int j = 0; j < tiles_J; j++) {
      elem_t_p *dram_ptr = ((elem_t_p *)B_in) + (k * DIM) * (MATMUL_N / 2) + j * DIM;
      gemmini_extended_mvin((void *)dram_ptr, b_base + (k * tiles_J + j) * DIM, DIM, DIM);
    }

  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base,
                       BANK_NUM * BANK_ROWS, 0, SPAD_DEST, false, false, false, false,
                       false, NO_ACTIVATION, 0, 0, false, 0x38);
  gemmini_fence();
  gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * DIM, MATMUL_M * MATMUL_N);

  printf("SPIKE_MX_BEGIN\n");
  const uint8_t *pk = (const uint8_t *)&C_hw[0][0];
  for (int i = 0; i < MATMUL_M * MATMUL_N; i++) printf("%02x\n", pk[i]);
  printf("SPIKE_MX_END\n");
  printf("SPIKE_SCALES_BEGIN\n");
  const uint8_t *sc = (const uint8_t *)scale_factors;
  for (int i = 0; i < MATMUL_M * (MATMUL_N / 32); i++) printf("%02x\n", sc[i]);
  printf("SPIKE_SCALES_END\n");
  return 0;
}
