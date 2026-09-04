// Reproduces dispatch_1 from mlp_wide gemmini compile (16x32x16 i8 matmul)
// using libgemmini's high-level API. Confirms that THIS SHAPE on a vanilla
// WS gemmini works under spike's RoCC model.
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "include/gemmini_testutils.h"  // chipyard headers
#define MAT_DIM_I 16
#define MAT_DIM_J 32
#define MAT_DIM_K 16
static elem_t A[MAT_DIM_I][MAT_DIM_K] row_align(1);
static elem_t B[MAT_DIM_K][MAT_DIM_J] row_align(1);
static elem_t D[MAT_DIM_I][MAT_DIM_J] row_align(1);
static elem_t C[MAT_DIM_I][MAT_DIM_J] row_align(1);
int main() {
  gemmini_flush(0);
  for (int i = 0; i < MAT_DIM_I; i++)
    for (int k = 0; k < MAT_DIM_K; k++) A[i][k] = (i+k) & 0x7f;
  for (int k = 0; k < MAT_DIM_K; k++)
    for (int j = 0; j < MAT_DIM_J; j++) B[k][j] = (k-j) & 0x7f;
  memset(D, 0, sizeof(D));
  memset(C, 0, sizeof(C));
  tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
                    (elem_t*)A, (elem_t*)B, NULL, (elem_t*)C,
                    MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
                    false, false,
                    false, false, 0,
                    WS);
  // Hash output to confirm completion.
  uint64_t h = 14695981039346656037ULL;
  for (int i = 0; i < MAT_DIM_I; i++)
    for (int j = 0; j < MAT_DIM_J; j++) {
      h ^= (uint64_t)(uint8_t)C[i][j];
      h *= 1099511628211ULL;
    }
  printf("dispatch1_repro C[0][0]=%d C[15][31]=%d  fnv1a=0x%016lx\n",
         C[0][0], C[15][31], h);
  return 0;
}
