// AUTO-GENERATED MX-Gemmini attention harness (S=128, D=64).
// O = softmax(Q K^T) V. gold = baseline hardware output (captured).
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#include <stdlib.h>
#endif
#include "include/gemmini_testutils.h"
#include "include/mxgen_att_qk_128x64.h"
#include "include/mxgen_att_v_128x64.h"

#define DIM 16
#define BF16_PER_WORD 4
#define ATT_S 128
#define ATT_D 64
#define S_OUT_COLS (ATT_S / BF16_PER_WORD)
#define D_OUT_COLS (ATT_D / BF16_PER_WORD)
#define GROUPS_S (ATT_S / 32)

typedef uint8_t  elem_t;
typedef uint64_t out_t;
#ifndef fence
#define fence() gemmini_fence()
#endif
#define OUTPUT_MATRIX_NAME O_hw

static const out_t gold[ATT_S][D_OUT_COLS] = {
  {0}
};

// scratch: scores (bf16-packed), P quantized + scales
static out_t  S1_hw[ATT_S][S_OUT_COLS];
static elem_t P_q[ATT_S][ATT_S];
static uint8_t P_scales[GROUPS_S][ATT_S];

static inline float fp8d(uint8_t b){int s=(b>>7)&1,e=(b>>3)&0xF,m=b&7;float sg=s?-1.f:1.f; if(e==0)return sg*(m/8.f)*pow2i(-6); return sg*(1.f+m/8.f)*pow2i(e-7);}
static inline float bf16f(uint16_t b){union{uint32_t u;float f;}v;v.u=((uint32_t)b)<<16;return v.f;}
static inline float bf16_to_f(uint16_t b) {
  union { uint32_t u; float f; } v; v.u = ((uint32_t)b) << 16; return v.f;
}

// --- libm-free float helpers (baremetal links without libm) ---
static inline uint32_t f_bits(float f) { union { uint32_t u; float f; } v; v.f = f; return v.u; }
static inline float bits_f(uint32_t u) { union { uint32_t u; float f; } v; v.u = u; return v.f; }
static inline float f_abs(float x) { return bits_f(f_bits(x) & 0x7FFFFFFFu); }
static inline int f_exp(float x) { return (int)((f_bits(x) >> 23) & 0xFF) - 127; }  // floor(log2|x|), normals
static inline float pow2i(int e) {
  if (e < -126) return 0.0f;
  if (e > 127) e = 127;
  return bits_f((uint32_t)(e + 127) << 23);
}
static inline float exp_f(float x) {
  if (x < -87.0f) return 0.0f;
  if (x > 88.0f) x = 88.0f;
  float z = x * 1.44269504f;                  // x / ln2
  int n = (int)(z + (z >= 0 ? 0.5f : -0.5f));
  float f = z - (float)n;                     // [-0.5, 0.5]
  float p = 0.9999999916f + f*(0.6931471825f + f*(0.2401536316f + f*(0.0558263185f + f*(0.0089893397f + f*0.0018775767f))));
  return p * pow2i(n);
}

// fp8 e4m3 round-toward-zero encode of x (|x| <= 448 after scaling)
static inline uint8_t fp8_e4m3_rtz(float x) {
  if (x == 0.0f) return 0;
  uint8_t s = (f_bits(x) >> 31) ? 0x80 : 0;
  float a = f_abs(x);
  if (a >= 448.0f) return s | 0x7E;
  int e = f_exp(a);
  if (e < -6) return s;                  // flush subnormal
  float m = a * pow2i(-e);               // [1, 2)
  int mant = (int)((m - 1.0f) * 8.0f);
  return s | (uint8_t)(((e + 7) << 3) | (mant & 7));
}

int full_is_equal(out_t x[ATT_S][D_OUT_COLS], const out_t y[ATT_S][D_OUT_COLS]) {
  for (int i = 0; i < ATT_S; i++)
    for (int j = 0; j < D_OUT_COLS; j++)
      if (x[i][j] != y[i][j]) return 0;
  return 1;
}

#define REPEAT_TEST_ITERS 1
#define RUN_BASELINE_CODE 1

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall"); return 1; }
#endif
  static out_t O_hw[ATT_S][D_OUT_COLS];
  uint32_t scale_factors[512] = {0};
  int SPAD_DEST = 128;
  for (int repeat_iters = 0; repeat_iters < REPEAT_TEST_ITERS; repeat_iters++) {
    memset(O_hw, 0, sizeof(O_hw));
      // ---- 1) S1 = Q @ K^T on Gemmini (bf16 out) ----
  {
    int tiles_I = ATT_S / DIM, tiles_J = ATT_S / DIM, tiles_K = ATT_D / DIM;
    uint32_t a_base = 0;
    uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
    gemmini_flush(0);
    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
    gemmini_mx_load_scales((uint64_t)&Q_scales_row, sizeof(Q_scales_row), 0);
    gemmini_mx_load_scales((uint64_t)&KT_scales_col, sizeof(KT_scales_col), 1);
    gemmini_config_ld(ATT_D * sizeof(elem_t));
    for (int i = 0; i < tiles_I; i++)
      for (int k = 0; k < tiles_K; k++)
        gemmini_extended_mvin((void*)(((elem_t*)Q_in) + i * DIM * ATT_D + k * DIM),
                              a_base + (i * tiles_K + k) * DIM, DIM, DIM);
    gemmini_config_ld(ATT_S * sizeof(elem_t));   // B=KT[D][S] stride N=S -> slot (k*tiles_J+j)
    for (int j = 0; j < tiles_J; j++)
      for (int k = 0; k < tiles_K; k++)
        gemmini_extended_mvin((void*)(((elem_t*)KT_in) + (k * DIM) * ATT_S + j * DIM),
                              b_base + (k * tiles_J + j) * DIM, DIM, DIM);
    gemmini_config_st(S_OUT_COLS * sizeof(out_t));
    gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
    gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                         SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
    gemmini_mx_read_smem(&S1_hw[0][0], SPAD_DEST * 16, ATT_S * ATT_S);
    gemmini_fence();
  }

  // ---- 2) P = softmax(S1) rows on CPU; quantize to fp8 + per-32-group scales ----
  for (int i = 0; i < ATT_S; i++) {
    float row[ATT_S];
    float maxv = -1e30f;
    for (int j = 0; j < ATT_S; j++) {
      uint16_t b = (uint16_t)((S1_hw[i][j / 4] >> ((j % 4) * 16)) & 0xFFFF);
      row[j] = bf16_to_f(b);
      if (row[j] > maxv) maxv = row[j];
    }
    float sum = 0.0f;
    for (int j = 0; j < ATT_S; j++) { row[j] = exp_f(row[j] - maxv); sum += row[j]; }
    float inv_sum = 1.0f / sum;
    for (int j = 0; j < ATT_S; j++) row[j] *= inv_sum;
    // per-32-group power-of-two scale, then fp8 RTZ encode of row[j]/scale
    for (int g = 0; g < GROUPS_S; g++) {
      float gmax = 0.0f;
      for (int j = g * 32; j < (g + 1) * 32; j++)
        if (f_abs(row[j]) > gmax) gmax = f_abs(row[j]);
      int e = 0;
      if (gmax > 0.0f) e = f_exp(gmax / 448.0f) + 1;   // ceil(log2(gmax/448))
      float inv_scale = pow2i(-e);
      P_scales[g][i] = (uint8_t)(e + 127);
      for (int j = g * 32; j < (g + 1) * 32; j++)
        P_q[i][j] = fp8_e4m3_rtz(row[j] * inv_scale);
    }
  }

  // ---- 3) O = P @ V on Gemmini (bf16 out) ----
  {
    int tiles_I = ATT_S / DIM, tiles_J = ATT_D / DIM, tiles_K = ATT_S / DIM;
    uint32_t a_base = 0;
    uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
    gemmini_flush(0);
    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
    gemmini_mx_load_scales((uint64_t)&P_scales, sizeof(P_scales), 0);
    gemmini_mx_load_scales((uint64_t)&V_scales_col, sizeof(V_scales_col), 1);
    gemmini_config_ld(ATT_S * sizeof(elem_t));
    for (int i = 0; i < tiles_I; i++)
      for (int k = 0; k < tiles_K; k++)
        gemmini_extended_mvin((void*)(((elem_t*)P_q) + i * DIM * ATT_S + k * DIM),
                              a_base + (i * tiles_K + k) * DIM, DIM, DIM);
    gemmini_config_ld(ATT_D * sizeof(elem_t));   // B=V[S][D] stride N=D -> slot (k*tiles_J+j)
    for (int j = 0; j < tiles_J; j++)
      for (int k = 0; k < tiles_K; k++)
        gemmini_extended_mvin((void*)(((elem_t*)V_in) + (k * DIM) * ATT_D + j * DIM),
                              b_base + (k * tiles_J + j) * DIM, DIM, DIM);
    gemmini_config_st(D_OUT_COLS * sizeof(out_t));
    gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
    gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                         SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
    gemmini_mx_read_smem(&O_hw[0][0], SPAD_DEST * 16, ATT_S * ATT_D);
    gemmini_fence();
  }

  printf("REF_PLACEHOLDER\n");
    // SUBSTITUTE END
  }
  printf("Correct result\n");
#ifndef BAREMETAL
  exit(0);
#else
  return 0;
#endif
}
