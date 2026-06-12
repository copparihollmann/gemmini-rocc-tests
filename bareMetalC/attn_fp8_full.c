#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "include/gemmini_testutils.h"
#include "include/mxgen_att_qk_128x64.h"
#include "include/mxgen_att_v_128x64.h"
#define GEMMINI_SF_MEM 0x40088000
#define GEMMINI_SF_MEM_A (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B GEMMINI_SF_MEM
#define DIM 16
#define ATT_S 128
#define ATT_D 64
#define BF16_PER_WORD 4
#define GROUPS_S (ATT_S/32)
#define S_OUT_COLS (ATT_S/BF16_PER_WORD)
#define D_OUT_COLS (ATT_D/BF16_PER_WORD)
#define GEMMINI_CTRL 0x40084000
#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)
#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)
#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)
#define GEMMINI_BUSY_ADDR (GEMMINI_CTRL + 0x20)
#define ADDR_LEN 32
#ifndef SPIKE_SIM
#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    *((volatile uint64_t *) GEMMINI_RS1_ADDR) = (rs1); \
    *((volatile uint64_t *) GEMMINI_RS2_ADDR) = (rs2); \
    *((volatile uint32_t*) GEMMINI_INST_ADDR) = (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
}
#undef gemmini_fence
#define gemmini_fence() { while (*((volatile uint32_t *) GEMMINI_BUSY_ADDR)) asm volatile ("nop"); }
#endif
typedef uint8_t elem_t; typedef uint64_t out_t;
void load_scale_factors(volatile uint64_t *sf, uint8_t *sc, int INDIM, int K){
  for(size_t k=0;k<(size_t)K/32;k++)for(size_t i=0;i<(size_t)INDIM/8;i++)
    sf[k*INDIM/8+i]=((uint64_t*)sc)[k*INDIM/8+i];
}
static out_t S1_hw[ATT_S][S_OUT_COLS];
static elem_t P_q[ATT_S][ATT_S];
static uint8_t P_scales[GROUPS_S][ATT_S];
static out_t O_hw[ATT_S][D_OUT_COLS];
#define MX_QLN2 710
#define MX_QLN2_INV 92
#define MX_QB 1385
#define MX_QC 1006165
#define MX_FRAC 24
#define MX_P_TARGET_LOG2 4   /* P group-max -> ~2^4=16, NOT fp8-max: avoids e4 acc overflow */
static inline int64_t mx_ibf16_to_q(uint16_t u) {   // bf16 bits -> Q(1/1024) fixed-point int
  int s = (u >> 15) & 1, e = (u >> 7) & 0xFF, m = u & 0x7F;
  int64_t mant = (e == 0) ? m : (0x80 | m);
  int sh = (int)e - 127 - 7 + 10;                    // value * 2^SL, SL=10
  int64_t q = (sh >= 0) ? (mant << sh)
                        : ((mant + ((int64_t)1 << (-sh - 1))) >> (-sh));  // round-nearest
  return s ? -q : q;
}
static inline int mx_ilog2(uint64_t n) {   // floor(log2 n), n>0; no __builtin_clz (baremetal -nostdlib has no libgcc __clzdi2)
  int r = 0;
  if (n >= ((uint64_t)1 << 32)) { n >>= 32; r += 32; }
  if (n >= ((uint64_t)1 << 16)) { n >>= 16; r += 16; }
  if (n >= ((uint64_t)1 << 8))  { n >>= 8;  r += 8;  }
  if (n >= ((uint64_t)1 << 4))  { n >>= 4;  r += 4;  }
  if (n >= ((uint64_t)1 << 2))  { n >>= 2;  r += 2;  }
  if (n >= ((uint64_t)1 << 1))  { r += 1; }
  return r;
}
static inline int64_t mx_iexp(int64_t qa) {          // qa <= 0; returns integer pseudo-prob
  int64_t z  = (-qa * MX_QLN2_INV) >> 16;
  int64_t qp = qa + z * MX_QLN2;
  int64_t qe = (qp + MX_QB) * (qp + MX_QB) + MX_QC;
  if (z >= 63) return 0;
  return qe >> z;
}
static inline uint8_t mx_ienc_fp8(int64_t p, uint64_t inv_sm, int ls, int e) {  // t=(p/sm)*2^-e -> fp8 e4m3 RNE
  if (p <= 0) return 0;
  uint64_t R = (((uint64_t)p * inv_sm) + ((uint64_t)1 << (ls - 1))) >> ls;  // ==round((p<<MX_FRAC)/sm); rounded reciprocal+R
  if (R == 0) return 0;
  int top = mx_ilog2(R);
  uint8_t M;
  if (top >= 4) {                                   // round-nearest-even on the 3-bit mantissa
    M = (uint8_t)((R >> (top - 3)) & 0x7);
    if ((R >> (top - 4)) & 1) { M++; if (M == 8) { M = 0; top++; } }   // mantissa carry -> bump exponent
  } else {
    M = (uint8_t)((R << (3 - top)) & 0x7);
  }
  int lg = top - MX_FRAC - e, E = lg + 7;
  if (E < 1) { int rs = 1 - E, sh = top - 3 + rs; return (rs <= 3 && sh >= 0) ? (uint8_t)((R >> sh) & 0x7) : 0; }
  if (E > 15) return 0x7E;
  if (E == 15 && M == 7) M = 6;
  return (uint8_t)((E << 3) | (M & 7));
}


static inline float mx_bf16_to_f32(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }

int main(){
  uint32_t scale_factors[512]={0};
  int SPAD_DEST=0;
  uint64_t _t0=read_cycles();
  // ===== 1) QK = Q@K^T =====
  { int tiles_I=ATT_S/DIM, tiles_J=ATT_S/DIM, tiles_K=ATT_D/DIM;
    uint32_t a_base=0, b_base=BANK_NUM*BANK_ROWS-tiles_K*tiles_J*DIM;
    gemmini_flush(0);
    gemmini_extended3_config_ex(WEIGHT_STATIONARY,0,0,ACC_SCALE_IDENTITY,1,1,0,0,false,0,0,0,0);  // out fp8 P
    // HW SOFTMAX on the QK output: act_msb=1 -> acc_act=SOFTMAX; load I-BERT iexp constants (S_q=1/1024)
    gemmini_config_norm(MX_QLN2,    0,0,1,0,MX_QB,MX_QC);
    gemmini_config_norm(MX_QLN2_INV,1,0,1,0,MX_QB,MX_QC);
#ifdef SPIKE_SIM
    gemmini_mx_load_scales((uint64_t)&Q_scales_row,sizeof(Q_scales_row),0);
    gemmini_mx_load_scales((uint64_t)&KT_scales_col,sizeof(KT_scales_col),1);
#else
    load_scale_factors((volatile uint64_t*)GEMMINI_SF_MEM_A,(uint8_t*)&Q_scales_row,ATT_S,ATT_D);
    load_scale_factors((volatile uint64_t*)GEMMINI_SF_MEM_B,(uint8_t*)&KT_scales_col,ATT_S,ATT_D);
#endif
    gemmini_config_ld(ATT_D*sizeof(elem_t));
    for(int i=0;i<tiles_I;i++)for(int k=0;k<tiles_K;k+=2){int bb=(tiles_K-k<2)?(tiles_K-k):2;
      gemmini_extended_mvin((void*)(((elem_t*)Q_in)+i*DIM*ATT_D+k*DIM),a_base+(i*tiles_K+k)*DIM,bb*DIM,DIM);}
    gemmini_config_ld(ATT_S*sizeof(elem_t));
    for(int k=0;k<tiles_K;k++)for(int j=0;j<tiles_J;j+=2){int bb=(tiles_J-j<2)?(tiles_J-j):2;
      gemmini_extended_mvin((void*)(((elem_t*)KT_in)+(k*DIM)*ATT_S+j*DIM),b_base+(k*tiles_J+j)*DIM,bb*DIM,DIM);}
    gemmini_config_st(DIM*sizeof(elem_t));   // store stride = DIM*sizeof(elem_t) (matches validated LEGAL2; NOT S_OUT_COLS*out_t)
    gemmini_mxquant_config_mvout((uint64_t)scale_factors,tiles_I,tiles_J,tiles_K,0,0,1);
    gemmini_loop_ws_spad(tiles_I,tiles_J,tiles_K,0,0,0,a_base,BANK_NUM*BANK_ROWS,0,SPAD_DEST,false,false,false,false,false,SOFTMAX,0,0,false,0x38);
    gemmini_fence();
#if defined(SPIKE_SIM) && !defined(SPIKE_TEST_MVOUT)
    gemmini_mx_read_smem(&P_q[0][0],SPAD_DEST*16,ATT_S*ATT_S/2);  // fp8 P packed row-major
#else
    for(int i=0;i<tiles_J*tiles_I*2;i++) gemmini_mvout((void*)((uint64_t*)S1_hw+i*2*DIM),SPAD_DEST+i*DIM);
    gemmini_fence();
#endif
  }
  // ---- 2) HW softmax already applied on the accelerator: S1_hw now holds the bf16 probabilities P.
  //         Host only quantizes P -> fp8 + per-32-group e8m0 scales (NO host exp/iexp/sum: the ~97%
  //         softmax cost moved to the accelerator). p[j] = round(P*2^20) so the VALIDATED integer
  //         encoder below runs unchanged (sum ~= 2^20). (Spike proof uses float bf16 unpack;
  //         the no-FPU-deployable variant reads fp8 P straight from smem -- task #50 part A2.)
  { uint8_t *sf=(uint8_t*)scale_factors;  // accelerator did softmax+fp8 encode; just transpose the e8m0 scales for PV
    for(int row=0;row<ATT_S;row++) for(int kg=0;kg<GROUPS_S;kg++) P_scales[kg][row]=sf[row*GROUPS_S+kg]; }



  // clear the sticky SOFTMAX acc_act before PV. acc_act is set by config_st (rs1[3:2]); config_st(0)
  // PRESERVES a sticky SOFTMAX(4) (gemmini.cc:596-599). Issue a RELU config_st to break the "4" (no
  // matmul runs here, so RELU never applies to anything); the PV config_st(NO_ACTIVATION) below then
  // resolves to NONE (acc_act 1 != 4) -> PV is a plain matmul, no softmax/relu on O.
  gemmini_extended_config_st(DIM*sizeof(elem_t), RELU, ACC_SCALE_IDENTITY);
  // ===== 3) O = P@V =====
  { int tiles_I=ATT_S/DIM, tiles_J=ATT_D/DIM, tiles_K=ATT_S/DIM;
    uint32_t a_base=0, b_base=BANK_NUM*BANK_ROWS-tiles_K*tiles_J*DIM;
    gemmini_flush(0);
    gemmini_extended3_config_ex(WEIGHT_STATIONARY,0,0,ACC_SCALE_IDENTITY,1,1,0,0,false,0,0,3,0);
#ifdef SPIKE_SIM
    gemmini_mx_load_scales((uint64_t)&P_scales,sizeof(P_scales),0);
    gemmini_mx_load_scales((uint64_t)&V_scales_col,sizeof(V_scales_col),1);
#else
    load_scale_factors((volatile uint64_t*)GEMMINI_SF_MEM_A,(uint8_t*)&P_scales,ATT_S,ATT_S);
    load_scale_factors((volatile uint64_t*)GEMMINI_SF_MEM_B,(uint8_t*)&V_scales_col,ATT_D,ATT_S);
#endif
    gemmini_config_ld(ATT_S*sizeof(elem_t));
    for(int i=0;i<tiles_I;i++)for(int k=0;k<tiles_K;k+=2){int bb=(tiles_K-k<2)?(tiles_K-k):2;
      gemmini_extended_mvin((void*)(((elem_t*)P_q)+i*DIM*ATT_S+k*DIM),a_base+(i*tiles_K+k)*DIM,bb*DIM,DIM);}
    gemmini_config_ld(ATT_D*sizeof(elem_t));
    for(int k=0;k<tiles_K;k++)for(int j=0;j<tiles_J;j+=2){int bb=(tiles_J-j<2)?(tiles_J-j):2;
      gemmini_extended_mvin((void*)(((elem_t*)V_in)+(k*DIM)*ATT_D+j*DIM),b_base+(k*tiles_J+j)*DIM,bb*DIM,DIM);}
    gemmini_config_st(DIM*sizeof(elem_t));   // store stride = DIM*sizeof(elem_t) (matches validated LEGAL2; NOT D_OUT_COLS*out_t)
    gemmini_mxquant_config_mvout((uint64_t)scale_factors,tiles_I,tiles_J,tiles_K,0,0,1);
    gemmini_loop_ws_spad(tiles_I,tiles_J,tiles_K,0,0,0,a_base,BANK_NUM*BANK_ROWS,0,SPAD_DEST,false,false,false,false,false,NO_ACTIVATION,0,0,false,0x38);
    gemmini_fence();
#if defined(SPIKE_SIM) && !defined(SPIKE_TEST_MVOUT)
    gemmini_mx_read_smem(&O_hw[0][0],SPAD_DEST*16,ATT_S*ATT_D);
#else
    for(int i=0;i<tiles_J*tiles_I*2;i++) gemmini_mvout((void*)((uint64_t*)O_hw+i*2*DIM),SPAD_DEST+i*DIM);
    gemmini_fence();
#endif
  }
  uint64_t _t1=read_cycles();
  printf("CYCLES %llu\n",(unsigned long long)(_t1-_t0));
  uint64_t cs=1469598103934665603ULL;
  for(int i=0;i<ATT_S;i++)for(int j=0;j<D_OUT_COLS;j++){cs^=O_hw[i][j];cs*=1099511628211ULL;}
  printf("ATTN_VCS checksum=0x%016llx\n",(unsigned long long)cs);
  for(int i=0;i<ATT_S;i++)for(int j=0;j<D_OUT_COLS;j++) printf("OO %d %d %016llx\n",i,j,(unsigned long long)O_hw[i][j]);
  printf("ATTN_VCS DONE\n");
  return 0;
}
