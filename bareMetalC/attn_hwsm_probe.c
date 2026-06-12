// Validate the spike HW MX softmax pass: run QK with act=SOFTMAX so the accelerator
// (modeled in mx_loop_ws_spad) produces softmax(scores) = P directly in mx_smem (bf16),
// then dump P. Compare to true softmax(QK scores) in Python. No PV, no CPU softmax.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "include/gemmini_testutils.h"
#include "include/mxgen_att_qk_128x64.h"
#define DIM 16
#define ATT_S 128
#define ATT_D 64
#define BF16_PER_WORD 4
#define S_OUT_COLS (ATT_S/BF16_PER_WORD)
typedef uint8_t elem_t; typedef uint64_t out_t;
static out_t S1_hw[ATT_S][S_OUT_COLS];
#define MX_QLN2 710
#define MX_QLN2_INV 92
#define MX_QB 1385
#define MX_QC 1006165

int main(){
  uint32_t scale_factors[512]={0};
  int SPAD_DEST=0;
  { int tiles_I=ATT_S/DIM, tiles_J=ATT_S/DIM, tiles_K=ATT_D/DIM;
    uint32_t a_base=0, b_base=BANK_NUM*BANK_ROWS-tiles_K*tiles_J*DIM;
    gemmini_flush(0);
    gemmini_extended3_config_ex(WEIGHT_STATIONARY,0,0,ACC_SCALE_IDENTITY,1,1,0,0,false,0,0,3,0);
    // enable HW softmax: act_msb=1 -> acc_act=SOFTMAX, load I-BERT iexp constants (S_q=1/1024)
    gemmini_config_norm(MX_QLN2,    0,0,1,0,MX_QB,MX_QC);
    gemmini_config_norm(MX_QLN2_INV,1,0,1,0,MX_QB,MX_QC);
    gemmini_mx_load_scales((uint64_t)&Q_scales_row,sizeof(Q_scales_row),0);
    gemmini_mx_load_scales((uint64_t)&KT_scales_col,sizeof(KT_scales_col),1);
    gemmini_config_ld(ATT_D*sizeof(elem_t));
    for(int i=0;i<tiles_I;i++)for(int k=0;k<tiles_K;k+=2){int bb=(tiles_K-k<2)?(tiles_K-k):2;
      gemmini_extended_mvin((void*)(((elem_t*)Q_in)+i*DIM*ATT_D+k*DIM),a_base+(i*tiles_K+k)*DIM,bb*DIM,DIM);}
    gemmini_config_ld(ATT_S*sizeof(elem_t));
    for(int k=0;k<tiles_K;k++)for(int j=0;j<tiles_J;j+=2){int bb=(tiles_J-j<2)?(tiles_J-j):2;
      gemmini_extended_mvin((void*)(((elem_t*)KT_in)+(k*DIM)*ATT_S+j*DIM),b_base+(k*tiles_J+j)*DIM,bb*DIM,DIM);}
    gemmini_config_st(DIM*sizeof(elem_t));
    gemmini_mxquant_config_mvout((uint64_t)scale_factors,tiles_I,tiles_J,tiles_K,0,0,1);
    gemmini_loop_ws_spad(tiles_I,tiles_J,tiles_K,0,0,0,a_base,BANK_NUM*BANK_ROWS,0,SPAD_DEST,false,false,false,false,false,NO_ACTIVATION,0,0,false,0x38);
    gemmini_fence();
    gemmini_mx_read_smem(&S1_hw[0][0],SPAD_DEST*16,ATT_S*ATT_S);
  }
  for(int i=0;i<ATT_S;i++)for(int j=0;j<S_OUT_COLS;j++) printf("PP %d %d %016llx\n",i,j,(unsigned long long)S1_hw[i][j]);
  printf("HWSM DONE\n");
  return 0;
}
