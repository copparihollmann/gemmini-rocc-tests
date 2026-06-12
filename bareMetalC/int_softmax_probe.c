// Phase-5 probe: does the FLOAT-FREE integer I-BERT softmax run on the no-FPU Rocket?
// The gemmini matmul + mvout readout are already VCS-validated (matmul *_LEGAL.c). The only
// NEW/untested piece of the integer attention is the host softmax C. This runs it standalone on
// deterministic bf16 scores, checksums P_q+P_scales, and checks vs a spike-captured gold.
// If float had leaked in, it would TRAP like fpu_probe; integer runs (cf. int_probe).
#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

#define ATT_S 128
#define GROUPS_S (ATT_S/32)
#define NROWS 8

// ---- integer softmax helpers (identical to attention_gen.py) ----
#define MX_QLN2 710
#define MX_QLN2_INV 92
#define MX_QB 1385
#define MX_QC 1006165
#define MX_FRAC 24
#define MX_P_TARGET_LOG2 4
static inline int64_t mx_ibf16_to_q(uint16_t u){
  int s=(u>>15)&1,e=(u>>7)&0xFF,m=u&0x7F; int64_t mant=(e==0)?m:(0x80|m); int sh=(int)e-127-7+10;
  int64_t q=(sh>=0)?(mant<<sh):((mant+((int64_t)1<<(-sh-1)))>>(-sh)); return s?-q:q;
}
static inline int mx_ilog2(uint64_t n){int r=0;
  if(n>=((uint64_t)1<<32)){n>>=32;r+=32;} if(n>=((uint64_t)1<<16)){n>>=16;r+=16;}
  if(n>=((uint64_t)1<<8)){n>>=8;r+=8;} if(n>=((uint64_t)1<<4)){n>>=4;r+=4;}
  if(n>=((uint64_t)1<<2)){n>>=2;r+=2;} if(n>=((uint64_t)1<<1)){r+=1;} return r;}
static inline int64_t mx_iexp(int64_t qa){int64_t z=(-qa*MX_QLN2_INV)>>16;int64_t qp=qa+z*MX_QLN2;
  if(z>=63)return 0;
  return ((qp+MX_QB)*(qp+MX_QB)+MX_QC)>>z;}
static inline uint8_t mx_ienc_fp8(int64_t p,uint64_t inv_sm,int ls,int e){
  if(p<=0)return 0; uint64_t R=(((uint64_t)p*inv_sm)+((uint64_t)1<<(ls-1)))>>ls; if(R==0)return 0;
  int top=mx_ilog2(R); uint8_t M;
  if(top>=4){M=(uint8_t)((R>>(top-3))&0x7); if((R>>(top-4))&1){M++; if(M==8){M=0;top++;}}}
  else M=(uint8_t)((R<<(3-top))&0x7);
  int lg=top-MX_FRAC-e,E=lg+7;
  if(E<1){int rs=1-E,sh=top-3+rs; return (rs<=3&&sh>=0)?(uint8_t)((R>>sh)&0x7):0;}
  if(E>15)return 0x7E; if(E==15&&M==7)M=6; return (uint8_t)((E<<3)|(M&7));
}

static uint8_t P_q[ATT_S][ATT_S];
static uint8_t P_scales[GROUPS_S][ATT_S];

int main(){
  // deterministic bf16 scores built with INTEGER ops only (no float): packed 4/word in W.
  static uint64_t W[ATT_S][ATT_S/4];
  for(int i=0;i<NROWS;i++) for(int w=0;w<ATT_S/4;w++){
    uint64_t v=0;
    for(int l=0;l<4;l++){ int j=w*4+l; int val=((i*131+j*17)%211)-105;
      uint16_t bf; if(val==0) bf=0; else {int a=val<0?-val:val; int lg=mx_ilog2((uint64_t)a);
        int e=lg+127; int man=((a<<3)>>lg)&0x7; bf=(uint16_t)(((val<0)<<15)|((e&0xFF)<<7)|(man<<4));}
      v|=((uint64_t)bf)<<(l*16);
    }
    W[i][w]=v;
  }
  // run the integer softmax
  for(int i=0;i<NROWS;i++){
    int64_t q[ATT_S]; int64_t mq=-((int64_t)1<<62);
    for(int j=0;j<ATT_S;j+=4){uint64_t w=W[i][j/4];
      for(int l=0;l<4;l++){int64_t qq=mx_ibf16_to_q((uint16_t)((w>>(l*16))&0xFFFF)); q[j+l]=qq; if(qq>mq)mq=qq;}}
    int64_t p[ATT_S],sm=0;
    for(int j=0;j<ATT_S;j++){int64_t pv=mx_iexp(q[j]-mq);p[j]=pv;sm+=pv;}
    int ls=mx_ilog2((uint64_t)sm);
    uint64_t inv_sm=(((uint64_t)1<<(ls+MX_FRAC))+((uint64_t)sm>>1))/(uint64_t)sm;
    for(int g=0;g<GROUPS_S;g++){
      int64_t gmax=0; for(int j=g*32;j<(g+1)*32;j++) if(p[j]>gmax)gmax=p[j];
      if(gmax==0){P_scales[g][i]=0; for(int j=g*32;j<(g+1)*32;j++)P_q[i][j]=0; continue;}
      int e=mx_ilog2((uint64_t)gmax)-ls-MX_P_TARGET_LOG2; P_scales[g][i]=(uint8_t)(e+127);
      for(int j=g*32;j<(g+1)*32;j++)P_q[i][j]=mx_ienc_fp8(p[j],inv_sm,ls,e);
    }
  }
  // checksum (order-sensitive)
  uint64_t cs=1469598103934665603ULL;
  for(int i=0;i<NROWS;i++){for(int j=0;j<ATT_S;j++){cs^=P_q[i][j];cs*=1099511628211ULL;}}
  for(int g=0;g<GROUPS_S;g++)for(int i=0;i<NROWS;i++){cs^=P_scales[g][i];cs*=1099511628211ULL;}
  printf("INT_SOFTMAX_PROBE checksum=0x%016llx\n",(unsigned long long)cs);
  printf("INT_SOFTMAX_PROBE DONE\n");
  return 0;
}
