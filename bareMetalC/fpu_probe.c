// Phase-0 probe: does the host core have a hardware FPU?
// Built -march=rv64gc (hard float) -> emits fmul.s/fadd.s/fcvt.w.s.
// On a no-FPU Rocket these are illegal -> trap/mis-execute. On spike (has FPU) -> runs.
#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

int main() {
  volatile float a = 3.5f, b = 2.0f, c = 1.0f;
  float r = a * b + c;          // fmul.s + fadd.s
  int ri = (int)r;              // fcvt.w.s
  printf("FPU_PROBE result=%d (expect 8)\n", ri);
  if (ri == 8) { printf("FPU_PROBE PASSED\n"); return 0; }
  printf("FPU_PROBE FAILED\n");
  return 1;
}
