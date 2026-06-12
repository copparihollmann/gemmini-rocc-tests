// Phase-0 control: pure-integer arithmetic. Must PASS on the no-FPU Rocket (VCS),
// proving the harness itself works and isolating the FPU as the cause of any fpu_probe failure.
#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

int main() {
  volatile int a = 7, b = 3, c = 1;
  int r = a * b + c;            // mul + add (RV64IMAC)
  printf("INT_PROBE result=%d (expect 22)\n", r);
  if (r == 22) { printf("INT_PROBE PASSED\n"); return 0; }
  printf("INT_PROBE FAILED\n");
  return 1;
}
