// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define ACTIVATION NO_ACTIVATION

// #define FULL_BIAS_WIDTH 0
// #if FULL_BIAS_WIDTH
// typedef acc_t ACC_T;
// #else
// typedef elem_t ACC_T;
// #endif

#define MAKESTR(NAME) #NAME
#define XMAKESTR(NAME) MAKESTR(NAME)

#define A_TRANSPOSE 0
#define B_TRANSPOSE 0

#define NO_BIAS 1
#define REPEATING_BIAS 0
#define SUB_BIAS 0

#define A_MATRIX_NAME A
#define B_MATRIX_NAME B
#define C_MATRIX_NAME C
#define OUTPUT_MATRIX_NAME C_MATRIX_NAME
#if NO_BIAS==0
  #define D_MATRIX_NAME D
#endif

#define MAT_DIM_I 256
#define MAT_DIM_K 256
#define MAT_DIM_J 256

#if A_TRANSPOSE==1
  #define A_ROWS MAT_DIM_K
  #define A_COLS MAT_DIM_I
#else
  #define A_ROWS MAT_DIM_I
  #define A_COLS MAT_DIM_K
#endif

#if B_TRANSPOSE==1
  #define B_ROWS MAT_DIM_J
  #define B_COLS MAT_DIM_K
#else
  #define B_ROWS MAT_DIM_K
  #define B_COLS MAT_DIM_J
#endif

#define CHECK_RESULT 1
#undef FAST
// #define FAST

void print_tile(elem_t* in, int tile_dim) {
  for (size_t r = 0; r < tile_dim; r++) {
    printf("row starts at: %p\n", in +r*MAT_DIM_J);
    for (size_t c = 0; c < tile_dim; c++) {
      printf("%d ", *(in +r*MAT_DIM_J + c));
    }
    printf("\n");
  }
}

// void full_matmul(elem_t A[MAT_DIM_I][MAT_DIM_K], elem_t B[MAT_DIM_K][MAT_DIM_J], ACC_T D[MAT_DIM_I][MAT_DIM_J], full_t C_full[MAT_DIM_I][MAT_DIM_J]) {
//   for (size_t r = 0; r < MAT_DIM_I; r++)
//     for (size_t c = 0; c < MAT_DIM_J; c++) {
//       C_full[r][c] = D[r][c];
//       for (size_t k = 0; k < MAT_DIM_K; k++)
//         C_full[r][c] += A[r][k]*B[k][c];
//     }
// }

void A_printMatrix(elem_t m[A_ROWS][A_COLS]) {
  for (size_t i = 0; i < A_ROWS; ++i) {
    for (size_t j = 0; j < A_COLS; ++j)
      printf("%d ", (int) m[i][j]);
    printf("\n");
  }
}

void B_printMatrix(elem_t m[B_ROWS][B_COLS]) {
  for (size_t i = 0; i < B_ROWS; ++i) {
    for (size_t j = 0; j < B_COLS; ++j)
      printf("%d ", (int) m[i][j]);
    printf("\n");
  }
}

void D_printMatrix(elem_t m[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      printf("%d ", (int) m[i][j]);
    printf("\n");
  }
}

void full_printMatrix(elem_t m[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      printf("%d ", (int) m[i][j]);
    printf("\n");
  }
}

// void my_printMatrix(elem_t **m, int I, int J) {
//   for (size_t i = 0; i < I; ++i) {
//     for (size_t j = 0; j < J; ++j)
//       printf("%d ", m[i][j]);
//     printf("\n");
//   }
// }

int full_is_equal(elem_t x[MAT_DIM_I][MAT_DIM_J], elem_t y[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i)
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      if (x[i][j] != y[i][j]) {
        return 0;
      }
  return 1;
}

void full_matscale(full_t full[MAT_DIM_I][MAT_DIM_J], elem_t out[MAT_DIM_I][MAT_DIM_J], acc_scale_t scale) {
  for (size_t r = 0; r < MAT_DIM_I; r++)                             
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      // Scale element
      full_t scaled = ACC_SCALE(full[r][c], scale);

      // Saturate and cast element
#ifndef ELEM_T_IS_FLOAT
      full_t elem = scaled > elem_t_max ? elem_t_max : (scaled < elem_t_min ? elem_t_min : scaled);
      out[r][c] = elem;
#else
      out[r][c] = scaled; // TODO should we also saturate when using floats?
#endif
    }
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    static elem_t A_MATRIX_NAME[A_ROWS][A_COLS] row_align(1);
    static elem_t B_MATRIX_NAME[B_ROWS][B_COLS] row_align(1);
#if NO_BIAS==0
    static elem_t D_MATRIX_NAME[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);
#endif
    static elem_t gold[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);
    static elem_t C_MATRIX_NAME[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);

#ifdef FAST
#define RAND 1
#else
#define RAND rand()
#endif

#define REPEAT_TEST_ITERS 5
  for (int repeat_iters = 0; repeat_iters < REPEAT_TEST_ITERS; repeat_iters++) {
    // printf("Init A\n");
    for (size_t i = 0; i < A_ROWS; i++) {
      for (size_t j = 0; j < A_COLS; j++) {
        A_MATRIX_NAME[i][j] = (elem_t) (RAND % (1<<3) == 0);
      }
    }

    // printf("Init B\n");
    for (size_t i = 0; i < B_ROWS; i++) {
      for (size_t j = 0; j < B_COLS; j++) {
        B_MATRIX_NAME[i][j] = (elem_t) (RAND % (1<<3) == 0);
      }
    }

#if NO_BIAS==0
    // printf("Init D\n");
    for (size_t i = 0; i < MAT_DIM_I; i++) {
      for (size_t j = 0; j < MAT_DIM_J; j++) {
        D_MATRIX_NAME[i][j] = RAND % 2;
      }
    }
#endif

#define RUN_BASELINE_CODE 1
#if RUN_BASELINE_CODE
    // Baseline implementation
#if NO_BIAS==0
    tiled_matmul_outer_eigen_bias(A_MATRIX_NAME, B_MATRIX_NAME, D_MATRIX_NAME, gold, MAT_DIM_I, MAT_DIM_K, MAT_DIM_J, A_TRANSPOSE, B_TRANSPOSE, SUB_BIAS);
#else
    tiled_matmul_outer_eigen(A_MATRIX_NAME, B_MATRIX_NAME, gold, MAT_DIM_I, MAT_DIM_K, MAT_DIM_J, A_TRANSPOSE, B_TRANSPOSE);
#endif
#endif

    // Generated implementation
    // SUBSTITUTE HERE
    int generated_implementation_start_cycle;
int generated_implementation_end_cycle;
    gemmini_flush(0);
    fence();
    generated_implementation_start_cycle = read_cycles();
    {
    
  config_st((256));
  config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, false, false);
  config_ld((256), 1.0f, 16, 2);
  config_ld((256), 1.0f, 16, 1);
  config_ld(0, 1.0f, 16, 0);

  for (int_fast32_t i = 0; i < 16; i++) {
    for (int_fast32_t j = 0; j < 4; j++) {
      uint32_t res = 1 << 31;
      for (int_fast32_t j_in_o = 0; j_in_o < 4; j_in_o++) {
        mvin( 0, res + ((j_in_o) * (256))/16,(16 + 0), (16 + 0) );
      }
      uint32_t a = 0;
      uint32_t b = 16 * 16 * 4 * 4 * sizeof(int8_t) / 16;
      for (int_fast32_t ko = 0; ko < 4; ko++) {
        mvin2( &A[(16 * i)][64 * ko], a + ((ko) * (1024))/16, 16*(4 + 0), (16 + 0) );
        for (int_fast32_t k = 0; k < 4; k++) {
          mvin3( &B[(64 * ko + 16 * k)][64 * j], b + ((ko) * (4096) + (k) * (1024))/16, 16*(4 + 0), (16 + 0) );
        }
        for (int_fast32_t k = 0; k < 4; k++) {
          for (int_fast32_t j_in_o = 0; j_in_o < 4; j_in_o++) {
            preload(b + ((ko) * (4096) + (k) * (1024) + (j_in_o) * (256))/16, res + ((j_in_o) * (256))/16 | 0x40000000, (16 + 0), (16 + 0), (16 + 0), (16 + 0));
            compute_preloaded(a + ((ko) * (1024) + (k) * (256))/16, ~((uint32_t)0), (16 + 0), (16 + 0), 16, 16);
          }
        }
      }
      for (int_fast32_t j_in_o = 0; j_in_o < 4; j_in_o++) {
        mvout( &C[(16 * i)][16 * j_in_o + 64 * j], res + ((j_in_o) * (256))/16, (16 + 0), (16 + 0) );
      }
    }
  }

    }
    fence();
    generated_implementation_end_cycle = read_cycles();
    if (!full_is_equal(OUTPUT_MATRIX_NAME, gold)) {
    printf("Incorrect result\n");
    exit(1);
    } else {
    printf("Generated implementation latency: %d cycles\n", generated_implementation_end_cycle - generated_implementation_start_cycle);
    }

    // SUBSTITUTE END
  }
  printf("Correct result\n");
}