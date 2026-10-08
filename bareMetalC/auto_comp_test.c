#include "include/autocomp_aliases.h"
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

#define REPEAT_TEST_ITERS 1
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

#define RUN_BASELINE_CODE 0
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
    
  // Configures the store instruction for moving data from accumulator to DRAM.
  // Expects 256 columns in the output matrix.
  config_st((256));

  // Configures the execution unit for weight-stationary dataflow.
  // NO_ACTIVATION: no activation function is applied.
  // A_stride: 1 means consecutive rows of A are fed.
  // A_transpose: false means A is not transposed.
  // B_transpose: false means B is not transposed.
  config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 1, false, false);

  // Configures mvin instruction with id 2.
  // dram_stride: 256 bytes stride for DRAM access.
  // scale_factor: 1.0f, no scaling.
  // spad_block_stride: 16, blocks of 16 columns are loaded.
  // id: 2 for mvin3.
  config_ld((256), 1.0f, 16, 2);

  // Configures mvin instruction with id 1.
  // dram_stride: 256 bytes stride for DRAM access.
  // scale_factor: 1.0f, no scaling.
  // spad_block_stride: 16, blocks of 16 columns are loaded.
  // id: 1 for mvin2.
  config_ld((256), 1.0f, 16, 1);

  // Configures mvin instruction with id 0.
  // dram_stride: 0, indicates zero fill for memory region.
  // scale_factor: 1.0f, no scaling.
  // spad_block_stride: 16, blocks of 16 columns are loaded.
  // id: 0 for mvin.
  config_ld(0, 1.0f, 16, 0);

  // Base address for A tiles in the scratchpad.
  // We load a 64x64 block of A for each 16x16 block of C.
  // Each row in scratchpad is DIM elements wide.
  // To store 64 rows of 64 elements, we need 64 * (64 / 16) = 256 rows in spad if each row is 16 elements.
  // We load A in 16-row blocks. For 64 rows, we have 4 such blocks (ko).
  // Each mvin2 loads 16 rows, 64 columns. The total A load per `i` is 64 rows, 64 cols.
  uint32_t a_spad_base = 0;

  // Base address for B tiles in the scratchpad.
  // We load a 64x256 block of B for each `i` iteration.
  // `ko` selects 16-row segments of B (0 to 63).
  // `k` selects 16-column segments of B (0 to 255).
  // The total B data for `i` will occupy 64 rows * (256/16) = 64 * 16 = 1024 rows in spad, per 64 columns of B.
  // Each mvin3 loads 16 rows, 64 columns.
  uint32_t b_spad_base = 64 * 16 * sizeof(int8_t); // 1024 elements for A, leaving space for B

  // The outer loops iterate through 16x16 output tiles.
  // i: Controls the row blocks of the output matrix C. (0 to 15)
  for (int_fast32_t i = 0; i < 16; i++) {
    // Load a larger chunk of A for the current row block of C (64 rows).
    // This block of A (64x64) will be reused across the 4 'j' iterations for this 'i'.
    // `ko` iterates through the 4 blocks of 16 rows within the 64 rows of A.
    for (int_fast32_t ko = 0; ko < 4; ko++) {
      // mvin2: Load 16 rows, 64 columns of A.
      // Source: &A[(16 * i)][64 * ko]
      // Destination: a_spad_base + (ko * 64) (each row in spad is 64 elements)
      mvin2(&A[(16 * i)][64 * ko], a_spad_base + (ko * 64), 64, 16); // Corrected cols and rows for mvin2
    }

    // The j loop iterates through 4 blocks of columns for the output matrix C.
    // Each 'j' iteration corresponds to a 16x64 block of C.
    for (int_fast32_t j = 0; j < 4; j++) {
      // Initialize the accumulator for C with zeros for the current output tile (16x64).
      // Bit 31 set to 1 indicates accumulator. Bit 30 is 0 for overwrite.
      uint32_t res = 1 << 31;
      for (int_fast32_t j_in_o = 0; j_in_o < 4; j_in_o++) {
        // Move zeros into the accumulator for the current 16x16 output sub-block.
        // Address is `res + (j_in_o * 256) / 16`.
        mvin(0, res + (j_in_o * 256) / 16, 16, 16); // Corrected cols and rows for mvin
      }

      // Process B tiles and perform computations by fusing mvin3 with preload/compute.
      // `ko` iterates through the 4 blocks of 16 rows within the 64 rows of B.
      // `k` iterates through the 4 blocks of 16 columns within the 64 columns of B for this `j` slice.
      for (int_fast32_t ko = 0; ko < 4; ko++) {
        for (int_fast32_t k = 0; k < 4; k++) {
          // Load a B tile using mvin3. This tile is 16 rows x 64 columns.
          uint32_t current_b_dram_row_offset = (64 * ko + 16 * k);
          uint32_t current_b_spad_dest_addr = b_spad_base + (ko * 256) + (k * 64);
          mvin3(&B[current_b_dram_row_offset][64 * j], current_b_spad_dest_addr, 64, 16); // Corrected cols and rows for mvin3

          // Now, for this loaded B tile (16x64), perform computations for the relevant j_in_o blocks.
          // The `j_in_o` loop selects which 16x16 output blocks use this particular B tile.
          for (int_fast32_t j_in_o = 0; j_in_o < 4; j_in_o++) {
            // Calculate the scratchpad address for the current B tile.
            // This B tile was just loaded to `current_b_spad_dest_addr` (16 rows, 64 columns).
            // We need to access the `j_in_o`-th 16x16 slice within this 16x64 block.
            uint32_t current_b_spad_addr = current_b_spad_dest_addr + (j_in_o * 16);

            // Calculate the accumulator address for the current C sub-block (16x16).
            // `res + (j_in_o * 256) / 16` gives the base address of the C sub-block.
            // `| 0x40000000` sets bit 30 to 1, indicating accumulation.
            uint32_t current_c_acc_addr = res + (j_in_o * 256) / 16 | 0x40000000;

            // Preload the B weights onto the systolic array.
            // The B tile being preloaded is 16x16.
            preload(current_b_spad_addr, current_c_acc_addr, 16, 16, 16, 16); // Corrected B_cols, B_rows, C_cols, C_rows

            // Calculate the scratchpad address for the current A tile.
            // A is loaded as a 64x64 block. We access 16x16 tiles from it.
            // `ko` selects the 16-row segment (0-63).
            // `k` selects the 16-column segment (0-63) within that 16-row segment.
            uint32_t current_a_spad_addr = a_spad_base + (ko * 64) + (k * 16);

            // Perform the computation with the preloaded B weights and loaded A tiles.
            // `~((uint32_t)0)` is used as the bias address, indicating no bias.
            // `16, 16` are the dimensions of the A tile.
            // `16, 16` are the dimensions of the bias if present, here set to DIM.
            compute_preloaded(current_a_spad_addr, ~((uint32_t)0), 16, 16, 16, 16); // Corrected A_cols, A_rows, bias_cols, bias_rows
          }
        }
      }

      // Move out the computed results from the accumulator to DRAM for the current 16x64 output tile.
      for (int_fast32_t j_in_o = 0; j_in_o < 4; j_in_o++) {
        // The address `&C[(16 * i)][16 * j_in_o + 64 * j]` is the destination in DRAM for the output C.
        // `res + ((j_in_o) * (256)) / 16` is the source address in the accumulator for the 16x16 sub-block.
        // `16` is the number of columns to move out.
        // `16` is the number of rows to move out.
        mvout(&C[(16 * i)][16 * j_in_o + 64 * j], res + (j_in_o * 256) / 16, 16, 16); // Corrected cols and rows for mvout
      }
    }
  }

    }
    fence();
    generated_implementation_end_cycle = read_cycles();
    printf("Generated implementation latency: %d cycles\n", generated_implementation_end_cycle - generated_implementation_start_cycle);

    // SUBSTITUTE END
  }
  printf("Correct result\n");
}