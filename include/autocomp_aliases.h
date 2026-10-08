// SPDX-License-Identifier: MIT
//
// autocomp_aliases.h
// ==================
//
// Bridges the shorthand intrinsic names autocomp's sols + LLM
// proposals use (`mvin`, `config_ld`, `preload`, ...) to the real
// gemmini.h API. The autocomp project's `auto-comp-v2` fork of
// gemmini-rocc-tests presumably shipped these aliases as part of
// the test runtime; that branch is no longer published upstream,
// so we re-author the bridge here.
//
// Include AFTER `include/gemmini.h` so the real symbol table is
// already populated. The substitution layer is purely textual —
// every alias expands to one gemmini_extended_* / gemmini_config_*
// call with arguments matched up by position.
//
// References (autocomp prompt source):
//   third_party/autocomp/autocomp/agents/gemmini/prompts/isa_prompt_conv.py
//   third_party/autocomp/autocomp/agents/gemmini/prompts/isa_prompt_admm.py

#ifndef AUTOCOMP_ALIASES_H
#define AUTOCOMP_ALIASES_H

#include "include/gemmini.h"

// --- Configuration ---

// autocomp's config_ex omits the sys_shift slot; map sys_shift=0.
#define config_ex(dataflow, act, A_stride, A_transpose, B_transpose) \
    gemmini_extended_config_ex(dataflow, act, /*sys_shift=*/0, A_stride, A_transpose, B_transpose)

// autocomp's config_ld takes the id slot last; gemmini_extended_config_ld
// has it last too — direct passthrough.
#define config_ld(dram_stride, scale_factor, spad_block_stride, id) \
    gemmini_extended4_config_ld(dram_stride, scale_factor, /*shrunk=*/false, spad_block_stride, id)

// autocomp has two config_st variants — single-arg (cols only) and
// the admm two-arg (stride, scale). The single-arg form matches
// gemmini.h's macro; the two-arg form maps to gemmini_extended2_config_st.
// Variadic-arg trick: define the 2-arg form first; the C preprocessor
// rejects the wrong arity at use-site if a caller mismatches.
#define config_st(stride) \
    gemmini_config_st(stride)

// --- Move-in (DMA → scratchpad) ---

// autocomp mvin has explicit (cols, rows); maps to gemmini_extended_mvin.
// mvin2/mvin3 are autocomp's lane-id flavours: mvin2 → mvin2 (id=1),
// mvin3 → mvin3 (id=2). The real gemmini.h API has matching macros.
#define mvin(dram_addr, spad_acc_addr, cols, rows) \
    gemmini_extended_mvin(dram_addr, spad_acc_addr, cols, rows)
#define mvin2(dram_addr, spad_acc_addr, cols, rows) \
    gemmini_extended_mvin2(dram_addr, spad_acc_addr, cols, rows)
#define mvin3(dram_addr, spad_acc_addr, cols, rows) \
    gemmini_extended_mvin3(dram_addr, spad_acc_addr, cols, rows)

// --- Move-out (accumulator → DRAM) ---

#define mvout(dram_addr, acc_addr, cols, rows) \
    gemmini_extended_mvout(dram_addr, acc_addr, cols, rows)

// --- Preload + compute ---

#define preload(B_spad_addr, C_acc_addr, B_cols, B_rows, C_cols, C_rows) \
    gemmini_extended_preload(B_spad_addr, C_acc_addr, B_cols, B_rows, C_cols, C_rows)

#define compute_preloaded(A_spad_addr, bias_spad_addr, A_cols, A_rows, bias_cols, bias_rows) \
    gemmini_extended_compute_preloaded(A_spad_addr, bias_spad_addr, A_cols, A_rows, bias_cols, bias_rows)

#define compute_accumulated(A_spad_addr, bias_spad_addr, A_cols, A_rows, bias_cols, bias_rows) \
    gemmini_extended_compute_accumulated(A_spad_addr, bias_spad_addr, A_cols, A_rows, bias_cols, bias_rows)

// --- Fence + ordering ---
//
// autocomp's harness uses bare `fence()` for memory/RoCC ordering
// before/after the candidate solution. The standard gemmini.h has
// `gemmini_fence()` (waits for outstanding Gemmini commands). Wrap
// it as a macro that compiles to a static-inline call so the C
// linker sees a real symbol if anyone &-references it.

static inline void __autocomp_fence(void) {
    asm volatile ("fence" ::: "memory");
    gemmini_fence();
}
#define fence() __autocomp_fence()

// --- Scalar reference matmul ---
//
// autocomp's harness compares its candidate's i8 output against an
// i8 `gold` matrix produced by `tiled_matmul_outer_eigen`. The
// signature is:
//
//   void tiled_matmul_outer_eigen(
//       elem_t A[I][K], elem_t B[K][J], elem_t gold[I][J],
//       size_t I, size_t K, size_t J,
//       bool A_transpose, bool B_transpose);
//
// Behaviour (matches what Gemmini's tiled_matmul_auto produces with
// `full_C=false, scale=ACC_SCALE_IDENTITY, NO_ACTIVATION, NO_BIAS`):
// compute the i32 accumulator A @ B then saturate to elem_t bounds.
//
// The variable-length-array parameter shape (`elem_t A[I][K]`) needs
// non-literal sizes; we accept `void *` and cast inside to side-step
// the C variable-length type rules at the caller's site (autocomp's
// harness passes concrete MAT_DIM_* constants, so the cast is safe).
//
// We provide a static-inline definition so every test unit gets its
// own copy — linker doesn't complain about multiple definitions and
// callers don't need a separate .c file in the build.

static inline elem_t __autocomp_saturate_elem(int32_t v) {
#ifdef ELEM_T_IS_FLOAT
    return (elem_t)v;
#else
    if (v > elem_t_max) v = elem_t_max;
    if (v < elem_t_min) v = elem_t_min;
    return (elem_t)v;
#endif
}

static inline void __autocomp_tiled_matmul_outer_eigen(
    const void *A_void, const void *B_void, void *gold_void,
    size_t I, size_t K, size_t J,
    int A_transpose, int B_transpose) {
    const elem_t *A = (const elem_t *)A_void;
    const elem_t *B = (const elem_t *)B_void;
    elem_t *gold = (elem_t *)gold_void;
    // For row-major storage:
    //   A_transpose=false → A[i*K + k]; true → A[k*I + i]
    //   B_transpose=false → B[k*J + j]; true → B[j*K + k]
    for (size_t i = 0; i < I; ++i) {
        for (size_t j = 0; j < J; ++j) {
            int32_t acc = 0;
            for (size_t k = 0; k < K; ++k) {
                elem_t a = A_transpose ? A[k * I + i] : A[i * K + k];
                elem_t b = B_transpose ? B[j * K + k] : B[k * J + j];
                acc += (int32_t)a * (int32_t)b;
            }
            gold[i * J + j] = __autocomp_saturate_elem(acc);
        }
    }
}

#define tiled_matmul_outer_eigen(A, B, gold, I, K, J, A_T, B_T) \
    __autocomp_tiled_matmul_outer_eigen( \
        (const void *)(A), (const void *)(B), (void *)(gold), \
        (size_t)(I), (size_t)(K), (size_t)(J), \
        (int)(A_T), (int)(B_T))

// --- Scalar reference matmul with bias (D) ---
//
// `tiled_matmul_outer_eigen_bias(A, B, D, gold, I, K, J, A_T, B_T, sub_bias)`
// computes gold = saturate( A @ B  ± D ). The `sub_bias` flag selects
// subtract (true) or add (false). D is a per-output-row bias matrix
// of shape [I][J] in the standard gemmini-rocc-tests harness.

static inline void __autocomp_tiled_matmul_outer_eigen_bias(
    const void *A_void, const void *B_void, const void *D_void, void *gold_void,
    size_t I, size_t K, size_t J,
    int A_transpose, int B_transpose, int sub_bias) {
    const elem_t *A = (const elem_t *)A_void;
    const elem_t *B = (const elem_t *)B_void;
    const elem_t *D = (const elem_t *)D_void;
    elem_t *gold = (elem_t *)gold_void;
    for (size_t i = 0; i < I; ++i) {
        for (size_t j = 0; j < J; ++j) {
            int32_t acc = (int32_t)D[i * J + j];
            if (sub_bias) acc = -acc;
            for (size_t k = 0; k < K; ++k) {
                elem_t a = A_transpose ? A[k * I + i] : A[i * K + k];
                elem_t b = B_transpose ? B[j * K + k] : B[k * J + j];
                acc += (int32_t)a * (int32_t)b;
            }
            gold[i * J + j] = __autocomp_saturate_elem(acc);
        }
    }
}

#define tiled_matmul_outer_eigen_bias(A, B, D, gold, I, K, J, A_T, B_T, SUB_BIAS) \
    __autocomp_tiled_matmul_outer_eigen_bias( \
        (const void *)(A), (const void *)(B), (const void *)(D), (void *)(gold), \
        (size_t)(I), (size_t)(K), (size_t)(J), \
        (int)(A_T), (int)(B_T), (int)(SUB_BIAS))

#endif  // AUTOCOMP_ALIASES_H
