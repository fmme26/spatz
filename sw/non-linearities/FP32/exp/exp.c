/*
 * Copyright (C) 2021 ETH Zurich
 * and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * -----------------------------------------------------------------------
 *
 * RVV expf (SEW=32) — strip-mined, maskless, runtime-selectable LMUL (8/4/2).
 *
 * Build-time params:
 *   -DLMUL_MODE=8|4|2     (default: 8)
 *   -DEXP_DEG=0|2|3|4     (0 = Schraudolph fast exp; 2/3/4 = Chebyshev + RNE reduction)
 *
 * Identities / approach:
 *   For EXP_DEG = 0 (Schraudolph):
 *     E(x) ≈ *(float*)&( (uint32_t)( x * C + B ) )
 *     with C = 2^23 / ln(2) ≈ 12102203.0f,  B ≈ 1064866805.0f
 *
 *   For EXP_DEG = 2/3/4 (RNE reduction + Chebyshev in r):
 *     y = x*LOG2E ; k = nearint(y) (RNE); r = x - k*LN2
 *     exp(x) ≈ P(r) * 2^k
 *     Degree-2:  P(r) = A0 + r*(A1 + r*A2)
 *     Degree-3:  P(r) = A0 + r*(A1 + r*(A2 + r*A3))
 *     Degree-4:  P(r) = A0 + r*(A1 + r*(A2 + r*(A3 + r*A4)))
 *
 * Notes:
 *   - Domain: all finite x (no special-case handling here).
 *   - Two-core split: core 0 → [0, N/2), core 1 → [N/2, N).
 *   - Golden assumed in golden/gold.h as outE.
 */

#include <stdint.h>
#include <math.h>
#include <snrt.h>
#include "printf.h"
#include <spatz_cluster_peripheral.h>
#include "data/data.h"
#include "golden/gold.h"
#include "benchmark/benchmark.c"
#include <stdio.h>
#include <string.h>

#ifndef LMUL_MODE
#define LMUL_MODE 8
#endif

#ifndef EXP_DEG
#define EXP_DEG 0
#endif

#define THRESHOLD 0.00010f

/* Reduction constants */
#define LOG2E_F  1.4426950408889634f
#define LN2_F    0.6931471805599453f

/* Degree-0 Schraudolph constants */
#define SCH_C  12102203.0f     /* 2^23 / ln(2) */
#define SCH_B  1064866805.0f   /* bias near (127<<23), tuned */

/* ========================= LMUL = 8 =========================
 * v0  : x / r / P / final
 * v8  : temp P or poly staging
 * v16 : k (i32) → bits(2^k)
 * v24 : temp
 */
static inline void vexp_m8_strip(const float* inp, float* out, int N) {
    const float *pin  = inp;
    float       *pout = out;
    int remaining = N;

    while (remaining > 0) {
        unsigned long vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma"
                     : "=r"(vl) : "r"(remaining) : "memory");

        asm volatile("vle32.v   v0, (%0)" :: "r"(pin) : "memory");

#if (EXP_DEG == 0)
        /* ---- Schraudolph fast exp ---- */
        asm volatile("vfmul.vf  v24, v0,  %[C]" :: [C]"f"(SCH_C));
        asm volatile("vfadd.vf  v24, v24, %[B]" :: [B]"f"(SCH_B));
        asm volatile("vfcvt.rtz.xu.f.v v24, v24");
        asm volatile("vse32.v   v24, (%0)" :: "r"(pout) : "memory");

#else  /* EXP_DEG >= 2: RNE reduction + Chebyshev */
        /* y = x*LOG2E; k=nearint(y); r = x - k*LN2 */
        asm volatile("vfmul.vf        v24, v0,  %[log2e]" :: [log2e]"f"(LOG2E_F));
        asm volatile("vfcvt.x.f.v     v16, v24");                                   /* k (i32, RNE) */
        asm volatile("vfcvt.f.x.v     v24, v16");                                   /* k (f32) */
        asm volatile("vfnmsac.vf      v0,  %[ln2], v24" :: [ln2]"f"(LN2_F));        /* r in v0 */

#  if   (EXP_DEG == 2)
        /* P(r) = A0 + r*(A1 + r*A2) */
        asm volatile("vfmv.v.f        v8,  %[a1]" :: [a1]"f"(1.01508951f));
        asm volatile("vfmacc.vf       v8,  %[a2], v0" :: [a2]"f"(0.50502354f));
        asm volatile("vfmv.v.f        v24, %[a0]" :: [a0]"f"(0.99992448f));
        asm volatile("vfmacc.vv       v24, v0,  v8");                                /* P in v24 */
#  elif (EXP_DEG == 3)
        /* P(r) = A0 + r*(A1 + r*(A2 + r*A3)) */
        asm volatile("vfmv.v.f        v8,  %[a2]" :: [a2]"f"(0.50502354f));
        asm volatile("vfmacc.vf       v8,  %[a3], v0" :: [a3]"f"(0.16792160f));     /* A2 + r*A3 */
        asm volatile("vfmul.vv        v8,  v8,  v0");                                /* r*(...) */
        asm volatile("vfadd.vf        v8,  v8,  %[a1]" :: [a1]"f"(0.99996230f));    /* A1 + r*(...) */
        asm volatile("vfmul.vv        v24, v0,  v8");                                /* r*u */
        asm volatile("vfadd.vf        v24, v24, %[a0]" :: [a0]"f"(0.99992450f));    /* P */
#  elif (EXP_DEG == 4)
        /* P(r) = A0 + r*(A1 + r*(A2 + r*(A3 + r*A4))) */
        asm volatile("vfmv.v.f        v8,  %[a3]" :: [a3]"f"(0.16792161f));
        asm volatile("vfmacc.vf       v8,  %[a4], v0" :: [a4]"f"(0.04191753f));     /* A3 + r*A4 */
        asm volatile("vfmul.vv        v8,  v8,  v0");                                /* r*s */
        asm volatile("vfadd.vf        v8,  v8,  %[a2]" :: [a2]"f"(0.49998870f));    /* A2 + r*s */
        asm volatile("vfmul.vv        v8,  v8,  v0");                                /* r*t */
        asm volatile("vfadd.vf        v8,  v8,  %[a1]" :: [a1]"f"(0.99996230f));    /* A1 + r*t */
        asm volatile("vfmul.vv        v24, v0,  v8");                                /* r*u */
        asm volatile("vfadd.vf        v24, v24, %[a0]" :: [a0]"f"(1.00000000f));    /* P */
#  endif

        /* y = P * 2^k via exponent injection */
        asm volatile("vadd.vx         v16, v16, %[bias]" :: [bias]"r"(127));
        asm volatile("vsll.vi         v16, v16, 23");
        asm volatile("vfmul.vv        v0,  v24, v16");                               /* result in v0 */
        asm volatile("vse32.v         v0, (%0)" :: "r"(pout) : "memory");
#endif

        pin  += vl;
        pout += vl;
        remaining -= (int)vl;
    }
}

/* ========================= LMUL = 4 =========================
 * v12 : x / r / P / result
 * v24 : temp
 * v16 : k (i32) → bits(2^k)
 * v8  : temp
 */
static inline void vexp_m4_strip(const float* inp, float* out, int N) {
    const float *pin  = inp;
    float       *pout = out;
    int remaining = N;

    while (remaining > 0) {
        unsigned long vl;
        asm volatile("vsetvli %0, %1, e32, m4, ta, ma"
                     : "=r"(vl) : "r"(remaining) : "memory");

        asm volatile("vle32.v   v12, (%0)" :: "r"(pin) : "memory");

#if (EXP_DEG == 0)
        asm volatile("vfmul.vf  v24, v12, %[C]" :: [C]"f"(SCH_C));
        asm volatile("vfadd.vf  v24, v24, %[B]" :: [B]"f"(SCH_B));
        asm volatile("vfcvt.rtz.xu.f.v v24, v24");
        asm volatile("vse32.v   v24, (%0)" :: "r"(pout) : "memory");

#else
        asm volatile("vfmul.vf        v24, v12, %[log2e]" :: [log2e]"f"(LOG2E_F));
        asm volatile("vfcvt.x.f.v     v16, v24");                                   /* k */
        asm volatile("vfcvt.f.x.v     v24, v16");                                   /* kf */
        asm volatile("vfnmsac.vf      v12, %[ln2], v24" :: [ln2]"f"(LN2_F));        /* r */

#  if   (EXP_DEG == 2)
        asm volatile("vfmv.v.f        v8,  %[a1]" :: [a1]"f"(1.01508951f));
        asm volatile("vfmacc.vf       v8,  %[a2], v12" :: [a2]"f"(0.50502354f));
        asm volatile("vfmv.v.f        v24, %[a0]" :: [a0]"f"(0.99992448f));
        asm volatile("vfmacc.vv       v24, v12, v8");                                /* P in v24 */
#  elif (EXP_DEG == 3)
        asm volatile("vfmv.v.f        v8,  %[a2]" :: [a2]"f"(0.50502354f));
        asm volatile("vfmacc.vf       v8,  %[a3], v12" :: [a3]"f"(0.16792160f));
        asm volatile("vfmul.vv        v8,  v8,  v12");
        asm volatile("vfadd.vf        v8,  v8,  %[a1]" :: [a1]"f"(0.99996230f));
        asm volatile("vfmul.vv        v24, v12, v8");
        asm volatile("vfadd.vf        v24, v24, %[a0]" :: [a0]"f"(0.99992450f));
#  elif (EXP_DEG == 4)
        asm volatile("vfmv.v.f        v8,  %[a3]" :: [a3]"f"(0.16792161f));
        asm volatile("vfmacc.vf       v8,  %[a4], v12" :: [a4]"f"(0.04191753f));
        asm volatile("vfmul.vv        v8,  v8,  v12");
        asm volatile("vfadd.vf        v8,  v8,  %[a2]" :: [a2]"f"(0.49998870f));
        asm volatile("vfmul.vv        v8,  v8,  v12");
        asm volatile("vfadd.vf        v8,  v8,  %[a1]" :: [a1]"f"(0.99996230f));
        asm volatile("vfmul.vv        v24, v12, v8");
        asm volatile("vfadd.vf        v24, v24, %[a0]" :: [a0]"f"(1.00000000f));
#  endif

        asm volatile("vadd.vx         v16, v16, %[bias]" :: [bias]"r"(127));
        asm volatile("vsll.vi         v16, v16, 23");
        asm volatile("vfmul.vv        v12, v24, v16");
        asm volatile("vse32.v         v12, (%0)" :: "r"(pout) : "memory");
#endif

        pin  += vl;
        pout += vl;
        remaining -= (int)vl;
    }
}

/* ========================= LMUL = 2 =========================
 * v4  : x / r / P / result
 * v10 : temp
 * v8  : k (i32) → bits(2^k)
 * v6  : temp
 */
static inline void vexp_m2_strip(const float* inp, float* out, int N) {
    const float *pin  = inp;
    float       *pout = out;
    int remaining = N;

    while (remaining > 0) {
        unsigned long vl;
        asm volatile("vsetvli %0, %1, e32, m2, ta, ma"
                     : "=r"(vl) : "r"(remaining) : "memory");

        asm volatile("vle32.v   v4, (%0)" :: "r"(pin) : "memory");

#if (EXP_DEG == 0)
        asm volatile("vfmul.vf  v10, v4,  %[C]" :: [C]"f"(SCH_C));
        asm volatile("vfadd.vf  v10, v10, %[B]" :: [B]"f"(SCH_B));
        asm volatile("vfcvt.rtz.xu.f.v v10, v10");
        asm volatile("vse32.v   v10, (%0)" :: "r"(pout) : "memory");

#else
        asm volatile("vfmul.vf        v10, v4,  %[log2e]" :: [log2e]"f"(LOG2E_F));
        asm volatile("vfcvt.x.f.v     v8,  v10");                                   /* k */
        asm volatile("vfcvt.f.x.v     v10, v8");
        asm volatile("vfnmsac.vf      v4,  %[ln2], v10" :: [ln2]"f"(LN2_F));        /* r */

#  if   (EXP_DEG == 2)
        asm volatile("vfmv.v.f        v6,  %[a1]" :: [a1]"f"(1.01508951f));
        asm volatile("vfmacc.vf       v6,  %[a2], v4" :: [a2]"f"(0.50502354f));
        asm volatile("vfmv.v.f        v10, %[a0]" :: [a0]"f"(0.99992448f));
        asm volatile("vfmacc.vv       v10, v4,  v6");                                /* P in v10 */
#  elif (EXP_DEG == 3)
        asm volatile("vfmv.v.f        v6,  %[a2]" :: [a2]"f"(0.50502354f));
        asm volatile("vfmacc.vf       v6,  %[a3], v4" :: [a3]"f"(0.16792160f));
        asm volatile("vfmul.vv        v6,  v6,  v4");
        asm volatile("vfadd.vf        v6,  v6,  %[a1]" :: [a1]"f"(0.99996230f));
        asm volatile("vfmul.vv        v10, v4,  v6");
        asm volatile("vfadd.vf        v10, v10, %[a0]" :: [a0]"f"(0.99992450f));
#  elif (EXP_DEG == 4)
        asm volatile("vfmv.v.f        v6,  %[a3]" :: [a3]"f"(0.16792161f));
        asm volatile("vfmacc.vf       v6,  %[a4], v4" :: [a4]"f"(0.04191753f));
        asm volatile("vfmul.vv        v6,  v6,  v4");
        asm volatile("vfadd.vf        v6,  v6,  %[a2]" :: [a2]"f"(0.49998870f));
        asm volatile("vfmul.vv        v6,  v6,  v4");
        asm volatile("vfadd.vf        v6,  v6,  %[a1]" :: [a1]"f"(0.99996230f));
        asm volatile("vfmul.vv        v10, v4,  v6");
        asm volatile("vfadd.vf        v10, v10, %[a0]" :: [a0]"f"(1.00000000f));
#  endif

        asm volatile("vadd.vx         v8,  v8,  %[bias]" :: [bias]"r"(127));
        asm volatile("vsll.vi         v8,  v8,  23");
        asm volatile("vfmul.vv        v4,  v10, v8");
        asm volatile("vse32.v         v4,  (%0)" :: "r"(pout) : "memory");
#endif

        pin  += vl;
        pout += vl;
        remaining -= (int)vl;
    }
}

/* ------------------- Kernel selector ------------------- */
static inline void vexp_strip(const float* inp, float* out, int N) {
#if   (LMUL_MODE == 8)
    vexp_m8_strip(inp, out, N);
#elif (LMUL_MODE == 4)
    vexp_m4_strip(inp, out, N);
#elif (LMUL_MODE == 2)
    vexp_m2_strip(inp, out, N);
#else
#   error "LMUL_MODE must be 8, 4, or 2"
#endif
}

/* ------------------ Simple golden checker ------------------ */
static void check_result(const float *input, const float *x, const float *ref, int r) {
    int err = 0;
    for (int i = 0; i < r; i++) {
        float diff = fabsf(x[i] - ref[i]);
        printf("At index %d:\t, value %f\t expected %f\t real %f\t error %f\n",
                i, input[i], ref[i], x[i], diff);
        
    }
    // if (err) printf("TEST FAILED with %d errors!!\n", err);
    // else     printf("TEST PASSED!!\n");
}

/* ----------------------------- Main ----------------------------- */
int main(void) {
    const unsigned int cid = snrt_cluster_core_idx();
    snrt_cluster_hw_barrier();

    const int B = B_Size, C = C_Size, T = T_Size;
    const int N = B * T * C;

    static float *g_in  = NULL;
    static float *g_out = NULL;

    if (cid == 0) {
        g_in  = (float*)snrt_l1alloc(N * sizeof(float));
        g_out = (float*)snrt_l1alloc(N * sizeof(float));
        if (!g_in || !g_out) { printf("alloc failed\n"); return 1; }

        /* Load input from DRAM once, publish to other core */
        snrt_dma_start_1d(g_in, data1_dram, N * sizeof(float));
        snrt_dma_wait_all();
        memset(g_out, 0, N * sizeof(float));
    }
    snrt_cluster_hw_barrier();  /* publish g_in/g_out */

    /* Two-core split */
    const int mid   = N >> 1;
    const int start = (cid == 0) ? 0   : mid;
    const int count = (cid == 0) ? mid : (N - mid);

    if (count > 0) {
        start_kernel();
        unsigned t0 = benchmark_get_cycle();

        vexp_strip(g_in + start, g_out + start, count);

        unsigned cycles = benchmark_get_cycle() - t0;
        stop_kernel();
        printf("[exp LMUL=%d DEG=%d] core %u cycles: %u\n", LMUL_MODE, EXP_DEG, cid, cycles);
    }

    snrt_cluster_hw_barrier();

    /* Validate on core 0 (expects golden `outE` in golden/gold.h) */
    if (cid == 0) {
        printf("CHECK RESULTS (exp)\n");
        check_result(g_in,g_out, outE, N);
    }

    snrt_cluster_hw_barrier();
    return 0;
}
