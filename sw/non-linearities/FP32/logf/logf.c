/*
 * Copyright (C) 2021 ETH Zurich and University of Bologna
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
 * 
 * --------------------------------------------------------------------------
 * 
 * RVV logf (SEW=32) — strip-mined, maskless, runtime-selectable LMUL (8/4/2).
 *
 * Domain reduction:
 *   x = m * 2^e, with m ∈ [1, 2).  ln(x) = ln(m) + e * ln(2)
 *
 * Approximation (degree-4 on t = m - 1):
 *   ln(m) ≈ t * (L1 + t*(L2 + t*(L3 + t*L4)))
 *
 * LMUL=8 variant uses only v0, v8, v16, v24 (no spills).
 * Two-core split: core 0 → [0, N/2), core 1 → [N/2, N)
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
#define LMUL_MODE 8     // set to 8/4/2 at compile time, e.g. -DLMUL_MODE=4
#endif

#define THRESHOLD 0.00010f

// Degree-4 Chebyshev-like fit for ln(1+t) on t∈[0,1)
#define L1   0.99999994f
#define L2  -0.49999905f
#define L3   0.33333197f
#define L4  -0.24999910f

// ln(2) split (higher precision accumulation with two FMAs)
#define LN2_HI  0.693145751953125f        // high part
#define LN2_LO  1.428606765330187e-06f    // low part so that LN2_HI + LN2_LO ≈ ln(2)

// L1 pointers shared in L1 (published by core 0)
static float *g_in  = NULL;
static float *g_out = NULL;

/* ========================= LMUL = 8 =========================
 * Reg plan (SEW=32, LMUL=8),  4 vector registers:
 *   v8  : x (float)
 *   v0  : xi / scalar ones (int/float-as-bits)
 *   v16 : m -> t -> ln(m) (float)
 *   v24 : e (int/float) or poly accumulator / temps
 */
static inline void vlogf_m8_strip(const float* inp, float* out, int N) {
    const float *pin  = inp;
    float       *pout = out;
    int remaining = N;

    while (remaining > 0) {
        unsigned long vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma"
                     : "=r"(vl) : "r"(remaining) : "memory");

        // Load x → v8; bitwise copy → v0
        asm volatile("vle32.v v8, (%0)" :: "r"(pin) : "memory");
        asm volatile("vmv.v.v  v0, v8");

        // e = ((xi >> 23) - 127)  (int in v24)
        asm volatile("vmv.v.v   v24, v0");
        asm volatile("vsrl.vi   v24, v24, 23");
        asm volatile("vmv.v.x   v16, %0" :: "r"(127));
        asm volatile("vsub.vv   v24, v24, v16");

        // m in [1,2): xi = (xi & 0x7FFFFF) | 0x3F800000
        asm volatile("vmv.v.x   v16, %0" :: "r"(0x007FFFFFu));
        asm volatile("vand.vv   v0,  v0,  v16");
        asm volatile("vmv.v.x   v16, %0" :: "r"(0x3F800000u));
        asm volatile("vor.vv    v0,  v0,  v16");

        // t = m - 1.0  (m as float → v16)
        asm volatile("vmv.v.v   v16, v0");
        asm volatile("vfmv.v.f  v0,  %0" :: "f"(1.0f));
        asm volatile("vfsub.vv  v16, v16, v0");   // v16 = t

        // ln(m) poly: v16 = t * (L1 + t*(L2 + t*(L3 + t*L4)))
        asm volatile("vfmv.v.f  v24, %0" :: "f"(L4));
        asm volatile("vfmul.vv  v24, v24, v16");
        asm volatile("vfadd.vf  v24, v24, %0" :: "f"(L3));
        asm volatile("vfmul.vv  v24, v24, v16");
        asm volatile("vfadd.vf  v24, v24, %0" :: "f"(L2));
        asm volatile("vfmul.vv  v24, v24, v16");
        asm volatile("vfadd.vf  v24, v24, %0" :: "f"(L1));
        asm volatile("vfmul.vv  v16, v16, v24");  // v16 = ln(m)

        // e again (cheap), convert to float
        asm volatile("vmv.v.v   v24, v8");
        asm volatile("vsrl.vi   v24, v24, 23");
        asm volatile("vmv.v.x   v0,  %0" :: "r"(127));
        asm volatile("vsub.vv   v24, v24, v0");
        asm volatile("vfcvt.f.x.v v24, v24");     // float(e)

        // ln(x) = ln(m) + e*ln2   (split for precision)
        asm volatile("vfmacc.vf v16, %0, v24" :: "f"(LN2_HI));
        asm volatile("vfmacc.vf v16, %0, v24" :: "f"(LN2_LO));

        // Store
        asm volatile("vse32.v v16, (%0)" :: "r"(pout) : "memory");

        pin  += vl;
        pout += vl;
        remaining -= (int)vl;
    }
}

/* ========================= LMUL = 4 =========================
 * Wider register set; conventional staging.
 */
static inline void vlogf_m4_strip(const float* inp, float* out, int N) {
    const float *pin  = inp;
    float       *pout = out;
    int remaining = N;

    while (remaining > 0) {
        unsigned long vl;
        asm volatile("vsetvli %0, %1, e32, m4, ta, ma"
                     : "=r"(vl) : "r"(remaining) : "memory");

        // x → v12; xi → v8
        asm volatile("vle32.v v12, (%0)" :: "r"(pin) : "memory");
        asm volatile("vmv.v.v  v8,  v12");

        // e = ((xi>>23)-127) (int in v28)
        asm volatile("vmv.v.v   v28, v8");
        asm volatile("vsrl.vi   v28, v28, 23");
        asm volatile("vmv.v.x   v4, %0" :: "r"(127));
        asm volatile("vsub.vv   v28, v28, v4");

        // m = (xi & 0x7FFFFF) | 0x3F800000
        asm volatile("vmv.v.x   v4, %0" :: "r"(0x007FFFFFu));
        asm volatile("vand.vv   v8, v8, v4");
        asm volatile("vmv.v.x   v4, %0" :: "r"(0x3F800000u));
        asm volatile("vor.vv    v8, v8, v4");

        // t = m - 1
        asm volatile("vmv.v.v   v16, v8");
        asm volatile("vfmv.v.f  v4,  %0" :: "f"(1.0f));
        asm volatile("vfsub.vv  v16, v16, v4");

        // ln(m) poly → v16
        asm volatile("vfmv.v.f  v24, %0" :: "f"(L4));
        asm volatile("vfmul.vv  v24, v24, v16");
        asm volatile("vfadd.vf  v24, v24, %0" :: "f"(L3));
        asm volatile("vfmul.vv  v24, v24, v16");
        asm volatile("vfadd.vf  v24, v24, %0" :: "f"(L2));
        asm volatile("vfmul.vv  v24, v24, v16");
        asm volatile("vfadd.vf  v24, v24, %0" :: "f"(L1));
        asm volatile("vfmul.vv  v16, v16, v24");

        // ln(x) = ln(m) + e*ln2  (split for precision)
        asm volatile("vfcvt.f.x.v v24, v28");
        asm volatile("vfmacc.vf   v16, %0, v24" :: "f"(LN2_HI));
        asm volatile("vfmacc.vf   v16, %0, v24" :: "f"(LN2_LO));

        asm volatile("vse32.v v16, (%0)" :: "r"(pout) : "memory");

        pin  += vl;
        pout += vl;
        remaining -= (int)vl;
    }
}

/* ========================= LMUL = 2 ========================= */
static inline void vlogf_m2_strip(const float* inp, float* out, int N) {
    const float *pin  = inp;
    float       *pout = out;
    int remaining = N;

    while (remaining > 0) {
        unsigned long vl;
        asm volatile("vsetvli %0, %1, e32, m2, ta, ma"
                     : "=r"(vl) : "r"(remaining) : "memory");

        // x → v4; xi → v22
        asm volatile("vle32.v v4, (%0)" :: "r"(pin) : "memory");
        asm volatile("vmv.v.v  v22, v4");

        // e = ((xi>>23)-127) (int in v24)
        asm volatile("vmv.v.v   v24, v22");
        asm volatile("vsrl.vi   v24, v24, 23");
        asm volatile("vmv.v.x   v2, %0" :: "r"(127));
        asm volatile("vsub.vv   v24, v24, v2");

        // m = (xi & 0x7FFFFF) | 0x3F800000
        asm volatile("vmv.v.x   v2, %0" :: "r"(0x007FFFFFu));
        asm volatile("vand.vv   v22, v22, v2");
        asm volatile("vmv.v.x   v2, %0" :: "r"(0x3F800000u));
        asm volatile("vor.vv    v22, v22, v2");

        // t = m - 1
        asm volatile("vmv.v.v   v6, v22");
        asm volatile("vfmv.v.f  v2,  %0" :: "f"(1.0f));
        asm volatile("vfsub.vv  v6, v6, v2");

        // ln(m) poly → v6
        asm volatile("vfmv.v.f  v8, %0" :: "f"(L4));
        asm volatile("vfmul.vv  v8, v8, v6");
        asm volatile("vfadd.vf  v8, v8, %0" :: "f"(L3));
        asm volatile("vfmul.vv  v8, v8, v6");
        asm volatile("vfadd.vf  v8, v8, %0" :: "f"(L2));
        asm volatile("vfmul.vv  v8, v8, v6");
        asm volatile("vfadd.vf  v8, v8, %0" :: "f"(L1));
        asm volatile("vfmul.vv  v6, v6, v8");

        // ln(x) = ln(m) + e*ln2  (split for precision)
        asm volatile("vfcvt.f.x.v v8, v24");
        asm volatile("vfmacc.vf   v6, %0, v8" :: "f"(LN2_HI));
        asm volatile("vfmacc.vf   v6, %0, v8" :: "f"(LN2_LO));

        asm volatile("vse32.v v6, (%0)" :: "r"(pout) : "memory");

        pin  += vl;
        pout += vl;
        remaining -= (int)vl;
    }
}

/* ------------------- Kernel selector ------------------- */
static inline void vlogf_strip(const float* inp, float* out, int N) {
#if   (LMUL_MODE == 8)
    vlogf_m8_strip(inp, out, N);
#elif (LMUL_MODE == 4)
    vlogf_m4_strip(inp, out, N);
#elif (LMUL_MODE == 2)
    vlogf_m2_strip(inp, out, N);
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
}

/* ----------------------------- Main ----------------------------- */
int main(void) {
    const unsigned int cid = snrt_cluster_core_idx();
    snrt_cluster_hw_barrier();

    const int B = B_Size, C = C_Size, T = T_Size;
    const int N = B * T * C;

    if (cid == 0) {
        g_in  = (float*)snrt_l1alloc(N * sizeof(float));
        g_out = (float*)snrt_l1alloc(N * sizeof(float));
        if (!g_in || !g_out) { printf("alloc failed\n"); return 1; }

        // Load input from DRAM once, publish to other core
        snrt_dma_start_1d(g_in, data1_dram, N * sizeof(float));
        snrt_dma_wait_all();
        memset(g_out, 0, N * sizeof(float));
    }
    snrt_cluster_hw_barrier();  // publish g_in/g_out

    // Two-core split: core 0 does first half, core 1 does second half
    const int mid   = N >> 1;
    const int start = (cid == 0) ? 0   : mid;
    const int count = (cid == 0) ? mid : (N - mid);

    // Per-core compute
    if (count > 0) {
        start_kernel();
        unsigned t0 = benchmark_get_cycle();

        vlogf_strip(g_in + start, g_out + start, count);

        unsigned cycles = benchmark_get_cycle() - t0;
        stop_kernel();
        printf("[logf LMUL=%d] core %u cycles: %u\n", LMUL_MODE, cid, cycles);
    }

    snrt_cluster_hw_barrier();

    // Validate on core 0
    if (cid == 0) {
        printf("CHECK RESULTS (log)\n");
        check_result(g_in, g_out, outL, N);
    }

    snrt_cluster_hw_barrier();
    return 0;
}
