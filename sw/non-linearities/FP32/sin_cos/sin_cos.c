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
 * -----------------------------------------------------------------------
 * RVV sincos (SEW=32) — strip-mined, maskless, runtime-selectable LMUL.
 *
 * Two kernels:
 *   - LMUL=4: compact, proven baseline.
 *   - LMUL=2: uses more registers and interleaves Horner steps for higher FPU utilization.
 *
 * Parallel execution:
 *   - Core 0 computes the first slice [0 .. split)
 *   - Core 1 computes the second slice [split .. N)
 *   - Generalized: each core i gets [floor(i*N/C) .. floor((i+1)*N/C))
 *
 * Quadrant remap (no masks, pure arithmetic):
 *   b0 =  q & 1
 *   b1 = (q >> 1) & 1
 *   sin_mag = (1-b0)*s0 + b0*c0
 *   cos_mag = (1-b0)*c0 + b0*s0
 *   sin = (1 - 2*b1) * sin_mag
 *   cos = (1 - 2*(b1 ^ b0)) * cos_mag
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
// Set to 4 or 2 (override with -DLMUL_MODE=4 or -DLMUL_MODE=2)
#define LMUL_MODE 4
#endif

#define THRESHOLD 0.00010f

// Shared working buffers (set by core 0, read by others after a barrier)
static float *g_inp  = NULL;
static float *g_outc = NULL;
static float *g_outs = NULL;

// Polynomial and reduction constants (match golden reference)
static inline void sincos_consts(float *INVPIO2, float *PIO2_HI, float *PIO2_LO,
                                 float *S1, float *S3, float *S5, float *S7,
                                 float *C2, float *C4, float *C6) {
    *INVPIO2 = 0.6366197723675814f;       // 2/pi
    *PIO2_HI = 1.5707962512969971f;       // Cody–Waite split (hi)
    *PIO2_LO = 7.5497894158615964e-08f;   // Cody–Waite split (lo)

    *S1 =  1.0f;
    *S3 = -0.1666666716337204f;
    *S5 =  0.008333333767950535f;
    *S7 = -0.0001984126984126984f;

    *C2 = -0.5f;
    *C4 =  0.0416666679084301f;
    *C6 = -0.0013888889225196834f;
}

/* ========================= LMUL = 4 =========================
 * Register plan (LMUL=4, aligned):
 *   v4   : y | kf | b0f | sign_c (float) | temps
 *   v8   : k (int) | b0 (int)
 *   v12  : x | r | (1-b0f) | sign_s (float)
 *   v16  : +s0  -> sin_mag -> final sin
 *   v20  : z | s0_copy | temps
 *   v24  : +c0  -> cos_mag -> final cos  (kept stable during blending)
 *   v28  : q (int)
 */
static inline void fast_sincos_poly_f32_m4_strip(
    const float* inp, float* outs, float* outc, int N) {

    float INVPIO2, PIO2_HI, PIO2_LO, S1, S3, S5, S7, C2, C4, C6;
    sincos_consts(&INVPIO2, &PIO2_HI, &PIO2_LO, &S1, &S3, &S5, &S7, &C2, &C4, &C6);

    const float *pin   = inp;
    float       *poutS = outs;
    float       *poutC = outc;

    int remaining = N;

    while (remaining > 0) {
        unsigned long vl_local;
        asm volatile("vsetvli %0, %1, e32, m4, ta, ma"
                     : "=r"(vl_local) : "r"(remaining) : "memory");

        // Load x -> v12
        asm volatile("vle32.v v12, (%0)" :: "r"(pin) : "memory");

        // Range reduction: k = trunc(x*2/pi + copysign(0.5, x*2/pi))
        asm volatile("vfmul.vf v4,  v12, %[A]" :: [A] "f"(INVPIO2));  // v4 = y
        asm volatile("vfmv.v.f  v20, %[H]" :: [H] "f"(0.5f));
        asm volatile("vfsgnj.vv v20, v20, v4");                       // copysign
        asm volatile("vfadd.vv  v4,  v4,  v20");                      // y += bias
        asm volatile("vfcvt.rtz.x.f.v v8,  v4");                      // v8 = k (int)
        asm volatile("vfcvt.f.x.v     v4,  v8");                      // v4 = kf

        // r = x - kf*PIO2_HI - kf*PIO2_LO  (into v12)
        asm volatile("vfnmsac.vf v12, %[HI], v4" :: [HI] "f"(PIO2_HI));
        asm volatile("vfnmsac.vf v12, %[LO], v4" :: [LO] "f"(PIO2_LO)); // v12 = r

        // z = r^2 -> v20
        asm volatile("vfmul.vv  v20, v12, v12");

        // COS polynomial -> v24 = +c0
        asm volatile("vfmv.v.f  v24, %[C6]" :: [C6] "f"(C6));
        asm volatile("vfmul.vv  v24, v24, v20");
        asm volatile("vfadd.vf  v24, v24, %[C4]" :: [C4] "f"(C4));
        asm volatile("vfmul.vv  v24, v24, v20");
        asm volatile("vfadd.vf  v24, v24, %[C2]" :: [C2] "f"(C2));
        asm volatile("vfmul.vv  v24, v24, v20");
        asm volatile("vfadd.vf  v24, v24, %[ONE]" :: [ONE] "f"(1.0f));  // +c0

        // SIN polynomial -> v16 = +s0  (keep a copy in v20)
        asm volatile("vfmv.v.f  v16, %[S7]" :: [S7] "f"(S7));
        asm volatile("vfmul.vv  v16, v16, v20");
        asm volatile("vfadd.vf  v16, v16, %[S5]" :: [S5] "f"(S5));
        asm volatile("vfmul.vv  v16, v16, v20");
        asm volatile("vfadd.vf  v16, v16, %[S3]" :: [S3] "f"(S3));
        asm volatile("vfmul.vv  v16, v16, v20");
        asm volatile("vfadd.vf  v16, v16, %[S1]" :: [S1] "f"(S1));
        asm volatile("vfmul.vv  v16, v16, v12");                      // +s0
        asm volatile("vmv.v.v   v20, v16");                           // s0_copy

        // q = (k & 3), b0 = (q & 1)
        asm volatile("vand.vi   v28, v8, 3");                         // q
        asm volatile("vand.vi   v8,  v28, 1");                        // b0

        // b0f in v4, (1-b0f) in v12
        asm volatile("vfcvt.f.x.v v4,  v8");                          // b0f
        asm volatile("vfmv.v.f    v12, %[ONE]" :: [ONE] "f"(1.0f));
        asm volatile("vfsub.vv    v12, v12, v4");                     // 1 - b0f

        // Blend magnitudes
        // sin_mag -> v16 = (1-b0)*s0 + b0*c0
        asm volatile("vfmul.vv    v16, v12, v16");
        asm volatile("vfmacc.vv   v16, v4,  v24");
        // cos_mag -> v24 = (1-b0)*c0 + b0*s0_copy
        asm volatile("vfmul.vv    v24, v12, v24");
        asm volatile("vfmacc.vv   v24, v4,  v20");

        // b1 = (q>>1) & 1
        asm volatile("vsrl.vi     v20, v28, 1");
        asm volatile("vand.vi     v20, v20, 1");

        // sign_s (int in v12): 1 - 2*b1
        asm volatile("vadd.vv     v12, v20, v20");
        asm volatile("vrsub.vi    v12, v12, 1");
        // sign_c (int in v4): 1 - 2*(b1^b0)
        asm volatile("vxor.vv     v4,  v20, v8");
        asm volatile("vadd.vv     v4,  v4,  v4");
        asm volatile("vrsub.vi    v4,  v4,  1");

        // Convert signs to float, apply
        asm volatile("vfcvt.f.x.v v20, v12");                         // sign_s
        asm volatile("vfcvt.f.x.v v12, v4");                          // sign_c
        asm volatile("vfmul.vv    v16, v16, v20");                    // sin *= sign_s
        asm volatile("vfmul.vv    v24, v24, v12");                    // cos *= sign_c

        // Store
        asm volatile("vse32.v v16, (%0)" :: "r"(poutS) : "memory");
        asm volatile("vse32.v v24, (%0)" :: "r"(poutC) : "memory");

        pin   += vl_local;
        poutS += vl_local;
        poutC += vl_local;
        remaining -= (int)vl_local;
    }
}

/* ========================= LMUL = 2 =========================
 * Wider register budget; interleaves sin/cos Horner steps to improve
 * FPU utilization (issue mix of mul/add on independent accumulators).
 *
 * Register plan (LMUL=2, aligned):
 *   v2  : y | kf | b0f (float)
 *   v4  : x | r (float)
 *   v6  : z = r^2 (float)
 *   v8  : c_acc -> +c0 (float)
 *   v10 : s_acc -> +s0 (float)
 *   v12 : s0_copy (float)
 *   v14 : (1-b0f) (float)
 *   v16 : cos_mag -> final cos (float)
 *   v18 : sign_s (float)
 *   v20 : sign_c (float)
 *   v22 : k (int)       v24 : q (int)
 *   v26 : b0 (int)      v28 : b1 (int)
 *   v30 : tmp int
 */
static inline void fast_sincos_poly_f32_m2_strip(
    const float* inp, float* outs, float* outc, int N) {

    float INVPIO2, PIO2_HI, PIO2_LO, S1, S3, S5, S7, C2, C4, C6;
    sincos_consts(&INVPIO2, &PIO2_HI, &PIO2_LO, &S1, &S3, &S5, &S7, &C2, &C4, &C6);

    const float *pin   = inp;
    float       *poutS = outs;
    float       *poutC = outc;

    int remaining = N;

    while (remaining > 0) {
        unsigned long vl_local;
        asm volatile("vsetvli %0, %1, e32, m2, ta, ma"
                     : "=r"(vl_local) : "r"(remaining) : "memory");

        // Load x -> v4
        asm volatile("vle32.v v4, (%0)" :: "r"(pin) : "memory");

        // Range reduction
        asm volatile("vfmul.vf v2,  v4,  %[A]" :: [A] "f"(INVPIO2));  // y
        asm volatile("vfmv.v.f  v16, %[H]" :: [H] "f"(0.5f));
        asm volatile("vfsgnj.vv v16, v16, v2");                       // copysign
        asm volatile("vfadd.vv  v2,  v2,  v16");                      // y += bias
        asm volatile("vfcvt.rtz.x.f.v v22, v2");                      // k (int)
        asm volatile("vfcvt.f.x.v     v2,  v22");                     // kf

        // r and z
        asm volatile("vfnmsac.vf v4, %[HI], v2" :: [HI] "f"(PIO2_HI));
        asm volatile("vfnmsac.vf v4, %[LO], v2" :: [LO] "f"(PIO2_LO)); // r
        asm volatile("vfmul.vv  v6,  v4,  v4");                        // z=r^2

        // Interleaved Horner for cos/sin
        asm volatile("vfmv.v.f  v8,  %[C6]" :: [C6] "f"(C6));          // c_acc
        asm volatile("vfmv.v.f  v10, %[S7]" :: [S7] "f"(S7));          // s_acc

        asm volatile("vfmul.vv  v8,  v8,  v6");
        asm volatile("vfmul.vv  v10, v10, v6");
        asm volatile("vfadd.vf  v8,  v8,  %[C4]" :: [C4] "f"(C4));
        asm volatile("vfadd.vf  v10, v10, %[S5]" :: [S5] "f"(S5));

        asm volatile("vfmul.vv  v8,  v8,  v6");
        asm volatile("vfmul.vv  v10, v10, v6");
        asm volatile("vfadd.vf  v8,  v8,  %[C2]" :: [C2] "f"(C2));
        asm volatile("vfadd.vf  v10, v10, %[S3]" :: [S3] "f"(S3));

        asm volatile("vfmul.vv  v8,  v8,  v6");
        asm volatile("vfmul.vv  v10, v10, v6");
        asm volatile("vfadd.vf  v8,  v8,  %[ONE]" :: [ONE] "f"(1.0f)); // +1
        asm volatile("vfadd.vf  v10, v10, %[S1]" :: [S1] "f"(S1));

        asm volatile("vfmul.vv  v10, v10, v4");                        // s0
        asm volatile("vmv.v.v   v12, v10");                            // s0_copy

        // Quadrant decode and blend
        asm volatile("vand.vi   v24, v22, 3");                         // q
        asm volatile("vand.vi   v26, v24, 1");                         // b0
        asm volatile("vfcvt.f.x.v v2,  v26");                          // b0f
        asm volatile("vfmv.v.f    v14, %[ONE]" :: [ONE] "f"(1.0f));
        asm volatile("vfsub.vv    v14, v14, v2");                      // 1-b0f

        // sin_mag in v10; cos_mag in v16
        asm volatile("vfmul.vv    v10, v14, v10");                     // (1-b0)*s0
        asm volatile("vfmacc.vv   v10, v2,  v8");                      // + b0*c0
        asm volatile("vfmul.vv    v16, v14, v8");                      // (1-b0)*c0
        asm volatile("vfmacc.vv   v16, v2,  v12");                     // + b0*s0_copy

        // Signs
        asm volatile("vsrl.vi     v28, v24, 1");                       // b1
        asm volatile("vand.vi     v28, v28, 1");

        asm volatile("vadd.vv     v30, v28, v28");                     // 2*b1
        asm volatile("vrsub.vi    v30, v30, 1");                       // 1-2*b1
        asm volatile("vxor.vv     v22, v28, v26");                     // b1^b0
        asm volatile("vadd.vv     v22, v22, v22");                     // 2*(b1^b0)
        asm volatile("vrsub.vi    v22, v22, 1");                       // 1-2*(b1^b0)

        asm volatile("vfcvt.f.x.v v18, v30");                          // sign_s
        asm volatile("vfcvt.f.x.v v20, v22");                          // sign_c
        asm volatile("vfmul.vv    v10, v10, v18");                     // sin *= sign_s
        asm volatile("vfmul.vv    v16, v16, v20");                     // cos *= sign_c

        // Store
        asm volatile("vse32.v v10, (%0)" :: "r"(poutS) : "memory");
        asm volatile("vse32.v v16, (%0)" :: "r"(poutC) : "memory");

        pin   += vl_local;
        poutS += vl_local;
        poutC += vl_local;
        remaining -= (int)vl_local;
    }
}

// Wrapper that selects the LMUL variant based on LMUL_MODE
static inline void fast_sincos_poly_f32_strip(
    const float* inp, float* outs, float* outc, int N) {
#if   (LMUL_MODE == 4)
    fast_sincos_poly_f32_m4_strip(inp, outs, outc, N);
#elif (LMUL_MODE == 2)
    fast_sincos_poly_f32_m2_strip(inp, outs, outc, N);
#else
#   error "LMUL_MODE must be 4 or 2"
#endif
}

// Golden checker
static void check_result(const float *input, const float *x, const float *ref, int r) {
    int err = 0;
    for (int i = 0; i < r; i++) {
        float diff = fabsf(x[i] - ref[i]);
        printf("At index %d:\t, value %f\t expected %f\t real %f\t error %f\n",
                i, input[i], ref[i], x[i], diff);
        
    }
}

int main() {
    volatile int errors = 0;

    const unsigned int cid   = snrt_cluster_core_idx();
    const unsigned int cores = snrt_cluster_core_num();
    snrt_cluster_hw_barrier();

    // Problem size from data headers
    const int B = B_Size, C = C_Size, T = T_Size;
    const int N = B * T * C;

    // Core 0 allocates in L1 and DMA-loads input; outputs are shared.
    if (cid == 0) {
        g_inp  = (float*)snrt_l1alloc(N * sizeof(float));
        g_outc = (float*)snrt_l1alloc(N * sizeof(float));
        g_outs = (float*)snrt_l1alloc(N * sizeof(float));
        if (!g_inp || !g_outc || !g_outs) { printf("alloc failed\n"); return 1; }

        snrt_dma_start_1d(g_inp, data1_dram, N * sizeof(float));
        snrt_dma_wait_all();

        // Initialize outputs (optional)
        memset(g_outc, 0, N * sizeof(float));
        memset(g_outs, 0, N * sizeof(float));
    }

    // Ensure all cores see initialized global pointers and input data.
    snrt_cluster_hw_barrier();

    // Compute per-core slice: [start, end)
    // For two cores, this yields halves; for C cores, balanced contiguous blocks.
    const int start = (int)((int64_t)cid * N / (int64_t)cores);
    const int end   = (int)((int64_t)(cid + 1) * N / (int64_t)cores);
    const int len   = end - start;

    // Synchronize before timing the parallel section
    snrt_cluster_hw_barrier();

    unsigned t0 = 0;
    if (cid == 0) {
        start_kernel();
        t0 = benchmark_get_cycle();
    }

    // Each core computes its disjoint slice (no load/store collisions).
    if (len > 0) {
        fast_sincos_poly_f32_strip(g_inp  + start,
                                   g_outs + start,
                                   g_outc + start,
                                   len);
    }

    // Wait for all cores to finish before stopping the timer and checking.
    snrt_cluster_hw_barrier();

    if (cid == 0) {
        unsigned cycles = benchmark_get_cycle() - t0;
        stop_kernel();

#if (LMUL_MODE == 4)
        printf("[LMUL=4] ");
#else
        printf("[LMUL=2] ");
#endif
        printf("Parallel sincos cycles: %u (cores=%u)\n", cycles, cores);
        printf("CHECK RESULTS (cos)\n");
        check_result(g_inp,g_outc, outC, N);
        printf("CHECK RESULTS (sin)\n");
        check_result(g_inp , g_outs, outS, N);
        
    }

    snrt_cluster_hw_barrier();
    return errors;
}
