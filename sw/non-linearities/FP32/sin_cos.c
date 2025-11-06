/*
 * Debug RVV sincos with mask dumps (first 8 lanes)
 * - Shows outs/outc[0..7]
 * - Shows mask vectors for q==0..3 (as 0/1 floats)
 * - Shows candidate source vectors: +s0, +c0, -s0, -c0
 */

#include <stdint.h>
#include <math.h>
#include <snrt.h>
#include "printf.h"
#include <spatz_cluster_peripheral.h>
#include "data/data.h"
#include "golden/gold.h"
#include "benchmark/benchmark.c"
//=====================================
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

#define VLMAX     128
#define THRESHOLD 0.00010f
#define SHOW_LANES 8

// ---------------- Debug taps (first chunk) ----------------
static volatile int g_dbg_enable = 0;
static float   *dbg_m0 = NULL, *dbg_m1 = NULL, *dbg_m2 = NULL, *dbg_m3 = NULL; // mask vectors q==0..3 (0/1)
static float   *dbg_s_pos = NULL;  // +s0 (v16)
static float   *dbg_c_pos = NULL;  // +c0 (v24)
static float   *dbg_s_neg = NULL;  // -s0 (v4)
static float   *dbg_c_neg = NULL;  // -c0 (v20)
static int32_t *dbg_k     = NULL;  // k
static int32_t *dbg_q     = NULL;  // q=k&3

// ======================= constants =======================
static inline void sincos_consts(float *INVPIO2, float *PIO2_HI, float *PIO2_LO,
                                 float *S1, float *S3, float *S5, float *S7,
                                 float *C2, float *C4, float *C6)
{
    *INVPIO2 = 0.6366197723675814f;       // 2/pi
    *PIO2_HI = 1.5707962512969971f;       // Cody–Waite split (hi)
    *PIO2_LO = 7.5497894158615964e-08f;   // split (lo)

    *S1 = 1.0f;
    *S3 = -0.1666666716337204f;
    *S5 =  0.008333333767950535f;
    *S7 = -0.0001984126984126984f;

    *C2 = -0.5f;
    *C4 =  0.0416666679084301f;
    *C6 = -0.0013888889225196834f;
}

/*
 * RVV kernel (SEW=32, LMUL=4)
 *
 * Reg plan (LMUL=4):
 *   v0  : mask
 *   v4  : -s0
 *   v8  : k (int)
 *   v12 : x -> r
 *   v16 : +s0
 *   v20 : z (tmp) -> -c0
 *   v24 : +c0
 *   v28 : q = k & 3 (int)
 *
 * No vmerge. Masked stores only.
 */
static inline void fast_sincos_poly_f32_m4_chunk(
    const float* inp, float* outs, float* outc, int vl)
{
    float INVPIO2, PIO2_HI, PIO2_LO, S1, S3, S5, S7, C2, C4, C6;
    sincos_consts(&INVPIO2, &PIO2_HI, &PIO2_LO, &S1, &S3, &S5, &S7, &C2, &C4, &C6);

    unsigned long vl_local;
    asm volatile("vsetvli %0, %1, e32, m4, ta, ma"
                 : "=r"(vl_local) : "r"(vl) : "memory");

    // ===== Load x =====
    asm volatile("vle32.v v12, (%0)" :: "r"(inp) : "memory");      // v12 = x

    // ===== Range reduction (nearest using ±0.5 + trunc) =====
    // v4 = y = x * 2/pi
    asm volatile("vfmul.vf v4,  v12, %[A]" :: [A] "f"(INVPIO2));
    // bias = copysign(0.5, y)
    asm volatile("vfmv.v.f  v24, %[HALF]" :: [HALF] "f"(0.5f));
    asm volatile("vfsgnj.vv v24, v24, v4");
    // y += bias -> trunc
    asm volatile("vfadd.vv  v4,  v4,  v24");
    asm volatile("vfcvt.rtz.x.f.v v8,  v4");                        // k
    if (g_dbg_enable && dbg_k) asm volatile("vse32.v v8, (%0)" :: "r"(dbg_k) : "memory");
    asm volatile("vfcvt.f.x.v     v4,  v8");                        // kf

    // r = x - kf*PIO2_HI - kf*PIO2_LO  (into v12)
    asm volatile("vfnmsac.vf v12, %[HI], v4" :: [HI] "f"(PIO2_HI));
    asm volatile("vfnmsac.vf v12, %[LO], v4" :: [LO] "f"(PIO2_LO)); // v12 = r

    // z = r^2 -> v20
    asm volatile("vfmul.vv  v20, v12, v12");

    // ===== COS core -> v24 (+c0)
    asm volatile("vfmv.v.f  v24, %[C6]" :: [C6] "f"(C6));
    asm volatile("vfmul.vv  v24, v24, v20");
    asm volatile("vfadd.vf  v24, v24, %[C4]" :: [C4] "f"(C4));
    asm volatile("vfmul.vv  v24, v24, v20");
    asm volatile("vfadd.vf  v24, v24, %[C2]" :: [C2] "f"(C2));
    asm volatile("vfmul.vv  v24, v24, v20");
    asm volatile("vfadd.vf  v24, v24, %[ONE]" :: [ONE] "f"(1.0f));  // v24 = +c0

    // ===== SIN core -> v16 (+s0)
    asm volatile("vfmv.v.f  v16, %[S7]" :: [S7] "f"(S7));
    asm volatile("vfmul.vv  v16, v16, v20");
    asm volatile("vfadd.vf  v16, v16, %[S5]" :: [S5] "f"(S5));
    asm volatile("vfmul.vv  v16, v16, v20");
    asm volatile("vfadd.vf  v16, v16, %[S3]" :: [S3] "f"(S3));
    asm volatile("vfmul.vv  v16, v16, v20");
    asm volatile("vfadd.vf  v16, v16, %[S1]" :: [S1] "f"(S1));
    asm volatile("vfmul.vv  v16, v16, v12");                        // v16 = +s0

    // ===== q = k & 3 =====
    asm volatile("vand.vi   v28, v8, 3");                           // v28 = q
    if (g_dbg_enable && dbg_q) asm volatile("vse32.v v28, (%0)" :: "r"(dbg_q) : "memory");

    // ===== Build negatives (robust) =====
    asm volatile("vfneg.v   v4,  v16");                             // v4  = -s0
    asm volatile("vfneg.v   v20, v24");                             // v20 = -c0

    // Tap candidate source vectors
    if (g_dbg_enable) {
        if (dbg_s_pos) asm volatile("vse32.v v16, (%0)" :: "r"(dbg_s_pos) : "memory"); // +s0
        if (dbg_c_pos) asm volatile("vse32.v v24, (%0)" :: "r"(dbg_c_pos) : "memory"); // +c0
        if (dbg_s_neg) asm volatile("vse32.v v4,  (%0)" :: "r"(dbg_s_neg) : "memory"); // -s0
        if (dbg_c_neg) asm volatile("vse32.v v20, (%0)" :: "r"(dbg_c_neg) : "memory"); // -c0
    }

    // ===== Materialize and dump the 4 masks as 0/1 float vectors =====
    if (g_dbg_enable && (dbg_m0 || dbg_m1 || dbg_m2 || dbg_m3)) {
        // v2 = 0.0
        asm volatile("vfmv.v.f  v2, %[ZERO]" :: [ZERO] "f"(0.0f));

        // q==0
        if (dbg_m0) {
            asm volatile("vmseq.vi  v0, v28, 0");
            asm volatile("vmv.v.v   v6, v2");                         // v6 = 0
            asm volatile("vfadd.vf  v6, v6, %[ONE], v0.t" :: [ONE] "f"(1.0f)); // v6 = 1 where mask
            asm volatile("vse32.v   v6, (%0)" :: "r"(dbg_m0) : "memory");
        }
        // q==1
        if (dbg_m1) {
            asm volatile("vmseq.vi  v0, v28, 1");
            asm volatile("vmv.v.v   v6, v2");
            asm volatile("vfadd.vf  v6, v6, %[ONE], v0.t" :: [ONE] "f"(1.0f));
            asm volatile("vse32.v   v6, (%0)" :: "r"(dbg_m1) : "memory");
        }
        // q==2
        if (dbg_m2) {
            asm volatile("vmseq.vi  v0, v28, 2");
            asm volatile("vmv.v.v   v6, v2");
            asm volatile("vfadd.vf  v6, v6, %[ONE], v0.t" :: [ONE] "f"(1.0f));
            asm volatile("vse32.v   v6, (%0)" :: "r"(dbg_m2) : "memory");
        }
        // q==3
        if (dbg_m3) {
            asm volatile("vmseq.vi  v0, v28, 3");
            asm volatile("vmv.v.v   v6, v2");
            asm volatile("vfadd.vf  v6, v6, %[ONE], v0.t" :: [ONE] "f"(1.0f));
            asm volatile("vse32.v   v6, (%0)" :: "r"(dbg_m3) : "memory");
        }
    }

    // ===== Final stores (mask on stores only) =====
    // q==0: sin=+s0 (v16), cos=+c0 (v24)
    asm volatile("vmseq.vi  v0, v28, 0");
    asm volatile("vse32.v   v16, (%0), v0.t" :: "r"(outs)  : "memory");
    asm volatile("vse32.v   v24, (%0), v0.t" :: "r"(outc)  : "memory");

    // q==1: sin=+c0 (v24), cos=-s0 (v4)
    asm volatile("vmseq.vi  v0, v28, 1");
    asm volatile("vse32.v   v24, (%0), v0.t" :: "r"(outs)  : "memory");
    asm volatile("vse32.v   v4,  (%0), v0.t" :: "r"(outc)  : "memory");

    // q==2: sin=-s0 (v4),  cos=-c0 (v20)
    asm volatile("vmseq.vi  v0, v28, 2");
    asm volatile("vse32.v   v4,  (%0), v0.t" :: "r"(outs)  : "memory");
    asm volatile("vse32.v   v20, (%0), v0.t" :: "r"(outc)  : "memory");

    // q==3: sin=-c0 (v20), cos=+s0 (v16)
    asm volatile("vmseq.vi  v0, v28, 3");
    asm volatile("vse32.v   v20, (%0), v0.t" :: "r"(outs)  : "memory");
    asm volatile("vse32.v   v16, (%0), v0.t" :: "r"(outc)  : "memory");
}

static void check_result(float *x,float *ref, int r){
    float diff = 0.0f;
    int err = 0;
    for (int i = 0; i < r; i++) {
        diff = fabsf(x[i] - ref[i]);
        if (diff > THRESHOLD) {
            err++;
            printf("Error at index %d:\t expected %f\t real %f\t error %f\n",
                   i, ref[i], x[i], diff);
            if (err > 64) break;
        }
    }
    if (err) printf("TEST FAILED with %d errors!!\n", err);
    else     printf("TEST PASSED!!\n");
}

static void print_first8(const char* tag, const float* a){
    printf("%s", tag);
    for (int i=0;i<SHOW_LANES;i++) printf(" % .7e", a[i]);
    printf("\n");
}
static void print_first8_mask(const char* tag, const float* m){
    printf("%s", tag);
    for (int i=0;i<SHOW_LANES;i++) printf(" %d", (int)(m[i]>0.5f));
    printf("\n");
}
static void print_first8_i32(const char* tag, const int32_t* a){
    printf("%s", tag);
    for (int i=0;i<SHOW_LANES;i++) printf(" %d", a[i]);
    printf("\n");
}

int main() {

    volatile int errors = 0;

    const unsigned int num_cores = snrt_cluster_core_num();
    const unsigned int cid       = snrt_cluster_core_idx();
    snrt_cluster_hw_barrier();

    unsigned int timer = (unsigned int)-1;
    int B = B_Size;
    int C = C_Size;
    int T = T_Size;

    const int N = B * T * C;

    float* inp   = NULL;
    float* out_c = NULL;
    float* out_s = NULL;

    if (cid == 0) {
        inp   = (float*)snrt_l1alloc(N * sizeof(float));
        out_c = (float*)snrt_l1alloc(N * sizeof(float));
        out_s = (float*)snrt_l1alloc(N * sizeof(float));
        snrt_dma_start_1d(inp, data1_dram, N * sizeof(float));
        snrt_dma_wait_all();

        memset(out_c, 0, N * sizeof(float));
        memset(out_s, 0, N * sizeof(float));

        // ---- allocate debug taps for 1 chunk ----
        dbg_m0 = (float*)  snrt_l1alloc(VLMAX * sizeof(float));
        dbg_m1 = (float*)  snrt_l1alloc(VLMAX * sizeof(float));
        dbg_m2 = (float*)  snrt_l1alloc(VLMAX * sizeof(float));
        dbg_m3 = (float*)  snrt_l1alloc(VLMAX * sizeof(float));
        dbg_s_pos = (float*)  snrt_l1alloc(VLMAX * sizeof(float));
        dbg_c_pos = (float*)  snrt_l1alloc(VLMAX * sizeof(float));
        dbg_s_neg = (float*)  snrt_l1alloc(VLMAX * sizeof(float));
        dbg_c_neg = (float*)  snrt_l1alloc(VLMAX * sizeof(float));
        dbg_k = (int32_t*) snrt_l1alloc(VLMAX * sizeof(int32_t));
        dbg_q = (int32_t*) snrt_l1alloc(VLMAX * sizeof(int32_t));
    }
    snrt_cluster_hw_barrier();

    if (cid == 0) {
        start_kernel();

        int full_chunks = N / VLMAX;
        int rem         = N % VLMAX;

        // Only instrument the very first chunk
        for (int i = 0; i < full_chunks; i++) {
            const float* in_ptr = inp   + i * VLMAX;
            float*       so_ptr = out_s + i * VLMAX;
            float*       co_ptr = out_c + i * VLMAX;

            g_dbg_enable = (i == 0) ? 1 : 0;

            fast_sincos_poly_f32_m4_chunk(in_ptr, so_ptr, co_ptr, VLMAX);

            if (i == 0) {
                // Dump first 8
                printf("---- MASK & SOURCES DUMP (first %d lanes) ----\n", SHOW_LANES);
                print_first8_i32("k: ", dbg_k);
                print_first8_i32("q: ", dbg_q);
                print_first8_mask("mask q==0:", dbg_m0);
                print_first8_mask("mask q==1:", dbg_m1);
                print_first8_mask("mask q==2:", dbg_m2);
                print_first8_mask("mask q==3:", dbg_m3);
                print_first8(" +s0:", dbg_s_pos);
                print_first8(" +c0:", dbg_c_pos);
                print_first8(" -s0:", dbg_s_neg);
                print_first8(" -c0:", dbg_c_neg);
                print_first8(" outs:", so_ptr);
                print_first8(" outc:", co_ptr);
                printf("---------------------------------------------\n");
            }
        }

        // Remainder (no extra prints)
        if (rem) {
            const float* in_ptr = inp   + full_chunks * VLMAX;
            float*       so_ptr = out_s + full_chunks * VLMAX;
            float*       co_ptr = out_c + full_chunks * VLMAX;
            g_dbg_enable = (full_chunks == 0) ? 1 : 0; // if only a tail exists
            fast_sincos_poly_f32_m4_chunk(in_ptr, so_ptr, co_ptr, rem);

            if (full_chunks == 0) {
                printf("---- MASK & SOURCES DUMP (tail, first %d lanes) ----\n", SHOW_LANES);
                print_first8_i32("k: ", dbg_k);
                print_first8_i32("q: ", dbg_q);
                print_first8_mask("mask q==0:", dbg_m0);
                print_first8_mask("mask q==1:", dbg_m1);
                print_first8_mask("mask q==2:", dbg_m2);
                print_first8_mask("mask q==3:", dbg_m3);
                print_first8(" +s0:", dbg_s_pos);
                print_first8(" +c0:", dbg_c_pos);
                print_first8(" -s0:", dbg_s_neg);
                print_first8(" -c0:", dbg_c_neg);
                print_first8(" outs:", so_ptr);
                print_first8(" outc:", co_ptr);
                printf("---------------------------------------------\n");
            }
        }

        stop_kernel();

        printf("CHECK RESULTS (sin)\n");
        check_result(out_s, outS, N);
        printf("CHECK RESULTS (cos)\n");
        check_result(out_c, outC, N);

        timer = benchmark_get_cycle() - timer;
        printf("The execution took %u cycles.\n", timer);
    }

    snrt_cluster_hw_barrier();
    return errors;
}
