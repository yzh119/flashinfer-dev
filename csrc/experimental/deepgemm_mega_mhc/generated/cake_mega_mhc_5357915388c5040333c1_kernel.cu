/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
struct CakeFastDivmod { int32_t divisor; uint32_t multiplier; uint32_t shift_right; };
static_assert(sizeof(CakeFastDivmod) == 12, "CakeFastDivmod CUDA ABI must be 12 bytes");
static_assert(alignof(CakeFastDivmod) == 4, "CakeFastDivmod CUDA ABI must be 4-byte aligned");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 256
#define TMEM_A_TMEM_OFFSET 0
#define TMEM_ACC_SMALL_OFFSET 128
#define TMEM_ACC_BIG_OFFSET 176
#define NUM_MAIN_STAGES 1
#define SMEM_RESIDUAL_OFF 0
#define SMEM_RESIDUAL_STAGE_BYTES 32768
#define SMEM_RESIDUAL_STRIDE 32768
#define SMEM_X_OFF 98304
#define SMEM_X_STAGE_BYTES 8192
#define SMEM_X_STRIDE 8192
#define SMEM_RESIDUAL_PAIRS_OFF 0
#define SMEM_RESIDUAL_PAIRS_STAGE_BYTES 98304
#define SMEM_RESIDUAL_PAIRS_STRIDE 98304
#define SMEM_X_PAIRS_OFF 98304
#define SMEM_X_PAIRS_STAGE_BYTES 24576
#define SMEM_X_PAIRS_STRIDE 24576
#define SMEM_FN_OFF 122880
#define SMEM_FN_STAGE_BYTES 36864
#define SMEM_FN_STRIDE 36864
#define SMEM_FN_PIECES_OFF 122880
#define SMEM_FN_PIECES_STAGE_BYTES 3072
#define SMEM_FN_PIECES_STRIDE 3072
#define SMEM_POST_OFF 196608
#define SMEM_POST_STAGE_BYTES 1024
#define SMEM_POST_STRIDE 1024
#define SMEM_COMB_OFF 198656
#define SMEM_COMB_STAGE_BYTES 4096
#define SMEM_COMB_STRIDE 4096
#define SMEM_HC_SUMS_OFF 206848
#define SMEM_HC_SUMS_STAGE_BYTES 512
#define SMEM_HC_SUMS_STRIDE 512
#define SMEM_TAIL_SCRATCH_OFF 207360
#define SMEM_TAIL_SCRATCH_STAGE_BYTES 256
#define SMEM_TAIL_SCRATCH_STRIDE 256
#define SMEM_TAIL_CONTROL_OFF 207360
#define SMEM_TAIL_CONTROL_STAGE_BYTES 256
#define SMEM_TAIL_CONTROL_STRIDE 256
#define SMEM_QUEUE_OFF 207616
#define SMEM_QUEUE_STAGE_BYTES 4
#define SMEM_QUEUE_STRIDE 4
#define SMEM_TOTAL 208896
#define THREADS 768
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}



// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}




__device__ __forceinline__ void mma_ts_step(
    int taddr_out, int taddr_a, int b_lo, uint32_t b_dhi,
    uint32_t i_desc, int enable_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 dhi;\n\t"
        ".reg .b64 db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %5, 0;\n\t"
        "mov.b32 dhi, %3;\n\t"
        "mov.b64 db, {%2, dhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], db, %4, p;\n\t"
        "}\n"
        :: "r"(taddr_out), "r"(taddr_a), "r"(b_lo), "r"(b_dhi),
           "r"(i_desc), "r"(enable_d));
}


__device__ __forceinline__ void elect_commit(int mbar_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr));
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}



extern "C" __device__ float __nv_expf(float);



// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}




__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}





__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(768, LAUNCH_MIN_BLOCKS) void
kernel_cake_mega_mhc_5357915388c5040333c1(const __grid_constant__ CUtensorMap residual_map, const __grid_constant__ CUtensorMap x_map, const __grid_constant__ CUtensorMap fnp_map, const __grid_constant__ CUtensorMap post_map, const __grid_constant__ CUtensorMap comb_map, const __grid_constant__ CUtensorMap new_residual_map, float* __restrict__ mix_scales, float* __restrict__ mix_bases, float* __restrict__ new_post_mix, float* __restrict__ new_comb_res_mix, __nv_bfloat16* __restrict__ rmsnorm_weight, __nv_bfloat16* __restrict__ new_residual, __nv_bfloat16* __restrict__ y_bf16, float* __restrict__ scratch, unsigned long long* __restrict__ split_barriers, unsigned long long* __restrict__ launch_epochs, unsigned int num_tokens, unsigned int num_splits, CakeFastDivmod split_div, unsigned int num_ctas, float hc_norm_eps, float hc_pre_eps, float hc_post_scale, float sinkhorn_eps, unsigned int num_sinkhorn_iters, float rmsnorm_eps, float rmsnorm_scale)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 207872;
    #define full_io_addr (mbar_base + 0)
    #define empty_io_addr (mbar_base + 24)
    #define store_ready_addr (mbar_base + 48)
    #define full_fn_addr (mbar_base + 72)
    #define empty_fn_addr (mbar_base + 88)
    #define full_a_addr (mbar_base + 104)
    #define empty_a_addr (mbar_base + 136)
    #define big_full_addr (mbar_base + 168)
    #define big_empty_addr (mbar_base + 184)
    #define small_full_addr (mbar_base + 200)
    #define small_empty_addr (mbar_base + 216)
    #define full_coeff_addr (mbar_base + 232)
    #define empty_coeff_addr (mbar_base + 248)
    #define full_stats_addr (mbar_base + 264)
    #define empty_stats_addr (mbar_base + 272)
    #define mix_arrival_addr (mbar_base + 280)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* residual = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int residual_addr = smem + 0;
    __nv_bfloat16* x = reinterpret_cast<__nv_bfloat16*>(smem_raw + 98304);
    const int x_addr = smem + 98304;
    unsigned int* residual_pairs = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int residual_pairs_addr = smem + 0;
    unsigned int* x_pairs = reinterpret_cast<unsigned int*>(smem_raw + 98304);
    const int x_pairs_addr = smem + 98304;
    __nv_bfloat16* fn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 122880);
    const int fn_addr = smem + 122880;
    __nv_bfloat16* fn_pieces = reinterpret_cast<__nv_bfloat16*>(smem_raw + 122880);
    const int fn_pieces_addr = smem + 122880;
    float* post = reinterpret_cast<float*>(smem_raw + 196608);
    const int post_addr = smem + 196608;
    float* comb = reinterpret_cast<float*>(smem_raw + 198656);
    const int comb_addr = smem + 198656;
    float* hc_sums = reinterpret_cast<float*>(smem_raw + 206848);
    const int hc_sums_addr = smem + 206848;
    float* tail_scratch = reinterpret_cast<float*>(smem_raw + 207360);
    const int tail_scratch_addr = smem + 207360;
    unsigned int* tail_control = reinterpret_cast<unsigned int*>(smem_raw + 207360);
    const int tail_control_addr = smem + 207360;
    unsigned int* queue = reinterpret_cast<unsigned int*>(smem_raw + 207616);
    const int queue_addr = smem + 207616;
    unsigned int lane_in_warp = threadIdx.x % 32;
    uint32_t _fast_div_q_0 = (split_div.divisor != 1) ? (__umulhi((uint32_t)(64), split_div.multiplier) >> split_div.shift_right) : (uint32_t)(64);
    uint32_t _fast_div_r_0 = (uint32_t)(64) - _fast_div_q_0 * (uint32_t)(split_div.divisor);
    if (warp == 0) {
        if (elect_sync()) {
            queue[0] = 0;
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&residual_map))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&x_map))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&fnp_map))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&post_map))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&comb_map))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&new_residual_map))) : "memory");
        }
    }

    // Mbarrier init (16 pipeline groups, 0 ordered-sequence groups, 36 barriers)
    // Mbarriers at smem_raw[207872..208160)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full_io: 3 barriers, init_count=1
            mbarrier_init(smem + 207872, 1);
            mbarrier_init(smem + 207880, 1);
            mbarrier_init(smem + 207888, 1);
            // empty_io: 3 barriers, init_count=1
            mbarrier_init(smem + 207896, 1);
            mbarrier_init(smem + 207904, 1);
            mbarrier_init(smem + 207912, 1);
            // store_ready: 3 barriers, init_count=8
            mbarrier_init(smem + 207920, 8);
            mbarrier_init(smem + 207928, 8);
            mbarrier_init(smem + 207936, 8);
            // full_coeff: 2 barriers, init_count=1
            mbarrier_init(smem + 208104, 1);
            mbarrier_init(smem + 208112, 1);
            // empty_coeff: 2 barriers, init_count=8
            mbarrier_init(smem + 208120, 8);
            mbarrier_init(smem + 208128, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full_fn: 2 barriers, init_count=1
            mbarrier_init(smem + 207944, 1);
            mbarrier_init(smem + 207952, 1);
            // empty_fn: 2 barriers, init_count=1
            mbarrier_init(smem + 207960, 1);
            mbarrier_init(smem + 207968, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 2) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full_a: 4 barriers, init_count=128
            mbarrier_init(smem + 207976, 128);
            mbarrier_init(smem + 207984, 128);
            mbarrier_init(smem + 207992, 128);
            mbarrier_init(smem + 208000, 128);
            // empty_a: 4 barriers, init_count=1
            mbarrier_init(smem + 208008, 1);
            mbarrier_init(smem + 208016, 1);
            mbarrier_init(smem + 208024, 1);
            mbarrier_init(smem + 208032, 1);
            // big_full: 2 barriers, init_count=1
            mbarrier_init(smem + 208040, 1);
            mbarrier_init(smem + 208048, 1);
            // big_empty: 2 barriers, init_count=4
            mbarrier_init(smem + 208056, 4);
            mbarrier_init(smem + 208064, 4);
            // small_full: 2 barriers, init_count=1
            mbarrier_init(smem + 208072, 1);
            mbarrier_init(smem + 208080, 1);
            // small_empty: 2 barriers, init_count=4
            mbarrier_init(smem + 208088, 4);
            mbarrier_init(smem + 208096, 4);
            // full_stats: 1 barriers, init_count=8
            mbarrier_init(smem + 208136, 8);
            // empty_stats: 1 barriers, init_count=4
            mbarrier_init(smem + 208144, 4);
            // mix_arrival: 1 barriers, init_count=5
            mbarrier_init(smem + 208152, 5);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 224 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 208160);
    if (warp == 3) {
        int _tmem_hold = smem + 208160;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_a_tmem = taddr;
    const int tmem_acc_small = taddr + 128;
    const int tmem_acc_big = taddr + 176;

    // ---- Role: control ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // control_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
            if (warp < 2) {
                unsigned int seq = 0;
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int task = (unsigned int)bid; task < (num_tokens + 63) / 64 * num_splits; task += num_ctas) {
                        uint32_t _fast_div_q_1 = (split_div.divisor != 1) ? (__umulhi((uint32_t)(task), split_div.multiplier) >> split_div.shift_right) : (uint32_t)(task);
                        uint32_t _fast_div_r_1 = (uint32_t)(task) - _fast_div_q_1 * (uint32_t)(split_div.divisor);
                        #pragma unroll 1
                        for (unsigned int kb = 0; kb < _fast_div_q_0 + ((_fast_div_r_1 < _fast_div_r_0) ? 1 : 0); kb++) {
                            unsigned int hidden = (_fast_div_r_1 * _fast_div_q_0 + ((_fast_div_r_1 < _fast_div_r_0) ? _fast_div_r_1 : _fast_div_r_0)) * 64 + kb * 64;
                            if (warp == 0) {
                                unsigned int stage = seq % 3;
                                unsigned int io_empty_phase = seq / 3 % 2 ^ 1;
                                mbarrier_wait(empty_io_addr + (stage) * 8, io_empty_phase);
                                tma_3d_gmem2smem(residual_addr + stage * 32768, (&residual_map), hidden, _fast_div_q_1 * 64, 0, full_io_addr + (stage) * 8);
                                tma_2d_gmem2smem(x_addr + stage * 8192, (&x_map), hidden, _fast_div_q_1 * 64, full_io_addr + (stage) * 8);
                                mbarrier_arrive_expect_tx(full_io_addr + (stage) * 8, 40960);
                            } else {
                                unsigned int stage_1 = seq % 2;
                                unsigned int fn_empty_phase = seq / 2 % 2 ^ 1;
                                mbarrier_wait(empty_fn_addr + (stage_1) * 8, fn_empty_phase);
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(fn_addr + stage_1 * 36864), "l"((&fnp_map)), "r"(hidden), "r"(0), "r"(0),
                                       "r"(full_fn_addr + (stage_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(fn_addr + stage_1 * 36864 + 9216), "l"((&fnp_map)), "r"(hidden + 4096), "r"(0), "r"(0),
                                       "r"(full_fn_addr + (stage_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(fn_addr + stage_1 * 36864 + 18432), "l"((&fnp_map)), "r"(hidden + 8192), "r"(0), "r"(0),
                                       "r"(full_fn_addr + (stage_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(fn_addr + stage_1 * 36864 + 27648), "l"((&fnp_map)), "r"(hidden + 12288), "r"(0), "r"(0),
                                       "r"(full_fn_addr + (stage_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                mbarrier_arrive_expect_tx(full_fn_addr + (stage_1) * 8, 36864);
                            }
                            seq += 1;
                        }
                    }
                }
                __syncwarp();
            } else if (warp == 2) {
                unsigned int fseq = 0;
                unsigned int aseq = 0;
                unsigned int cseq = 0;
                unsigned int useq = 0;
                #pragma unroll 1
                for (unsigned int task_1 = (unsigned int)bid; task_1 < (num_tokens + 63) / 64 * num_splits; task_1 += num_ctas) {
                    uint32_t _fast_div_q_2 = (split_div.divisor != 1) ? (__umulhi((uint32_t)(task_1), split_div.multiplier) >> split_div.shift_right) : (uint32_t)(task_1);
                    uint32_t _fast_div_r_2 = (uint32_t)(task_1) - _fast_div_q_2 * (uint32_t)(split_div.divisor);
                    unsigned int ustage = useq % 2;
                    unsigned int small_empty_phase = useq / 2 % 2 ^ 1;
                    mbarrier_wait(small_empty_addr + (ustage) * 8, small_empty_phase);
                    #pragma unroll 1
                    for (unsigned int kb_1 = 0; kb_1 < _fast_div_q_0 + ((_fast_div_r_2 < _fast_div_r_0) ? 1 : 0); kb_1++) {
                        unsigned int fstage = fseq % 2;
                        unsigned int fn_full_phase = fseq / 2 % 2;
                        mbarrier_wait(full_fn_addr + (fstage) * 8, fn_full_phase);
                        unsigned int cstage = cseq % 2;
                        unsigned int big_empty_phase = cseq / 2 % 2 ^ 1;
                        mbarrier_wait(big_empty_addr + (cstage) * 8, big_empty_phase);
                        unsigned int astage = aseq % 4;
                        unsigned int a_full_phase = aseq / 4 % 2;
                        mbarrier_wait(full_a_addr + (astage) * 8, a_full_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_0 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage * 32, _smem_desc_lo_0, 0x40004040, 67503248, ((kb_1 == 0) ? 0 : 1));
                        int _smem_desc_lo_1 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage * 32, _smem_desc_lo_1, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_2 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 24)), (unsigned int)tmem_a_tmem + astage * 32, _smem_desc_lo_2, 0x40004040, 67503248, 0);
                        int _smem_desc_lo_3 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage * 32 + 8), _smem_desc_lo_3, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_4 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage * 32 + 8), _smem_desc_lo_4, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_5 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage * 32 + 8), _smem_desc_lo_5, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_6 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage * 32 + 16), _smem_desc_lo_6, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_7 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage * 32 + 16), _smem_desc_lo_7, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_8 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage * 32 + 16), _smem_desc_lo_8, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_9 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage * 32 + 24), _smem_desc_lo_9, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_10 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage * 32 + 24), _smem_desc_lo_10, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_11 = make_warp_uniform((((fn_pieces_addr) >> 4) & 0x7FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage * 32 + 24), _smem_desc_lo_11, 0x40004040, 67503248, 1);
                        elect_commit(empty_a_addr + (astage) * 8);
                        aseq += 1;
                        unsigned int astage_0 = aseq % 4;
                        unsigned int a_full_phase_1 = aseq / 4 % 2;
                        mbarrier_wait(full_a_addr + (astage_0) * 8, a_full_phase_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_12 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_0 * 32, _smem_desc_lo_12, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_13 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_0 * 32, _smem_desc_lo_13, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_14 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 24)), (unsigned int)tmem_a_tmem + astage_0 * 32, _smem_desc_lo_14, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_15 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 8), _smem_desc_lo_15, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_16 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 8), _smem_desc_lo_16, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_17 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 8), _smem_desc_lo_17, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_18 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 16), _smem_desc_lo_18, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_19 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 16), _smem_desc_lo_19, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_20 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 16), _smem_desc_lo_20, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_21 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 24), _smem_desc_lo_21, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_22 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 24), _smem_desc_lo_22, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_23 = make_warp_uniform((((fn_pieces_addr + 32) >> 4) & 0x7FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 24), _smem_desc_lo_23, 0x40004040, 67503248, 1);
                        elect_commit(empty_a_addr + (astage_0) * 8);
                        aseq += 1;
                        elect_commit(big_full_addr + (cstage) * 8);
                        cseq += 1;
                        unsigned int cstage_2 = cseq % 2;
                        unsigned int big_empty_phase_3 = cseq / 2 % 2 ^ 1;
                        mbarrier_wait(big_empty_addr + (cstage_2) * 8, big_empty_phase_3);
                        unsigned int astage_4 = aseq % 4;
                        unsigned int a_full_phase_5 = aseq / 4 % 2;
                        mbarrier_wait(full_a_addr + (astage_4) * 8, a_full_phase_5);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_24 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_4 * 32, _smem_desc_lo_24, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_25 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_4 * 32, _smem_desc_lo_25, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_26 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2 * 24)), (unsigned int)tmem_a_tmem + astage_4 * 32, _smem_desc_lo_26, 0x40004040, 67503248, 0);
                        int _smem_desc_lo_27 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_4 * 32 + 8), _smem_desc_lo_27, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_28 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_4 * 32 + 8), _smem_desc_lo_28, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_29 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2 * 24)), (unsigned int)tmem_a_tmem + (astage_4 * 32 + 8), _smem_desc_lo_29, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_30 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_4 * 32 + 16), _smem_desc_lo_30, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_31 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_4 * 32 + 16), _smem_desc_lo_31, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_32 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2 * 24)), (unsigned int)tmem_a_tmem + (astage_4 * 32 + 16), _smem_desc_lo_32, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_33 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_4 * 32 + 24), _smem_desc_lo_33, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_34 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_4 * 32 + 24), _smem_desc_lo_34, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_35 = make_warp_uniform((((fn_pieces_addr + 64) >> 4) & 0x7FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2 * 24)), (unsigned int)tmem_a_tmem + (astage_4 * 32 + 24), _smem_desc_lo_35, 0x40004040, 67503248, 1);
                        elect_commit(empty_a_addr + (astage_4) * 8);
                        aseq += 1;
                        unsigned int astage_6 = aseq % 4;
                        unsigned int a_full_phase_7 = aseq / 4 % 2;
                        mbarrier_wait(full_a_addr + (astage_6) * 8, a_full_phase_7);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_36 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_6 * 32, _smem_desc_lo_36, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_37 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + astage_6 * 32, _smem_desc_lo_37, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_38 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2 * 24)), (unsigned int)tmem_a_tmem + astage_6 * 32, _smem_desc_lo_38, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_39 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_6 * 32 + 8), _smem_desc_lo_39, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_40 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 3 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_6 * 32 + 8), _smem_desc_lo_40, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_41 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 3) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2 * 24)), (unsigned int)tmem_a_tmem + (astage_6 * 32 + 8), _smem_desc_lo_41, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_42 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_6 * 32 + 16), _smem_desc_lo_42, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_43 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 6 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_6 * 32 + 16), _smem_desc_lo_43, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_44 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 6) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2 * 24)), (unsigned int)tmem_a_tmem + (astage_6 * 32 + 16), _smem_desc_lo_44, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_45 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 2) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_6 * 32 + 24), _smem_desc_lo_45, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_46 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 9 + 1) * 192);
                        mma_ts_step((tmem_acc_small + (ustage * 24)), (unsigned int)tmem_a_tmem + (astage_6 * 32 + 24), _smem_desc_lo_46, 0x40004040, 67503248, 1);
                        int _smem_desc_lo_47 = make_warp_uniform((((fn_pieces_addr + 96) >> 4) & 0x7FFF) + (fstage * 12 + 9) * 192);
                        mma_ts_step((tmem_acc_big + (cstage_2 * 24)), (unsigned int)tmem_a_tmem + (astage_6 * 32 + 24), _smem_desc_lo_47, 0x40004040, 67503248, 1);
                        elect_commit(empty_a_addr + (astage_6) * 8);
                        aseq += 1;
                        elect_commit(big_full_addr + (cstage_2) * 8);
                        cseq += 1;
                        elect_commit(empty_fn_addr + (fstage) * 8);
                        fseq += 1;
                    }
                    elect_commit(small_full_addr + (ustage) * 8);
                    useq += 1;
                }
            } else {
                unsigned int io_seq = 0;
                unsigned int coeff_seq = 0;
                if ((unsigned int)bid < (num_tokens + 63) / 64 * num_splits) {
                    mbarrier_wait(empty_coeff_addr, 1);
                    if (elect_sync()) {
                        uint32_t _fast_div_q_3 = (split_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)bid), split_div.multiplier) >> split_div.shift_right) : (uint32_t)((unsigned int)bid);
                        uint32_t _fast_div_r_3 = (uint32_t)((unsigned int)bid) - _fast_div_q_3 * (uint32_t)(split_div.divisor);
                        tma_2d_gmem2smem(post_addr, (&post_map), 0, _fast_div_q_3 * 64, full_coeff_addr);
                        tma_2d_gmem2smem(comb_addr, (&comb_map), 0, _fast_div_q_3 * 64, full_coeff_addr);
                        mbarrier_arrive_expect_tx(full_coeff_addr, 5120);
                    }
                    __syncwarp();
                    coeff_seq += 1;
                }
                #pragma unroll 1
                for (unsigned int task_2 = (unsigned int)bid; task_2 < (num_tokens + 63) / 64 * num_splits; task_2 += num_ctas) {
                    uint32_t _fast_div_q_4 = (split_div.divisor != 1) ? (__umulhi((uint32_t)(task_2), split_div.multiplier) >> split_div.shift_right) : (uint32_t)(task_2);
                    uint32_t _fast_div_r_4 = (uint32_t)(task_2) - _fast_div_q_4 * (uint32_t)(split_div.divisor);
                    unsigned int blocks = _fast_div_q_0 + ((_fast_div_r_4 < _fast_div_r_0) ? 1 : 0);
                    #pragma unroll 1
                    for (unsigned int kb_2 = 0; kb_2 < blocks; kb_2++) {
                        unsigned int stage_2 = io_seq % 3;
                        unsigned int store_phase = io_seq / 3 % 2;
                        mbarrier_wait(store_ready_addr + (stage_2) * 8, store_phase);
                        unsigned int prefetch = kb_2 + 1 == blocks && task_2 + num_ctas < (num_tokens + 63) / 64 * num_splits;
                        if (prefetch != 0) {
                            unsigned int coeff_stage = coeff_seq % 2;
                            unsigned int coeff_empty_phase = coeff_seq / 2 % 2 ^ 1;
                            mbarrier_wait(empty_coeff_addr + (coeff_stage) * 8, coeff_empty_phase);
                        }
                        if (elect_sync()) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3}], [%4], %5;"
                                :: "l"((&new_residual_map)), "r"((_fast_div_r_4 * _fast_div_q_0 + ((_fast_div_r_4 < _fast_div_r_0) ? _fast_div_r_4 : _fast_div_r_0)) * 64 + kb_2 * 64), "r"(_fast_div_q_4 * 64), "r"(0), "r"(residual_addr + stage_2 * 32768), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                            if (prefetch != 0) {
                                uint32_t _fast_div_q_5 = (split_div.divisor != 1) ? (__umulhi((uint32_t)(task_2 + num_ctas), split_div.multiplier) >> split_div.shift_right) : (uint32_t)(task_2 + num_ctas);
                                uint32_t _fast_div_r_5 = (uint32_t)(task_2 + num_ctas) - _fast_div_q_5 * (uint32_t)(split_div.divisor);
                                tma_2d_gmem2smem(post_addr + coeff_seq % 2 * 1024, (&post_map), 0, _fast_div_q_5 * 64, full_coeff_addr + (coeff_seq % 2) * 8);
                                tma_2d_gmem2smem(comb_addr + coeff_seq % 2 * 4096, (&comb_map), 0, _fast_div_q_5 * 64, full_coeff_addr + (coeff_seq % 2) * 8);
                                mbarrier_arrive_expect_tx(full_coeff_addr + (coeff_seq % 2) * 8, 5120);
                            }
                            if (blocks > kb_2 + 1) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            } else {
                                asm volatile("cp.async.bulk.wait_group 0;");
                            }
                            mbarrier_arrive(empty_io_addr + (stage_2) * 8);
                        }
                        if (prefetch != 0) {
                            coeff_seq += 1;
                        }
                        io_seq += 1;
                    }
                    if (elect_sync()) {
                        mbarrier_arrive(mix_arrival_addr);
                    }
                }
            }
        }
    }
    // ---- Role: post0 ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // post0_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("setmaxnreg.inc.sync.aligned.u32 112;");
            unsigned int io_seq_1 = 0;
            unsigned int a_seq = 0;
            unsigned int coeff_seq_1 = 0;
            unsigned int first_row = warp % 4 * 16 + lane_in_warp / 4;
            unsigned int stats_phase = 1;
            #pragma unroll 1
            for (unsigned int task_3 = (unsigned int)bid; task_3 < (num_tokens + 63) / 64 * num_splits; task_3 += num_ctas) {
                uint32_t _fast_div_q_6 = (split_div.divisor != 1) ? (__umulhi((uint32_t)(task_3), split_div.multiplier) >> split_div.shift_right) : (uint32_t)(task_3);
                uint32_t _fast_div_r_6 = (uint32_t)(task_3) - _fast_div_q_6 * (uint32_t)(split_div.divisor);
                unsigned int coeff_stage_1 = coeff_seq_1 % 2;
                unsigned int coeff_phase = coeff_seq_1 / 2 % 2;
                mbarrier_wait(full_coeff_addr + (coeff_stage_1) * 8, coeff_phase);
                float post_coeff[8];
                float comb_coeff[32];
                #pragma unroll
                for (int ri = 0; ri < 2; ri++) {
                    unsigned int row = first_row + (unsigned int)(ri * 8);
                    #pragma unroll
                    for (int route = 0; route < 4; route++) {
                        post_coeff[ri * 4 + route] = post[coeff_stage_1 * 256 + row * 4 + (unsigned int)route];
                    }
                    #pragma unroll
                    for (int ci = 0; ci < 16; ci++) {
                        comb_coeff[ri * 16 + ci] = comb[coeff_stage_1 * 1024 + row * 16 + ((unsigned int)(ci / 4) ^ row / 2 & 3) * 4 + (unsigned int)(ci % 4)];
                    }
                }
                __syncwarp();
                if (elect_sync()) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(empty_coeff_addr + (coeff_stage_1) * 8);
                }
                coeff_seq_1 += 1;
                float hc_acc[2];
                float hc_comp[2];
                hc_acc[0] = 0.0f;
                hc_comp[0] = 0.0f;
                hc_acc[1] = 0.0f;
                hc_comp[1] = 0.0f;
                #pragma unroll 1
                for (unsigned int kb_3 = 0; kb_3 < _fast_div_q_0 + ((_fast_div_r_6 < _fast_div_r_0) ? 1 : 0); kb_3++) {
                    unsigned int io_stage = io_seq_1 % 3;
                    unsigned int io_phase = io_seq_1 / 3 % 2;
                    mbarrier_wait(full_io_addr + (io_stage) * 8, io_phase);
                    #pragma unroll
                    for (int atom = 0; atom < 4; atom += 2) {
                        unsigned int words[16];
                        unsigned int pair = (unsigned int)(atom * 8) + lane_in_warp % 4 * 2;
                        unsigned int row_1 = first_row;
                        unsigned int offset = row_1 * 32 + (pair / 4 ^ row_1 & 7) * 4 + pair % 4;
                        unsigned int x_word = x_pairs[io_stage * 2048 + offset];
                        float2 xv = make_float2(__uint_as_float(x_word << 16), __uint_as_float(x_word & 4294901760u));
                        float rv[8];
                        unsigned int word = residual_pairs[io_stage * 8192 + offset];
                        rv[0] = __uint_as_float(word << 16);
                        rv[1] = __uint_as_float(word & 4294901760u);
                        unsigned int word_0 = residual_pairs[io_stage * 8192 + 2048 + offset];
                        rv[2] = __uint_as_float(word_0 << 16);
                        rv[3] = __uint_as_float(word_0 & 4294901760u);
                        unsigned int word_1 = residual_pairs[io_stage * 8192 + 4096 + offset];
                        rv[4] = __uint_as_float(word_1 << 16);
                        rv[5] = __uint_as_float(word_1 & 4294901760u);
                        unsigned int word_2 = residual_pairs[io_stage * 8192 + 6144 + offset];
                        rv[6] = __uint_as_float(word_2 << 16);
                        rv[7] = __uint_as_float(word_2 & 4294901760u);
                        float sq[2];
                        sq[0] = 0.0f;
                        sq[1] = 0.0f;
                        float coeff = post_coeff[0];
                        float2 _f32x2_mul_a_0 = make_float2(coeff, coeff);
                        float2 value;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value) : "l"(*(const unsigned long long*)&_f32x2_mul_a_0), "l"(*(const unsigned long long*)&xv));
                        float cc = comb_coeff[0];
                        value = fma_f32x2_rn_noftz(make_float2(cc, cc), make_float2(rv[0], rv[1]), value);
                        float cc_3 = comb_coeff[4];
                        value = fma_f32x2_rn_noftz(make_float2(cc_3, cc_3), make_float2(rv[2], rv[3]), value);
                        float cc_4 = comb_coeff[8];
                        value = fma_f32x2_rn_noftz(make_float2(cc_4, cc_4), make_float2(rv[4], rv[5]), value);
                        float cc_5 = comb_coeff[12];
                        value = fma_f32x2_rn_noftz(make_float2(cc_5, cc_5), make_float2(rv[6], rv[7]), value);
                        unsigned int word_6 = __as_u32(__float22bfloat162_rn(make_float2(value.x, value.y)));
                        residual_pairs[io_stage * 8192 + offset] = word_6;
                        words[0] = word_6;
                        float2 rounded_pair = make_float2(__uint_as_float(word_6 << 16), __uint_as_float(word_6 & 4294901760u));
                        float2 squares = fma_f32x2_rn_noftz(rounded_pair, rounded_pair, make_float2(sq[0], sq[1]));
                        sq[0] = squares.x;
                        sq[1] = squares.y;
                        float coeff_7 = post_coeff[1];
                        float2 _f32x2_mul_a_1 = make_float2(coeff_7, coeff_7);
                        float2 value_8;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1), "l"(*(const unsigned long long*)&xv));
                        float cc_9 = comb_coeff[1];
                        value_8 = fma_f32x2_rn_noftz(make_float2(cc_9, cc_9), make_float2(rv[0], rv[1]), value_8);
                        float cc_10 = comb_coeff[5];
                        value_8 = fma_f32x2_rn_noftz(make_float2(cc_10, cc_10), make_float2(rv[2], rv[3]), value_8);
                        float cc_11 = comb_coeff[9];
                        value_8 = fma_f32x2_rn_noftz(make_float2(cc_11, cc_11), make_float2(rv[4], rv[5]), value_8);
                        float cc_12 = comb_coeff[13];
                        value_8 = fma_f32x2_rn_noftz(make_float2(cc_12, cc_12), make_float2(rv[6], rv[7]), value_8);
                        unsigned int word_13 = __as_u32(__float22bfloat162_rn(make_float2(value_8.x, value_8.y)));
                        residual_pairs[io_stage * 8192 + 2048 + offset] = word_13;
                        words[4] = word_13;
                        float2 rounded_pair_14 = make_float2(__uint_as_float(word_13 << 16), __uint_as_float(word_13 & 4294901760u));
                        float2 squares_15 = fma_f32x2_rn_noftz(rounded_pair_14, rounded_pair_14, make_float2(sq[0], sq[1]));
                        sq[0] = squares_15.x;
                        sq[1] = squares_15.y;
                        float coeff_16 = post_coeff[2];
                        float2 _f32x2_mul_a_2 = make_float2(coeff_16, coeff_16);
                        float2 value_17;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2), "l"(*(const unsigned long long*)&xv));
                        float cc_18 = comb_coeff[2];
                        value_17 = fma_f32x2_rn_noftz(make_float2(cc_18, cc_18), make_float2(rv[0], rv[1]), value_17);
                        float cc_19 = comb_coeff[6];
                        value_17 = fma_f32x2_rn_noftz(make_float2(cc_19, cc_19), make_float2(rv[2], rv[3]), value_17);
                        float cc_20 = comb_coeff[10];
                        value_17 = fma_f32x2_rn_noftz(make_float2(cc_20, cc_20), make_float2(rv[4], rv[5]), value_17);
                        float cc_21 = comb_coeff[14];
                        value_17 = fma_f32x2_rn_noftz(make_float2(cc_21, cc_21), make_float2(rv[6], rv[7]), value_17);
                        unsigned int word_22 = __as_u32(__float22bfloat162_rn(make_float2(value_17.x, value_17.y)));
                        residual_pairs[io_stage * 8192 + 4096 + offset] = word_22;
                        words[8] = word_22;
                        float2 rounded_pair_23 = make_float2(__uint_as_float(word_22 << 16), __uint_as_float(word_22 & 4294901760u));
                        float2 squares_24 = fma_f32x2_rn_noftz(rounded_pair_23, rounded_pair_23, make_float2(sq[0], sq[1]));
                        sq[0] = squares_24.x;
                        sq[1] = squares_24.y;
                        float coeff_25 = post_coeff[3];
                        float2 _f32x2_mul_a_3 = make_float2(coeff_25, coeff_25);
                        float2 value_26;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_3), "l"(*(const unsigned long long*)&xv));
                        float cc_27 = comb_coeff[3];
                        value_26 = fma_f32x2_rn_noftz(make_float2(cc_27, cc_27), make_float2(rv[0], rv[1]), value_26);
                        float cc_28 = comb_coeff[7];
                        value_26 = fma_f32x2_rn_noftz(make_float2(cc_28, cc_28), make_float2(rv[2], rv[3]), value_26);
                        float cc_29 = comb_coeff[11];
                        value_26 = fma_f32x2_rn_noftz(make_float2(cc_29, cc_29), make_float2(rv[4], rv[5]), value_26);
                        float cc_30 = comb_coeff[15];
                        value_26 = fma_f32x2_rn_noftz(make_float2(cc_30, cc_30), make_float2(rv[6], rv[7]), value_26);
                        unsigned int word_31 = __as_u32(__float22bfloat162_rn(make_float2(value_26.x, value_26.y)));
                        residual_pairs[io_stage * 8192 + 6144 + offset] = word_31;
                        words[12] = word_31;
                        float2 rounded_pair_32 = make_float2(__uint_as_float(word_31 << 16), __uint_as_float(word_31 & 4294901760u));
                        float2 squares_33 = fma_f32x2_rn_noftz(rounded_pair_32, rounded_pair_32, make_float2(sq[0], sq[1]));
                        sq[0] = squares_33.x;
                        sq[1] = squares_33.y;
                        float _add_4;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(sq[0]), "f"(sq[1]));
                        float _sub_0;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_0) : "f"(_add_4), "f"(hc_comp[0]));
                        float _add_0;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_0) : "f"(hc_acc[0]), "f"(_sub_0));
                        float _sub_6;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6) : "f"(_add_0), "f"(hc_acc[0]));
                        float _sub_5;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_5) : "f"(_sub_6), "f"(_sub_0));
                        hc_comp[0] = _sub_5;
                        hc_acc[0] = _add_0;
                        unsigned int row_34 = first_row + 8;
                        unsigned int offset_35 = row_34 * 32 + (pair / 4 ^ row_34 & 7) * 4 + pair % 4;
                        unsigned int x_word_36 = x_pairs[io_stage * 2048 + offset_35];
                        float2 xv_37 = make_float2(__uint_as_float(x_word_36 << 16), __uint_as_float(x_word_36 & 4294901760u));
                        float rv_38[8];
                        unsigned int word_39 = residual_pairs[io_stage * 8192 + offset_35];
                        rv_38[0] = __uint_as_float(word_39 << 16);
                        rv_38[1] = __uint_as_float(word_39 & 4294901760u);
                        unsigned int word_40 = residual_pairs[io_stage * 8192 + 2048 + offset_35];
                        rv_38[2] = __uint_as_float(word_40 << 16);
                        rv_38[3] = __uint_as_float(word_40 & 4294901760u);
                        unsigned int word_41 = residual_pairs[io_stage * 8192 + 4096 + offset_35];
                        rv_38[4] = __uint_as_float(word_41 << 16);
                        rv_38[5] = __uint_as_float(word_41 & 4294901760u);
                        unsigned int word_42 = residual_pairs[io_stage * 8192 + 6144 + offset_35];
                        rv_38[6] = __uint_as_float(word_42 << 16);
                        rv_38[7] = __uint_as_float(word_42 & 4294901760u);
                        float sq_43[2];
                        sq_43[0] = 0.0f;
                        sq_43[1] = 0.0f;
                        float coeff_44 = post_coeff[4];
                        float2 _f32x2_mul_a_7 = make_float2(coeff_44, coeff_44);
                        float2 value_45;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_45) : "l"(*(const unsigned long long*)&_f32x2_mul_a_7), "l"(*(const unsigned long long*)&xv_37));
                        float cc_46 = comb_coeff[16];
                        value_45 = fma_f32x2_rn_noftz(make_float2(cc_46, cc_46), make_float2(rv_38[0], rv_38[1]), value_45);
                        float cc_47 = comb_coeff[20];
                        value_45 = fma_f32x2_rn_noftz(make_float2(cc_47, cc_47), make_float2(rv_38[2], rv_38[3]), value_45);
                        float cc_48 = comb_coeff[24];
                        value_45 = fma_f32x2_rn_noftz(make_float2(cc_48, cc_48), make_float2(rv_38[4], rv_38[5]), value_45);
                        float cc_49 = comb_coeff[28];
                        value_45 = fma_f32x2_rn_noftz(make_float2(cc_49, cc_49), make_float2(rv_38[6], rv_38[7]), value_45);
                        unsigned int word_50 = __as_u32(__float22bfloat162_rn(make_float2(value_45.x, value_45.y)));
                        residual_pairs[io_stage * 8192 + offset_35] = word_50;
                        words[2] = word_50;
                        float2 rounded_pair_51 = make_float2(__uint_as_float(word_50 << 16), __uint_as_float(word_50 & 4294901760u));
                        float2 squares_52 = fma_f32x2_rn_noftz(rounded_pair_51, rounded_pair_51, make_float2(sq_43[0], sq_43[1]));
                        sq_43[0] = squares_52.x;
                        sq_43[1] = squares_52.y;
                        float coeff_53 = post_coeff[5];
                        float2 _f32x2_mul_a_8 = make_float2(coeff_53, coeff_53);
                        float2 value_54;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_54) : "l"(*(const unsigned long long*)&_f32x2_mul_a_8), "l"(*(const unsigned long long*)&xv_37));
                        float cc_55 = comb_coeff[17];
                        value_54 = fma_f32x2_rn_noftz(make_float2(cc_55, cc_55), make_float2(rv_38[0], rv_38[1]), value_54);
                        float cc_56 = comb_coeff[21];
                        value_54 = fma_f32x2_rn_noftz(make_float2(cc_56, cc_56), make_float2(rv_38[2], rv_38[3]), value_54);
                        float cc_57 = comb_coeff[25];
                        value_54 = fma_f32x2_rn_noftz(make_float2(cc_57, cc_57), make_float2(rv_38[4], rv_38[5]), value_54);
                        float cc_58 = comb_coeff[29];
                        value_54 = fma_f32x2_rn_noftz(make_float2(cc_58, cc_58), make_float2(rv_38[6], rv_38[7]), value_54);
                        unsigned int word_59 = __as_u32(__float22bfloat162_rn(make_float2(value_54.x, value_54.y)));
                        residual_pairs[io_stage * 8192 + 2048 + offset_35] = word_59;
                        words[6] = word_59;
                        float2 rounded_pair_60 = make_float2(__uint_as_float(word_59 << 16), __uint_as_float(word_59 & 4294901760u));
                        float2 squares_61 = fma_f32x2_rn_noftz(rounded_pair_60, rounded_pair_60, make_float2(sq_43[0], sq_43[1]));
                        sq_43[0] = squares_61.x;
                        sq_43[1] = squares_61.y;
                        float coeff_62 = post_coeff[6];
                        float2 _f32x2_mul_a_9 = make_float2(coeff_62, coeff_62);
                        float2 value_63;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_63) : "l"(*(const unsigned long long*)&_f32x2_mul_a_9), "l"(*(const unsigned long long*)&xv_37));
                        float cc_64 = comb_coeff[18];
                        value_63 = fma_f32x2_rn_noftz(make_float2(cc_64, cc_64), make_float2(rv_38[0], rv_38[1]), value_63);
                        float cc_65 = comb_coeff[22];
                        value_63 = fma_f32x2_rn_noftz(make_float2(cc_65, cc_65), make_float2(rv_38[2], rv_38[3]), value_63);
                        float cc_66 = comb_coeff[26];
                        value_63 = fma_f32x2_rn_noftz(make_float2(cc_66, cc_66), make_float2(rv_38[4], rv_38[5]), value_63);
                        float cc_67 = comb_coeff[30];
                        value_63 = fma_f32x2_rn_noftz(make_float2(cc_67, cc_67), make_float2(rv_38[6], rv_38[7]), value_63);
                        unsigned int word_68 = __as_u32(__float22bfloat162_rn(make_float2(value_63.x, value_63.y)));
                        residual_pairs[io_stage * 8192 + 4096 + offset_35] = word_68;
                        words[10] = word_68;
                        float2 rounded_pair_69 = make_float2(__uint_as_float(word_68 << 16), __uint_as_float(word_68 & 4294901760u));
                        float2 squares_70 = fma_f32x2_rn_noftz(rounded_pair_69, rounded_pair_69, make_float2(sq_43[0], sq_43[1]));
                        sq_43[0] = squares_70.x;
                        sq_43[1] = squares_70.y;
                        float coeff_71 = post_coeff[7];
                        float2 _f32x2_mul_a_10 = make_float2(coeff_71, coeff_71);
                        float2 value_72;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_72) : "l"(*(const unsigned long long*)&_f32x2_mul_a_10), "l"(*(const unsigned long long*)&xv_37));
                        float cc_73 = comb_coeff[19];
                        value_72 = fma_f32x2_rn_noftz(make_float2(cc_73, cc_73), make_float2(rv_38[0], rv_38[1]), value_72);
                        float cc_74 = comb_coeff[23];
                        value_72 = fma_f32x2_rn_noftz(make_float2(cc_74, cc_74), make_float2(rv_38[2], rv_38[3]), value_72);
                        float cc_75 = comb_coeff[27];
                        value_72 = fma_f32x2_rn_noftz(make_float2(cc_75, cc_75), make_float2(rv_38[4], rv_38[5]), value_72);
                        float cc_76 = comb_coeff[31];
                        value_72 = fma_f32x2_rn_noftz(make_float2(cc_76, cc_76), make_float2(rv_38[6], rv_38[7]), value_72);
                        unsigned int word_77 = __as_u32(__float22bfloat162_rn(make_float2(value_72.x, value_72.y)));
                        residual_pairs[io_stage * 8192 + 6144 + offset_35] = word_77;
                        words[14] = word_77;
                        float2 rounded_pair_78 = make_float2(__uint_as_float(word_77 << 16), __uint_as_float(word_77 & 4294901760u));
                        float2 squares_79 = fma_f32x2_rn_noftz(rounded_pair_78, rounded_pair_78, make_float2(sq_43[0], sq_43[1]));
                        sq_43[0] = squares_79.x;
                        sq_43[1] = squares_79.y;
                        float _add_11;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(sq_43[0]), "f"(sq_43[1]));
                        float _sub_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1) : "f"(_add_11), "f"(hc_comp[1]));
                        float _add_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(hc_acc[1]), "f"(_sub_1));
                        float _sub_13;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13) : "f"(_add_1), "f"(hc_acc[1]));
                        float _sub_12;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_12) : "f"(_sub_13), "f"(_sub_1));
                        hc_comp[1] = _sub_12;
                        hc_acc[1] = _add_1;
                        unsigned int pair_80 = (unsigned int)(atom * 8 + 1) + lane_in_warp % 4 * 2;
                        unsigned int row_81 = first_row;
                        unsigned int offset_82 = row_81 * 32 + (pair_80 / 4 ^ row_81 & 7) * 4 + pair_80 % 4;
                        unsigned int x_word_83 = x_pairs[io_stage * 2048 + offset_82];
                        float2 xv_84 = make_float2(__uint_as_float(x_word_83 << 16), __uint_as_float(x_word_83 & 4294901760u));
                        float rv_85[8];
                        unsigned int word_86 = residual_pairs[io_stage * 8192 + offset_82];
                        rv_85[0] = __uint_as_float(word_86 << 16);
                        rv_85[1] = __uint_as_float(word_86 & 4294901760u);
                        unsigned int word_87 = residual_pairs[io_stage * 8192 + 2048 + offset_82];
                        rv_85[2] = __uint_as_float(word_87 << 16);
                        rv_85[3] = __uint_as_float(word_87 & 4294901760u);
                        unsigned int word_88 = residual_pairs[io_stage * 8192 + 4096 + offset_82];
                        rv_85[4] = __uint_as_float(word_88 << 16);
                        rv_85[5] = __uint_as_float(word_88 & 4294901760u);
                        unsigned int word_89 = residual_pairs[io_stage * 8192 + 6144 + offset_82];
                        rv_85[6] = __uint_as_float(word_89 << 16);
                        rv_85[7] = __uint_as_float(word_89 & 4294901760u);
                        float sq_90[2];
                        sq_90[0] = 0.0f;
                        sq_90[1] = 0.0f;
                        float coeff_91 = post_coeff[0];
                        float2 _f32x2_mul_a_14 = make_float2(coeff_91, coeff_91);
                        float2 value_92;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_92) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&xv_84));
                        float cc_93 = comb_coeff[0];
                        value_92 = fma_f32x2_rn_noftz(make_float2(cc_93, cc_93), make_float2(rv_85[0], rv_85[1]), value_92);
                        float cc_94 = comb_coeff[4];
                        value_92 = fma_f32x2_rn_noftz(make_float2(cc_94, cc_94), make_float2(rv_85[2], rv_85[3]), value_92);
                        float cc_95 = comb_coeff[8];
                        value_92 = fma_f32x2_rn_noftz(make_float2(cc_95, cc_95), make_float2(rv_85[4], rv_85[5]), value_92);
                        float cc_96 = comb_coeff[12];
                        value_92 = fma_f32x2_rn_noftz(make_float2(cc_96, cc_96), make_float2(rv_85[6], rv_85[7]), value_92);
                        unsigned int word_97 = __as_u32(__float22bfloat162_rn(make_float2(value_92.x, value_92.y)));
                        residual_pairs[io_stage * 8192 + offset_82] = word_97;
                        words[1] = word_97;
                        float2 rounded_pair_98 = make_float2(__uint_as_float(word_97 << 16), __uint_as_float(word_97 & 4294901760u));
                        float2 squares_99 = fma_f32x2_rn_noftz(rounded_pair_98, rounded_pair_98, make_float2(sq_90[0], sq_90[1]));
                        sq_90[0] = squares_99.x;
                        sq_90[1] = squares_99.y;
                        float coeff_100 = post_coeff[1];
                        float2 _f32x2_mul_a_15 = make_float2(coeff_100, coeff_100);
                        float2 value_101;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_101) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&xv_84));
                        float cc_102 = comb_coeff[1];
                        value_101 = fma_f32x2_rn_noftz(make_float2(cc_102, cc_102), make_float2(rv_85[0], rv_85[1]), value_101);
                        float cc_103 = comb_coeff[5];
                        value_101 = fma_f32x2_rn_noftz(make_float2(cc_103, cc_103), make_float2(rv_85[2], rv_85[3]), value_101);
                        float cc_104 = comb_coeff[9];
                        value_101 = fma_f32x2_rn_noftz(make_float2(cc_104, cc_104), make_float2(rv_85[4], rv_85[5]), value_101);
                        float cc_105 = comb_coeff[13];
                        value_101 = fma_f32x2_rn_noftz(make_float2(cc_105, cc_105), make_float2(rv_85[6], rv_85[7]), value_101);
                        unsigned int word_106 = __as_u32(__float22bfloat162_rn(make_float2(value_101.x, value_101.y)));
                        residual_pairs[io_stage * 8192 + 2048 + offset_82] = word_106;
                        words[5] = word_106;
                        float2 rounded_pair_107 = make_float2(__uint_as_float(word_106 << 16), __uint_as_float(word_106 & 4294901760u));
                        float2 squares_108 = fma_f32x2_rn_noftz(rounded_pair_107, rounded_pair_107, make_float2(sq_90[0], sq_90[1]));
                        sq_90[0] = squares_108.x;
                        sq_90[1] = squares_108.y;
                        float coeff_109 = post_coeff[2];
                        float2 _f32x2_mul_a_16 = make_float2(coeff_109, coeff_109);
                        float2 value_110;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_110) : "l"(*(const unsigned long long*)&_f32x2_mul_a_16), "l"(*(const unsigned long long*)&xv_84));
                        float cc_111 = comb_coeff[2];
                        value_110 = fma_f32x2_rn_noftz(make_float2(cc_111, cc_111), make_float2(rv_85[0], rv_85[1]), value_110);
                        float cc_112 = comb_coeff[6];
                        value_110 = fma_f32x2_rn_noftz(make_float2(cc_112, cc_112), make_float2(rv_85[2], rv_85[3]), value_110);
                        float cc_113 = comb_coeff[10];
                        value_110 = fma_f32x2_rn_noftz(make_float2(cc_113, cc_113), make_float2(rv_85[4], rv_85[5]), value_110);
                        float cc_114 = comb_coeff[14];
                        value_110 = fma_f32x2_rn_noftz(make_float2(cc_114, cc_114), make_float2(rv_85[6], rv_85[7]), value_110);
                        unsigned int word_115 = __as_u32(__float22bfloat162_rn(make_float2(value_110.x, value_110.y)));
                        residual_pairs[io_stage * 8192 + 4096 + offset_82] = word_115;
                        words[9] = word_115;
                        float2 rounded_pair_116 = make_float2(__uint_as_float(word_115 << 16), __uint_as_float(word_115 & 4294901760u));
                        float2 squares_117 = fma_f32x2_rn_noftz(rounded_pair_116, rounded_pair_116, make_float2(sq_90[0], sq_90[1]));
                        sq_90[0] = squares_117.x;
                        sq_90[1] = squares_117.y;
                        float coeff_118 = post_coeff[3];
                        float2 _f32x2_mul_a_17 = make_float2(coeff_118, coeff_118);
                        float2 value_119;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_119) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&xv_84));
                        float cc_120 = comb_coeff[3];
                        value_119 = fma_f32x2_rn_noftz(make_float2(cc_120, cc_120), make_float2(rv_85[0], rv_85[1]), value_119);
                        float cc_121 = comb_coeff[7];
                        value_119 = fma_f32x2_rn_noftz(make_float2(cc_121, cc_121), make_float2(rv_85[2], rv_85[3]), value_119);
                        float cc_122 = comb_coeff[11];
                        value_119 = fma_f32x2_rn_noftz(make_float2(cc_122, cc_122), make_float2(rv_85[4], rv_85[5]), value_119);
                        float cc_123 = comb_coeff[15];
                        value_119 = fma_f32x2_rn_noftz(make_float2(cc_123, cc_123), make_float2(rv_85[6], rv_85[7]), value_119);
                        unsigned int word_124 = __as_u32(__float22bfloat162_rn(make_float2(value_119.x, value_119.y)));
                        residual_pairs[io_stage * 8192 + 6144 + offset_82] = word_124;
                        words[13] = word_124;
                        float2 rounded_pair_125 = make_float2(__uint_as_float(word_124 << 16), __uint_as_float(word_124 & 4294901760u));
                        float2 squares_126 = fma_f32x2_rn_noftz(rounded_pair_125, rounded_pair_125, make_float2(sq_90[0], sq_90[1]));
                        sq_90[0] = squares_126.x;
                        sq_90[1] = squares_126.y;
                        float _add_18;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(sq_90[0]), "f"(sq_90[1]));
                        float _sub_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_2) : "f"(_add_18), "f"(hc_comp[0]));
                        float _add_2;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_2) : "f"(hc_acc[0]), "f"(_sub_2));
                        float _sub_20;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20) : "f"(_add_2), "f"(hc_acc[0]));
                        float _sub_19;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19) : "f"(_sub_20), "f"(_sub_2));
                        hc_comp[0] = _sub_19;
                        hc_acc[0] = _add_2;
                        unsigned int row_127 = first_row + 8;
                        unsigned int offset_128 = row_127 * 32 + (pair_80 / 4 ^ row_127 & 7) * 4 + pair_80 % 4;
                        unsigned int x_word_129 = x_pairs[io_stage * 2048 + offset_128];
                        float2 xv_130 = make_float2(__uint_as_float(x_word_129 << 16), __uint_as_float(x_word_129 & 4294901760u));
                        float rv_131[8];
                        unsigned int word_132 = residual_pairs[io_stage * 8192 + offset_128];
                        rv_131[0] = __uint_as_float(word_132 << 16);
                        rv_131[1] = __uint_as_float(word_132 & 4294901760u);
                        unsigned int word_133 = residual_pairs[io_stage * 8192 + 2048 + offset_128];
                        rv_131[2] = __uint_as_float(word_133 << 16);
                        rv_131[3] = __uint_as_float(word_133 & 4294901760u);
                        unsigned int word_134 = residual_pairs[io_stage * 8192 + 4096 + offset_128];
                        rv_131[4] = __uint_as_float(word_134 << 16);
                        rv_131[5] = __uint_as_float(word_134 & 4294901760u);
                        unsigned int word_135 = residual_pairs[io_stage * 8192 + 6144 + offset_128];
                        rv_131[6] = __uint_as_float(word_135 << 16);
                        rv_131[7] = __uint_as_float(word_135 & 4294901760u);
                        float sq_136[2];
                        sq_136[0] = 0.0f;
                        sq_136[1] = 0.0f;
                        float coeff_137 = post_coeff[4];
                        float2 _f32x2_mul_a_21 = make_float2(coeff_137, coeff_137);
                        float2 value_138;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_138) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&xv_130));
                        float cc_139 = comb_coeff[16];
                        value_138 = fma_f32x2_rn_noftz(make_float2(cc_139, cc_139), make_float2(rv_131[0], rv_131[1]), value_138);
                        float cc_140 = comb_coeff[20];
                        value_138 = fma_f32x2_rn_noftz(make_float2(cc_140, cc_140), make_float2(rv_131[2], rv_131[3]), value_138);
                        float cc_141 = comb_coeff[24];
                        value_138 = fma_f32x2_rn_noftz(make_float2(cc_141, cc_141), make_float2(rv_131[4], rv_131[5]), value_138);
                        float cc_142 = comb_coeff[28];
                        value_138 = fma_f32x2_rn_noftz(make_float2(cc_142, cc_142), make_float2(rv_131[6], rv_131[7]), value_138);
                        unsigned int word_143 = __as_u32(__float22bfloat162_rn(make_float2(value_138.x, value_138.y)));
                        residual_pairs[io_stage * 8192 + offset_128] = word_143;
                        words[3] = word_143;
                        float2 rounded_pair_144 = make_float2(__uint_as_float(word_143 << 16), __uint_as_float(word_143 & 4294901760u));
                        float2 squares_145 = fma_f32x2_rn_noftz(rounded_pair_144, rounded_pair_144, make_float2(sq_136[0], sq_136[1]));
                        sq_136[0] = squares_145.x;
                        sq_136[1] = squares_145.y;
                        float coeff_146 = post_coeff[5];
                        float2 _f32x2_mul_a_22 = make_float2(coeff_146, coeff_146);
                        float2 value_147;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_147) : "l"(*(const unsigned long long*)&_f32x2_mul_a_22), "l"(*(const unsigned long long*)&xv_130));
                        float cc_148 = comb_coeff[17];
                        value_147 = fma_f32x2_rn_noftz(make_float2(cc_148, cc_148), make_float2(rv_131[0], rv_131[1]), value_147);
                        float cc_149 = comb_coeff[21];
                        value_147 = fma_f32x2_rn_noftz(make_float2(cc_149, cc_149), make_float2(rv_131[2], rv_131[3]), value_147);
                        float cc_150 = comb_coeff[25];
                        value_147 = fma_f32x2_rn_noftz(make_float2(cc_150, cc_150), make_float2(rv_131[4], rv_131[5]), value_147);
                        float cc_151 = comb_coeff[29];
                        value_147 = fma_f32x2_rn_noftz(make_float2(cc_151, cc_151), make_float2(rv_131[6], rv_131[7]), value_147);
                        unsigned int word_152 = __as_u32(__float22bfloat162_rn(make_float2(value_147.x, value_147.y)));
                        residual_pairs[io_stage * 8192 + 2048 + offset_128] = word_152;
                        words[7] = word_152;
                        float2 rounded_pair_153 = make_float2(__uint_as_float(word_152 << 16), __uint_as_float(word_152 & 4294901760u));
                        float2 squares_154 = fma_f32x2_rn_noftz(rounded_pair_153, rounded_pair_153, make_float2(sq_136[0], sq_136[1]));
                        sq_136[0] = squares_154.x;
                        sq_136[1] = squares_154.y;
                        float coeff_155 = post_coeff[6];
                        float2 _f32x2_mul_a_23 = make_float2(coeff_155, coeff_155);
                        float2 value_156;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_156) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&xv_130));
                        float cc_157 = comb_coeff[18];
                        value_156 = fma_f32x2_rn_noftz(make_float2(cc_157, cc_157), make_float2(rv_131[0], rv_131[1]), value_156);
                        float cc_158 = comb_coeff[22];
                        value_156 = fma_f32x2_rn_noftz(make_float2(cc_158, cc_158), make_float2(rv_131[2], rv_131[3]), value_156);
                        float cc_159 = comb_coeff[26];
                        value_156 = fma_f32x2_rn_noftz(make_float2(cc_159, cc_159), make_float2(rv_131[4], rv_131[5]), value_156);
                        float cc_160 = comb_coeff[30];
                        value_156 = fma_f32x2_rn_noftz(make_float2(cc_160, cc_160), make_float2(rv_131[6], rv_131[7]), value_156);
                        unsigned int word_161 = __as_u32(__float22bfloat162_rn(make_float2(value_156.x, value_156.y)));
                        residual_pairs[io_stage * 8192 + 4096 + offset_128] = word_161;
                        words[11] = word_161;
                        float2 rounded_pair_162 = make_float2(__uint_as_float(word_161 << 16), __uint_as_float(word_161 & 4294901760u));
                        float2 squares_163 = fma_f32x2_rn_noftz(rounded_pair_162, rounded_pair_162, make_float2(sq_136[0], sq_136[1]));
                        sq_136[0] = squares_163.x;
                        sq_136[1] = squares_163.y;
                        float coeff_164 = post_coeff[7];
                        float2 _f32x2_mul_a_24 = make_float2(coeff_164, coeff_164);
                        float2 value_165;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_165) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&xv_130));
                        float cc_166 = comb_coeff[19];
                        value_165 = fma_f32x2_rn_noftz(make_float2(cc_166, cc_166), make_float2(rv_131[0], rv_131[1]), value_165);
                        float cc_167 = comb_coeff[23];
                        value_165 = fma_f32x2_rn_noftz(make_float2(cc_167, cc_167), make_float2(rv_131[2], rv_131[3]), value_165);
                        float cc_168 = comb_coeff[27];
                        value_165 = fma_f32x2_rn_noftz(make_float2(cc_168, cc_168), make_float2(rv_131[4], rv_131[5]), value_165);
                        float cc_169 = comb_coeff[31];
                        value_165 = fma_f32x2_rn_noftz(make_float2(cc_169, cc_169), make_float2(rv_131[6], rv_131[7]), value_165);
                        unsigned int word_170 = __as_u32(__float22bfloat162_rn(make_float2(value_165.x, value_165.y)));
                        residual_pairs[io_stage * 8192 + 6144 + offset_128] = word_170;
                        words[15] = word_170;
                        float2 rounded_pair_171 = make_float2(__uint_as_float(word_170 << 16), __uint_as_float(word_170 & 4294901760u));
                        float2 squares_172 = fma_f32x2_rn_noftz(rounded_pair_171, rounded_pair_171, make_float2(sq_136[0], sq_136[1]));
                        sq_136[0] = squares_172.x;
                        sq_136[1] = squares_172.y;
                        float _add_25;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(sq_136[0]), "f"(sq_136[1]));
                        float _sub_3;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_3) : "f"(_add_25), "f"(hc_comp[1]));
                        float _add_3;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(hc_acc[1]), "f"(_sub_3));
                        float _sub_27;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27) : "f"(_add_3), "f"(hc_acc[1]));
                        float _sub_26;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26) : "f"(_sub_27), "f"(_sub_3));
                        hc_comp[1] = _sub_26;
                        hc_acc[1] = _add_3;
                        unsigned int ast = a_seq % 4;
                        unsigned int a_empty_phase = a_seq / 4 % 2 ^ 1;
                        mbarrier_wait(empty_a_addr + (ast) * 8, a_empty_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x4.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"((unsigned int)tmem_a_tmem + ast * 32), "r"(*reinterpret_cast<const uint32_t*>(&words[0])), "r"(*reinterpret_cast<const uint32_t*>(&words[1])), "r"(*reinterpret_cast<const uint32_t*>(&words[2])), "r"(*reinterpret_cast<const uint32_t*>(&words[3])), "r"(*reinterpret_cast<const uint32_t*>(&words[4])), "r"(*reinterpret_cast<const uint32_t*>(&words[5])), "r"(*reinterpret_cast<const uint32_t*>(&words[6])), "r"(*reinterpret_cast<const uint32_t*>(&words[7])), "r"(*reinterpret_cast<const uint32_t*>(&words[8])), "r"(*reinterpret_cast<const uint32_t*>(&words[9])), "r"(*reinterpret_cast<const uint32_t*>(&words[10])), "r"(*reinterpret_cast<const uint32_t*>(&words[11])), "r"(*reinterpret_cast<const uint32_t*>(&words[12])), "r"(*reinterpret_cast<const uint32_t*>(&words[13])), "r"(*reinterpret_cast<const uint32_t*>(&words[14])), "r"(*reinterpret_cast<const uint32_t*>(&words[15])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(full_a_addr + (ast) * 8);
                        a_seq += 2;
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(store_ready_addr + (io_stage) * 8);
                    }
                    io_seq_1 += 1;
                }
                mbarrier_wait(empty_stats_addr, stats_phase);
                stats_phase ^= 1;
                unsigned int row_2 = first_row;
                float _sub_4;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_4) : "f"(hc_acc[0]), "f"(hc_comp[0]));
                float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, _sub_4, 2);
                float _add_4_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4_1) : "f"(_sub_4), "f"(_shfl_xor_sync_28));
                float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, _add_4_1, 1);
                float _add_5;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(_add_4_1), "f"(_shfl_xor_sync_29));
                if (lane_in_warp % 4 == 0) {
                    hc_sums[row_2] = _add_5;
                }
                unsigned int row_0 = first_row + 8;
                float _sub_5_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_5_1) : "f"(hc_acc[1]), "f"(hc_comp[1]));
                float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, _sub_5_1, 2);
                float _add_6;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_6) : "f"(_sub_5_1), "f"(_shfl_xor_sync_30));
                float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, _add_6, 1);
                float _add_7;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(_add_6), "f"(_shfl_xor_sync_31));
                if (lane_in_warp % 4 == 0) {
                    hc_sums[row_0] = _add_7;
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(full_stats_addr);
                }
                __syncwarp();
            }
            asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
        }
    }
    // ---- Role: post1 ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // post1_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("setmaxnreg.inc.sync.aligned.u32 112;");
            unsigned int io_seq_2 = 0;
            unsigned int a_seq_1 = 1;
            unsigned int coeff_seq_2 = 0;
            unsigned int first_row_1 = warp % 4 * 16 + lane_in_warp / 4;
            unsigned int stats_phase_1 = 1;
            #pragma unroll 1
            for (unsigned int task_4 = (unsigned int)bid; task_4 < (num_tokens + 63) / 64 * num_splits; task_4 += num_ctas) {
                uint32_t _fast_div_q_7 = (split_div.divisor != 1) ? (__umulhi((uint32_t)(task_4), split_div.multiplier) >> split_div.shift_right) : (uint32_t)(task_4);
                uint32_t _fast_div_r_7 = (uint32_t)(task_4) - _fast_div_q_7 * (uint32_t)(split_div.divisor);
                unsigned int coeff_stage_2 = coeff_seq_2 % 2;
                unsigned int coeff_phase_1 = coeff_seq_2 / 2 % 2;
                mbarrier_wait(full_coeff_addr + (coeff_stage_2) * 8, coeff_phase_1);
                float post_coeff_1[8];
                float comb_coeff_1[32];
                #pragma unroll
                for (int ri_1 = 0; ri_1 < 2; ri_1++) {
                    unsigned int row_3 = first_row_1 + (unsigned int)(ri_1 * 8);
                    #pragma unroll
                    for (int route_1 = 0; route_1 < 4; route_1++) {
                        post_coeff_1[ri_1 * 4 + route_1] = post[coeff_stage_2 * 256 + row_3 * 4 + (unsigned int)route_1];
                    }
                    #pragma unroll
                    for (int ci_1 = 0; ci_1 < 16; ci_1++) {
                        comb_coeff_1[ri_1 * 16 + ci_1] = comb[coeff_stage_2 * 1024 + row_3 * 16 + ((unsigned int)(ci_1 / 4) ^ row_3 / 2 & 3) * 4 + (unsigned int)(ci_1 % 4)];
                    }
                }
                __syncwarp();
                if (elect_sync()) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(empty_coeff_addr + (coeff_stage_2) * 8);
                }
                coeff_seq_2 += 1;
                float hc_acc_1[2];
                float hc_comp_1[2];
                hc_acc_1[0] = 0.0f;
                hc_comp_1[0] = 0.0f;
                hc_acc_1[1] = 0.0f;
                hc_comp_1[1] = 0.0f;
                #pragma unroll 1
                for (unsigned int kb_4 = 0; kb_4 < _fast_div_q_0 + ((_fast_div_r_7 < _fast_div_r_0) ? 1 : 0); kb_4++) {
                    unsigned int io_stage_1 = io_seq_2 % 3;
                    unsigned int io_phase_1 = io_seq_2 / 3 % 2;
                    mbarrier_wait(full_io_addr + (io_stage_1) * 8, io_phase_1);
                    #pragma unroll
                    for (int atom_1 = 1; atom_1 < 4; atom_1 += 2) {
                        unsigned int words_1[16];
                        unsigned int pair_1 = (unsigned int)(atom_1 * 8) + lane_in_warp % 4 * 2;
                        unsigned int row_4 = first_row_1;
                        unsigned int offset_1 = row_4 * 32 + (pair_1 / 4 ^ row_4 & 7) * 4 + pair_1 % 4;
                        unsigned int x_word_1 = x_pairs[io_stage_1 * 2048 + offset_1];
                        float2 xv_1 = make_float2(__uint_as_float(x_word_1 << 16), __uint_as_float(x_word_1 & 4294901760u));
                        float rv_1[8];
                        unsigned int word_3 = residual_pairs[io_stage_1 * 8192 + offset_1];
                        rv_1[0] = __uint_as_float(word_3 << 16);
                        rv_1[1] = __uint_as_float(word_3 & 4294901760u);
                        unsigned int word_0_1 = residual_pairs[io_stage_1 * 8192 + 2048 + offset_1];
                        rv_1[2] = __uint_as_float(word_0_1 << 16);
                        rv_1[3] = __uint_as_float(word_0_1 & 4294901760u);
                        unsigned int word_1_1 = residual_pairs[io_stage_1 * 8192 + 4096 + offset_1];
                        rv_1[4] = __uint_as_float(word_1_1 << 16);
                        rv_1[5] = __uint_as_float(word_1_1 & 4294901760u);
                        unsigned int word_2_1 = residual_pairs[io_stage_1 * 8192 + 6144 + offset_1];
                        rv_1[6] = __uint_as_float(word_2_1 << 16);
                        rv_1[7] = __uint_as_float(word_2_1 & 4294901760u);
                        float sq_1[2];
                        sq_1[0] = 0.0f;
                        sq_1[1] = 0.0f;
                        float coeff_1 = post_coeff_1[0];
                        float2 _f32x2_mul_a_0 = make_float2(coeff_1, coeff_1);
                        float2 value_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_0), "l"(*(const unsigned long long*)&xv_1));
                        float cc_1 = comb_coeff_1[0];
                        value_1 = fma_f32x2_rn_noftz(make_float2(cc_1, cc_1), make_float2(rv_1[0], rv_1[1]), value_1);
                        float cc_3_1 = comb_coeff_1[4];
                        value_1 = fma_f32x2_rn_noftz(make_float2(cc_3_1, cc_3_1), make_float2(rv_1[2], rv_1[3]), value_1);
                        float cc_4_1 = comb_coeff_1[8];
                        value_1 = fma_f32x2_rn_noftz(make_float2(cc_4_1, cc_4_1), make_float2(rv_1[4], rv_1[5]), value_1);
                        float cc_5_1 = comb_coeff_1[12];
                        value_1 = fma_f32x2_rn_noftz(make_float2(cc_5_1, cc_5_1), make_float2(rv_1[6], rv_1[7]), value_1);
                        unsigned int word_6_1 = __as_u32(__float22bfloat162_rn(make_float2(value_1.x, value_1.y)));
                        residual_pairs[io_stage_1 * 8192 + offset_1] = word_6_1;
                        words_1[0] = word_6_1;
                        float2 rounded_pair_1 = make_float2(__uint_as_float(word_6_1 << 16), __uint_as_float(word_6_1 & 4294901760u));
                        float2 squares_1 = fma_f32x2_rn_noftz(rounded_pair_1, rounded_pair_1, make_float2(sq_1[0], sq_1[1]));
                        sq_1[0] = squares_1.x;
                        sq_1[1] = squares_1.y;
                        float coeff_7_1 = post_coeff_1[1];
                        float2 _f32x2_mul_a_1 = make_float2(coeff_7_1, coeff_7_1);
                        float2 value_8_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_8_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1), "l"(*(const unsigned long long*)&xv_1));
                        float cc_9_1 = comb_coeff_1[1];
                        value_8_1 = fma_f32x2_rn_noftz(make_float2(cc_9_1, cc_9_1), make_float2(rv_1[0], rv_1[1]), value_8_1);
                        float cc_10_1 = comb_coeff_1[5];
                        value_8_1 = fma_f32x2_rn_noftz(make_float2(cc_10_1, cc_10_1), make_float2(rv_1[2], rv_1[3]), value_8_1);
                        float cc_11_1 = comb_coeff_1[9];
                        value_8_1 = fma_f32x2_rn_noftz(make_float2(cc_11_1, cc_11_1), make_float2(rv_1[4], rv_1[5]), value_8_1);
                        float cc_12_1 = comb_coeff_1[13];
                        value_8_1 = fma_f32x2_rn_noftz(make_float2(cc_12_1, cc_12_1), make_float2(rv_1[6], rv_1[7]), value_8_1);
                        unsigned int word_13_1 = __as_u32(__float22bfloat162_rn(make_float2(value_8_1.x, value_8_1.y)));
                        residual_pairs[io_stage_1 * 8192 + 2048 + offset_1] = word_13_1;
                        words_1[4] = word_13_1;
                        float2 rounded_pair_14_1 = make_float2(__uint_as_float(word_13_1 << 16), __uint_as_float(word_13_1 & 4294901760u));
                        float2 squares_15_1 = fma_f32x2_rn_noftz(rounded_pair_14_1, rounded_pair_14_1, make_float2(sq_1[0], sq_1[1]));
                        sq_1[0] = squares_15_1.x;
                        sq_1[1] = squares_15_1.y;
                        float coeff_16_1 = post_coeff_1[2];
                        float2 _f32x2_mul_a_2 = make_float2(coeff_16_1, coeff_16_1);
                        float2 value_17_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_17_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2), "l"(*(const unsigned long long*)&xv_1));
                        float cc_18_1 = comb_coeff_1[2];
                        value_17_1 = fma_f32x2_rn_noftz(make_float2(cc_18_1, cc_18_1), make_float2(rv_1[0], rv_1[1]), value_17_1);
                        float cc_19_1 = comb_coeff_1[6];
                        value_17_1 = fma_f32x2_rn_noftz(make_float2(cc_19_1, cc_19_1), make_float2(rv_1[2], rv_1[3]), value_17_1);
                        float cc_20_1 = comb_coeff_1[10];
                        value_17_1 = fma_f32x2_rn_noftz(make_float2(cc_20_1, cc_20_1), make_float2(rv_1[4], rv_1[5]), value_17_1);
                        float cc_21_1 = comb_coeff_1[14];
                        value_17_1 = fma_f32x2_rn_noftz(make_float2(cc_21_1, cc_21_1), make_float2(rv_1[6], rv_1[7]), value_17_1);
                        unsigned int word_22_1 = __as_u32(__float22bfloat162_rn(make_float2(value_17_1.x, value_17_1.y)));
                        residual_pairs[io_stage_1 * 8192 + 4096 + offset_1] = word_22_1;
                        words_1[8] = word_22_1;
                        float2 rounded_pair_23_1 = make_float2(__uint_as_float(word_22_1 << 16), __uint_as_float(word_22_1 & 4294901760u));
                        float2 squares_24_1 = fma_f32x2_rn_noftz(rounded_pair_23_1, rounded_pair_23_1, make_float2(sq_1[0], sq_1[1]));
                        sq_1[0] = squares_24_1.x;
                        sq_1[1] = squares_24_1.y;
                        float coeff_25_1 = post_coeff_1[3];
                        float2 _f32x2_mul_a_3 = make_float2(coeff_25_1, coeff_25_1);
                        float2 value_26_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_26_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_3), "l"(*(const unsigned long long*)&xv_1));
                        float cc_27_1 = comb_coeff_1[3];
                        value_26_1 = fma_f32x2_rn_noftz(make_float2(cc_27_1, cc_27_1), make_float2(rv_1[0], rv_1[1]), value_26_1);
                        float cc_28_1 = comb_coeff_1[7];
                        value_26_1 = fma_f32x2_rn_noftz(make_float2(cc_28_1, cc_28_1), make_float2(rv_1[2], rv_1[3]), value_26_1);
                        float cc_29_1 = comb_coeff_1[11];
                        value_26_1 = fma_f32x2_rn_noftz(make_float2(cc_29_1, cc_29_1), make_float2(rv_1[4], rv_1[5]), value_26_1);
                        float cc_30_1 = comb_coeff_1[15];
                        value_26_1 = fma_f32x2_rn_noftz(make_float2(cc_30_1, cc_30_1), make_float2(rv_1[6], rv_1[7]), value_26_1);
                        unsigned int word_31_1 = __as_u32(__float22bfloat162_rn(make_float2(value_26_1.x, value_26_1.y)));
                        residual_pairs[io_stage_1 * 8192 + 6144 + offset_1] = word_31_1;
                        words_1[12] = word_31_1;
                        float2 rounded_pair_32_1 = make_float2(__uint_as_float(word_31_1 << 16), __uint_as_float(word_31_1 & 4294901760u));
                        float2 squares_33_1 = fma_f32x2_rn_noftz(rounded_pair_32_1, rounded_pair_32_1, make_float2(sq_1[0], sq_1[1]));
                        sq_1[0] = squares_33_1.x;
                        sq_1[1] = squares_33_1.y;
                        float _add_4_2;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4_2) : "f"(sq_1[0]), "f"(sq_1[1]));
                        float _sub_6_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6_1) : "f"(_add_4_2), "f"(hc_comp_1[0]));
                        float _add_8;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(hc_acc_1[0]), "f"(_sub_6_1));
                        float _sub_6_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6_2) : "f"(_add_8), "f"(hc_acc_1[0]));
                        float _sub_5_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_5_2) : "f"(_sub_6_2), "f"(_sub_6_1));
                        hc_comp_1[0] = _sub_5_2;
                        hc_acc_1[0] = _add_8;
                        unsigned int row_34_1 = first_row_1 + 8;
                        unsigned int offset_35_1 = row_34_1 * 32 + (pair_1 / 4 ^ row_34_1 & 7) * 4 + pair_1 % 4;
                        unsigned int x_word_36_1 = x_pairs[io_stage_1 * 2048 + offset_35_1];
                        float2 xv_37_1 = make_float2(__uint_as_float(x_word_36_1 << 16), __uint_as_float(x_word_36_1 & 4294901760u));
                        float rv_38_1[8];
                        unsigned int word_39_1 = residual_pairs[io_stage_1 * 8192 + offset_35_1];
                        rv_38_1[0] = __uint_as_float(word_39_1 << 16);
                        rv_38_1[1] = __uint_as_float(word_39_1 & 4294901760u);
                        unsigned int word_40_1 = residual_pairs[io_stage_1 * 8192 + 2048 + offset_35_1];
                        rv_38_1[2] = __uint_as_float(word_40_1 << 16);
                        rv_38_1[3] = __uint_as_float(word_40_1 & 4294901760u);
                        unsigned int word_41_1 = residual_pairs[io_stage_1 * 8192 + 4096 + offset_35_1];
                        rv_38_1[4] = __uint_as_float(word_41_1 << 16);
                        rv_38_1[5] = __uint_as_float(word_41_1 & 4294901760u);
                        unsigned int word_42_1 = residual_pairs[io_stage_1 * 8192 + 6144 + offset_35_1];
                        rv_38_1[6] = __uint_as_float(word_42_1 << 16);
                        rv_38_1[7] = __uint_as_float(word_42_1 & 4294901760u);
                        float sq_43_1[2];
                        sq_43_1[0] = 0.0f;
                        sq_43_1[1] = 0.0f;
                        float coeff_44_1 = post_coeff_1[4];
                        float2 _f32x2_mul_a_7 = make_float2(coeff_44_1, coeff_44_1);
                        float2 value_45_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_45_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_7), "l"(*(const unsigned long long*)&xv_37_1));
                        float cc_46_1 = comb_coeff_1[16];
                        value_45_1 = fma_f32x2_rn_noftz(make_float2(cc_46_1, cc_46_1), make_float2(rv_38_1[0], rv_38_1[1]), value_45_1);
                        float cc_47_1 = comb_coeff_1[20];
                        value_45_1 = fma_f32x2_rn_noftz(make_float2(cc_47_1, cc_47_1), make_float2(rv_38_1[2], rv_38_1[3]), value_45_1);
                        float cc_48_1 = comb_coeff_1[24];
                        value_45_1 = fma_f32x2_rn_noftz(make_float2(cc_48_1, cc_48_1), make_float2(rv_38_1[4], rv_38_1[5]), value_45_1);
                        float cc_49_1 = comb_coeff_1[28];
                        value_45_1 = fma_f32x2_rn_noftz(make_float2(cc_49_1, cc_49_1), make_float2(rv_38_1[6], rv_38_1[7]), value_45_1);
                        unsigned int word_50_1 = __as_u32(__float22bfloat162_rn(make_float2(value_45_1.x, value_45_1.y)));
                        residual_pairs[io_stage_1 * 8192 + offset_35_1] = word_50_1;
                        words_1[2] = word_50_1;
                        float2 rounded_pair_51_1 = make_float2(__uint_as_float(word_50_1 << 16), __uint_as_float(word_50_1 & 4294901760u));
                        float2 squares_52_1 = fma_f32x2_rn_noftz(rounded_pair_51_1, rounded_pair_51_1, make_float2(sq_43_1[0], sq_43_1[1]));
                        sq_43_1[0] = squares_52_1.x;
                        sq_43_1[1] = squares_52_1.y;
                        float coeff_53_1 = post_coeff_1[5];
                        float2 _f32x2_mul_a_8 = make_float2(coeff_53_1, coeff_53_1);
                        float2 value_54_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_54_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_8), "l"(*(const unsigned long long*)&xv_37_1));
                        float cc_55_1 = comb_coeff_1[17];
                        value_54_1 = fma_f32x2_rn_noftz(make_float2(cc_55_1, cc_55_1), make_float2(rv_38_1[0], rv_38_1[1]), value_54_1);
                        float cc_56_1 = comb_coeff_1[21];
                        value_54_1 = fma_f32x2_rn_noftz(make_float2(cc_56_1, cc_56_1), make_float2(rv_38_1[2], rv_38_1[3]), value_54_1);
                        float cc_57_1 = comb_coeff_1[25];
                        value_54_1 = fma_f32x2_rn_noftz(make_float2(cc_57_1, cc_57_1), make_float2(rv_38_1[4], rv_38_1[5]), value_54_1);
                        float cc_58_1 = comb_coeff_1[29];
                        value_54_1 = fma_f32x2_rn_noftz(make_float2(cc_58_1, cc_58_1), make_float2(rv_38_1[6], rv_38_1[7]), value_54_1);
                        unsigned int word_59_1 = __as_u32(__float22bfloat162_rn(make_float2(value_54_1.x, value_54_1.y)));
                        residual_pairs[io_stage_1 * 8192 + 2048 + offset_35_1] = word_59_1;
                        words_1[6] = word_59_1;
                        float2 rounded_pair_60_1 = make_float2(__uint_as_float(word_59_1 << 16), __uint_as_float(word_59_1 & 4294901760u));
                        float2 squares_61_1 = fma_f32x2_rn_noftz(rounded_pair_60_1, rounded_pair_60_1, make_float2(sq_43_1[0], sq_43_1[1]));
                        sq_43_1[0] = squares_61_1.x;
                        sq_43_1[1] = squares_61_1.y;
                        float coeff_62_1 = post_coeff_1[6];
                        float2 _f32x2_mul_a_9 = make_float2(coeff_62_1, coeff_62_1);
                        float2 value_63_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_63_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_9), "l"(*(const unsigned long long*)&xv_37_1));
                        float cc_64_1 = comb_coeff_1[18];
                        value_63_1 = fma_f32x2_rn_noftz(make_float2(cc_64_1, cc_64_1), make_float2(rv_38_1[0], rv_38_1[1]), value_63_1);
                        float cc_65_1 = comb_coeff_1[22];
                        value_63_1 = fma_f32x2_rn_noftz(make_float2(cc_65_1, cc_65_1), make_float2(rv_38_1[2], rv_38_1[3]), value_63_1);
                        float cc_66_1 = comb_coeff_1[26];
                        value_63_1 = fma_f32x2_rn_noftz(make_float2(cc_66_1, cc_66_1), make_float2(rv_38_1[4], rv_38_1[5]), value_63_1);
                        float cc_67_1 = comb_coeff_1[30];
                        value_63_1 = fma_f32x2_rn_noftz(make_float2(cc_67_1, cc_67_1), make_float2(rv_38_1[6], rv_38_1[7]), value_63_1);
                        unsigned int word_68_1 = __as_u32(__float22bfloat162_rn(make_float2(value_63_1.x, value_63_1.y)));
                        residual_pairs[io_stage_1 * 8192 + 4096 + offset_35_1] = word_68_1;
                        words_1[10] = word_68_1;
                        float2 rounded_pair_69_1 = make_float2(__uint_as_float(word_68_1 << 16), __uint_as_float(word_68_1 & 4294901760u));
                        float2 squares_70_1 = fma_f32x2_rn_noftz(rounded_pair_69_1, rounded_pair_69_1, make_float2(sq_43_1[0], sq_43_1[1]));
                        sq_43_1[0] = squares_70_1.x;
                        sq_43_1[1] = squares_70_1.y;
                        float coeff_71_1 = post_coeff_1[7];
                        float2 _f32x2_mul_a_10 = make_float2(coeff_71_1, coeff_71_1);
                        float2 value_72_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_72_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_10), "l"(*(const unsigned long long*)&xv_37_1));
                        float cc_73_1 = comb_coeff_1[19];
                        value_72_1 = fma_f32x2_rn_noftz(make_float2(cc_73_1, cc_73_1), make_float2(rv_38_1[0], rv_38_1[1]), value_72_1);
                        float cc_74_1 = comb_coeff_1[23];
                        value_72_1 = fma_f32x2_rn_noftz(make_float2(cc_74_1, cc_74_1), make_float2(rv_38_1[2], rv_38_1[3]), value_72_1);
                        float cc_75_1 = comb_coeff_1[27];
                        value_72_1 = fma_f32x2_rn_noftz(make_float2(cc_75_1, cc_75_1), make_float2(rv_38_1[4], rv_38_1[5]), value_72_1);
                        float cc_76_1 = comb_coeff_1[31];
                        value_72_1 = fma_f32x2_rn_noftz(make_float2(cc_76_1, cc_76_1), make_float2(rv_38_1[6], rv_38_1[7]), value_72_1);
                        unsigned int word_77_1 = __as_u32(__float22bfloat162_rn(make_float2(value_72_1.x, value_72_1.y)));
                        residual_pairs[io_stage_1 * 8192 + 6144 + offset_35_1] = word_77_1;
                        words_1[14] = word_77_1;
                        float2 rounded_pair_78_1 = make_float2(__uint_as_float(word_77_1 << 16), __uint_as_float(word_77_1 & 4294901760u));
                        float2 squares_79_1 = fma_f32x2_rn_noftz(rounded_pair_78_1, rounded_pair_78_1, make_float2(sq_43_1[0], sq_43_1[1]));
                        sq_43_1[0] = squares_79_1.x;
                        sq_43_1[1] = squares_79_1.y;
                        float _add_11_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11_1) : "f"(sq_43_1[0]), "f"(sq_43_1[1]));
                        float _sub_7;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_7) : "f"(_add_11_1), "f"(hc_comp_1[1]));
                        float _add_9;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(hc_acc_1[1]), "f"(_sub_7));
                        float _sub_13_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13_1) : "f"(_add_9), "f"(hc_acc_1[1]));
                        float _sub_12_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_12_1) : "f"(_sub_13_1), "f"(_sub_7));
                        hc_comp_1[1] = _sub_12_1;
                        hc_acc_1[1] = _add_9;
                        unsigned int pair_80_1 = (unsigned int)(atom_1 * 8 + 1) + lane_in_warp % 4 * 2;
                        unsigned int row_81_1 = first_row_1;
                        unsigned int offset_82_1 = row_81_1 * 32 + (pair_80_1 / 4 ^ row_81_1 & 7) * 4 + pair_80_1 % 4;
                        unsigned int x_word_83_1 = x_pairs[io_stage_1 * 2048 + offset_82_1];
                        float2 xv_84_1 = make_float2(__uint_as_float(x_word_83_1 << 16), __uint_as_float(x_word_83_1 & 4294901760u));
                        float rv_85_1[8];
                        unsigned int word_86_1 = residual_pairs[io_stage_1 * 8192 + offset_82_1];
                        rv_85_1[0] = __uint_as_float(word_86_1 << 16);
                        rv_85_1[1] = __uint_as_float(word_86_1 & 4294901760u);
                        unsigned int word_87_1 = residual_pairs[io_stage_1 * 8192 + 2048 + offset_82_1];
                        rv_85_1[2] = __uint_as_float(word_87_1 << 16);
                        rv_85_1[3] = __uint_as_float(word_87_1 & 4294901760u);
                        unsigned int word_88_1 = residual_pairs[io_stage_1 * 8192 + 4096 + offset_82_1];
                        rv_85_1[4] = __uint_as_float(word_88_1 << 16);
                        rv_85_1[5] = __uint_as_float(word_88_1 & 4294901760u);
                        unsigned int word_89_1 = residual_pairs[io_stage_1 * 8192 + 6144 + offset_82_1];
                        rv_85_1[6] = __uint_as_float(word_89_1 << 16);
                        rv_85_1[7] = __uint_as_float(word_89_1 & 4294901760u);
                        float sq_90_1[2];
                        sq_90_1[0] = 0.0f;
                        sq_90_1[1] = 0.0f;
                        float coeff_91_1 = post_coeff_1[0];
                        float2 _f32x2_mul_a_14 = make_float2(coeff_91_1, coeff_91_1);
                        float2 value_92_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_92_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&xv_84_1));
                        float cc_93_1 = comb_coeff_1[0];
                        value_92_1 = fma_f32x2_rn_noftz(make_float2(cc_93_1, cc_93_1), make_float2(rv_85_1[0], rv_85_1[1]), value_92_1);
                        float cc_94_1 = comb_coeff_1[4];
                        value_92_1 = fma_f32x2_rn_noftz(make_float2(cc_94_1, cc_94_1), make_float2(rv_85_1[2], rv_85_1[3]), value_92_1);
                        float cc_95_1 = comb_coeff_1[8];
                        value_92_1 = fma_f32x2_rn_noftz(make_float2(cc_95_1, cc_95_1), make_float2(rv_85_1[4], rv_85_1[5]), value_92_1);
                        float cc_96_1 = comb_coeff_1[12];
                        value_92_1 = fma_f32x2_rn_noftz(make_float2(cc_96_1, cc_96_1), make_float2(rv_85_1[6], rv_85_1[7]), value_92_1);
                        unsigned int word_97_1 = __as_u32(__float22bfloat162_rn(make_float2(value_92_1.x, value_92_1.y)));
                        residual_pairs[io_stage_1 * 8192 + offset_82_1] = word_97_1;
                        words_1[1] = word_97_1;
                        float2 rounded_pair_98_1 = make_float2(__uint_as_float(word_97_1 << 16), __uint_as_float(word_97_1 & 4294901760u));
                        float2 squares_99_1 = fma_f32x2_rn_noftz(rounded_pair_98_1, rounded_pair_98_1, make_float2(sq_90_1[0], sq_90_1[1]));
                        sq_90_1[0] = squares_99_1.x;
                        sq_90_1[1] = squares_99_1.y;
                        float coeff_100_1 = post_coeff_1[1];
                        float2 _f32x2_mul_a_15 = make_float2(coeff_100_1, coeff_100_1);
                        float2 value_101_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_101_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&xv_84_1));
                        float cc_102_1 = comb_coeff_1[1];
                        value_101_1 = fma_f32x2_rn_noftz(make_float2(cc_102_1, cc_102_1), make_float2(rv_85_1[0], rv_85_1[1]), value_101_1);
                        float cc_103_1 = comb_coeff_1[5];
                        value_101_1 = fma_f32x2_rn_noftz(make_float2(cc_103_1, cc_103_1), make_float2(rv_85_1[2], rv_85_1[3]), value_101_1);
                        float cc_104_1 = comb_coeff_1[9];
                        value_101_1 = fma_f32x2_rn_noftz(make_float2(cc_104_1, cc_104_1), make_float2(rv_85_1[4], rv_85_1[5]), value_101_1);
                        float cc_105_1 = comb_coeff_1[13];
                        value_101_1 = fma_f32x2_rn_noftz(make_float2(cc_105_1, cc_105_1), make_float2(rv_85_1[6], rv_85_1[7]), value_101_1);
                        unsigned int word_106_1 = __as_u32(__float22bfloat162_rn(make_float2(value_101_1.x, value_101_1.y)));
                        residual_pairs[io_stage_1 * 8192 + 2048 + offset_82_1] = word_106_1;
                        words_1[5] = word_106_1;
                        float2 rounded_pair_107_1 = make_float2(__uint_as_float(word_106_1 << 16), __uint_as_float(word_106_1 & 4294901760u));
                        float2 squares_108_1 = fma_f32x2_rn_noftz(rounded_pair_107_1, rounded_pair_107_1, make_float2(sq_90_1[0], sq_90_1[1]));
                        sq_90_1[0] = squares_108_1.x;
                        sq_90_1[1] = squares_108_1.y;
                        float coeff_109_1 = post_coeff_1[2];
                        float2 _f32x2_mul_a_16 = make_float2(coeff_109_1, coeff_109_1);
                        float2 value_110_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_110_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_16), "l"(*(const unsigned long long*)&xv_84_1));
                        float cc_111_1 = comb_coeff_1[2];
                        value_110_1 = fma_f32x2_rn_noftz(make_float2(cc_111_1, cc_111_1), make_float2(rv_85_1[0], rv_85_1[1]), value_110_1);
                        float cc_112_1 = comb_coeff_1[6];
                        value_110_1 = fma_f32x2_rn_noftz(make_float2(cc_112_1, cc_112_1), make_float2(rv_85_1[2], rv_85_1[3]), value_110_1);
                        float cc_113_1 = comb_coeff_1[10];
                        value_110_1 = fma_f32x2_rn_noftz(make_float2(cc_113_1, cc_113_1), make_float2(rv_85_1[4], rv_85_1[5]), value_110_1);
                        float cc_114_1 = comb_coeff_1[14];
                        value_110_1 = fma_f32x2_rn_noftz(make_float2(cc_114_1, cc_114_1), make_float2(rv_85_1[6], rv_85_1[7]), value_110_1);
                        unsigned int word_115_1 = __as_u32(__float22bfloat162_rn(make_float2(value_110_1.x, value_110_1.y)));
                        residual_pairs[io_stage_1 * 8192 + 4096 + offset_82_1] = word_115_1;
                        words_1[9] = word_115_1;
                        float2 rounded_pair_116_1 = make_float2(__uint_as_float(word_115_1 << 16), __uint_as_float(word_115_1 & 4294901760u));
                        float2 squares_117_1 = fma_f32x2_rn_noftz(rounded_pair_116_1, rounded_pair_116_1, make_float2(sq_90_1[0], sq_90_1[1]));
                        sq_90_1[0] = squares_117_1.x;
                        sq_90_1[1] = squares_117_1.y;
                        float coeff_118_1 = post_coeff_1[3];
                        float2 _f32x2_mul_a_17 = make_float2(coeff_118_1, coeff_118_1);
                        float2 value_119_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_119_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&xv_84_1));
                        float cc_120_1 = comb_coeff_1[3];
                        value_119_1 = fma_f32x2_rn_noftz(make_float2(cc_120_1, cc_120_1), make_float2(rv_85_1[0], rv_85_1[1]), value_119_1);
                        float cc_121_1 = comb_coeff_1[7];
                        value_119_1 = fma_f32x2_rn_noftz(make_float2(cc_121_1, cc_121_1), make_float2(rv_85_1[2], rv_85_1[3]), value_119_1);
                        float cc_122_1 = comb_coeff_1[11];
                        value_119_1 = fma_f32x2_rn_noftz(make_float2(cc_122_1, cc_122_1), make_float2(rv_85_1[4], rv_85_1[5]), value_119_1);
                        float cc_123_1 = comb_coeff_1[15];
                        value_119_1 = fma_f32x2_rn_noftz(make_float2(cc_123_1, cc_123_1), make_float2(rv_85_1[6], rv_85_1[7]), value_119_1);
                        unsigned int word_124_1 = __as_u32(__float22bfloat162_rn(make_float2(value_119_1.x, value_119_1.y)));
                        residual_pairs[io_stage_1 * 8192 + 6144 + offset_82_1] = word_124_1;
                        words_1[13] = word_124_1;
                        float2 rounded_pair_125_1 = make_float2(__uint_as_float(word_124_1 << 16), __uint_as_float(word_124_1 & 4294901760u));
                        float2 squares_126_1 = fma_f32x2_rn_noftz(rounded_pair_125_1, rounded_pair_125_1, make_float2(sq_90_1[0], sq_90_1[1]));
                        sq_90_1[0] = squares_126_1.x;
                        sq_90_1[1] = squares_126_1.y;
                        float _add_18_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18_1) : "f"(sq_90_1[0]), "f"(sq_90_1[1]));
                        float _sub_8;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_8) : "f"(_add_18_1), "f"(hc_comp_1[0]));
                        float _add_10;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(hc_acc_1[0]), "f"(_sub_8));
                        float _sub_20_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20_1) : "f"(_add_10), "f"(hc_acc_1[0]));
                        float _sub_19_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19_1) : "f"(_sub_20_1), "f"(_sub_8));
                        hc_comp_1[0] = _sub_19_1;
                        hc_acc_1[0] = _add_10;
                        unsigned int row_127_1 = first_row_1 + 8;
                        unsigned int offset_128_1 = row_127_1 * 32 + (pair_80_1 / 4 ^ row_127_1 & 7) * 4 + pair_80_1 % 4;
                        unsigned int x_word_129_1 = x_pairs[io_stage_1 * 2048 + offset_128_1];
                        float2 xv_130_1 = make_float2(__uint_as_float(x_word_129_1 << 16), __uint_as_float(x_word_129_1 & 4294901760u));
                        float rv_131_1[8];
                        unsigned int word_132_1 = residual_pairs[io_stage_1 * 8192 + offset_128_1];
                        rv_131_1[0] = __uint_as_float(word_132_1 << 16);
                        rv_131_1[1] = __uint_as_float(word_132_1 & 4294901760u);
                        unsigned int word_133_1 = residual_pairs[io_stage_1 * 8192 + 2048 + offset_128_1];
                        rv_131_1[2] = __uint_as_float(word_133_1 << 16);
                        rv_131_1[3] = __uint_as_float(word_133_1 & 4294901760u);
                        unsigned int word_134_1 = residual_pairs[io_stage_1 * 8192 + 4096 + offset_128_1];
                        rv_131_1[4] = __uint_as_float(word_134_1 << 16);
                        rv_131_1[5] = __uint_as_float(word_134_1 & 4294901760u);
                        unsigned int word_135_1 = residual_pairs[io_stage_1 * 8192 + 6144 + offset_128_1];
                        rv_131_1[6] = __uint_as_float(word_135_1 << 16);
                        rv_131_1[7] = __uint_as_float(word_135_1 & 4294901760u);
                        float sq_136_1[2];
                        sq_136_1[0] = 0.0f;
                        sq_136_1[1] = 0.0f;
                        float coeff_137_1 = post_coeff_1[4];
                        float2 _f32x2_mul_a_21 = make_float2(coeff_137_1, coeff_137_1);
                        float2 value_138_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_138_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&xv_130_1));
                        float cc_139_1 = comb_coeff_1[16];
                        value_138_1 = fma_f32x2_rn_noftz(make_float2(cc_139_1, cc_139_1), make_float2(rv_131_1[0], rv_131_1[1]), value_138_1);
                        float cc_140_1 = comb_coeff_1[20];
                        value_138_1 = fma_f32x2_rn_noftz(make_float2(cc_140_1, cc_140_1), make_float2(rv_131_1[2], rv_131_1[3]), value_138_1);
                        float cc_141_1 = comb_coeff_1[24];
                        value_138_1 = fma_f32x2_rn_noftz(make_float2(cc_141_1, cc_141_1), make_float2(rv_131_1[4], rv_131_1[5]), value_138_1);
                        float cc_142_1 = comb_coeff_1[28];
                        value_138_1 = fma_f32x2_rn_noftz(make_float2(cc_142_1, cc_142_1), make_float2(rv_131_1[6], rv_131_1[7]), value_138_1);
                        unsigned int word_143_1 = __as_u32(__float22bfloat162_rn(make_float2(value_138_1.x, value_138_1.y)));
                        residual_pairs[io_stage_1 * 8192 + offset_128_1] = word_143_1;
                        words_1[3] = word_143_1;
                        float2 rounded_pair_144_1 = make_float2(__uint_as_float(word_143_1 << 16), __uint_as_float(word_143_1 & 4294901760u));
                        float2 squares_145_1 = fma_f32x2_rn_noftz(rounded_pair_144_1, rounded_pair_144_1, make_float2(sq_136_1[0], sq_136_1[1]));
                        sq_136_1[0] = squares_145_1.x;
                        sq_136_1[1] = squares_145_1.y;
                        float coeff_146_1 = post_coeff_1[5];
                        float2 _f32x2_mul_a_22 = make_float2(coeff_146_1, coeff_146_1);
                        float2 value_147_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_147_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_22), "l"(*(const unsigned long long*)&xv_130_1));
                        float cc_148_1 = comb_coeff_1[17];
                        value_147_1 = fma_f32x2_rn_noftz(make_float2(cc_148_1, cc_148_1), make_float2(rv_131_1[0], rv_131_1[1]), value_147_1);
                        float cc_149_1 = comb_coeff_1[21];
                        value_147_1 = fma_f32x2_rn_noftz(make_float2(cc_149_1, cc_149_1), make_float2(rv_131_1[2], rv_131_1[3]), value_147_1);
                        float cc_150_1 = comb_coeff_1[25];
                        value_147_1 = fma_f32x2_rn_noftz(make_float2(cc_150_1, cc_150_1), make_float2(rv_131_1[4], rv_131_1[5]), value_147_1);
                        float cc_151_1 = comb_coeff_1[29];
                        value_147_1 = fma_f32x2_rn_noftz(make_float2(cc_151_1, cc_151_1), make_float2(rv_131_1[6], rv_131_1[7]), value_147_1);
                        unsigned int word_152_1 = __as_u32(__float22bfloat162_rn(make_float2(value_147_1.x, value_147_1.y)));
                        residual_pairs[io_stage_1 * 8192 + 2048 + offset_128_1] = word_152_1;
                        words_1[7] = word_152_1;
                        float2 rounded_pair_153_1 = make_float2(__uint_as_float(word_152_1 << 16), __uint_as_float(word_152_1 & 4294901760u));
                        float2 squares_154_1 = fma_f32x2_rn_noftz(rounded_pair_153_1, rounded_pair_153_1, make_float2(sq_136_1[0], sq_136_1[1]));
                        sq_136_1[0] = squares_154_1.x;
                        sq_136_1[1] = squares_154_1.y;
                        float coeff_155_1 = post_coeff_1[6];
                        float2 _f32x2_mul_a_23 = make_float2(coeff_155_1, coeff_155_1);
                        float2 value_156_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_156_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&xv_130_1));
                        float cc_157_1 = comb_coeff_1[18];
                        value_156_1 = fma_f32x2_rn_noftz(make_float2(cc_157_1, cc_157_1), make_float2(rv_131_1[0], rv_131_1[1]), value_156_1);
                        float cc_158_1 = comb_coeff_1[22];
                        value_156_1 = fma_f32x2_rn_noftz(make_float2(cc_158_1, cc_158_1), make_float2(rv_131_1[2], rv_131_1[3]), value_156_1);
                        float cc_159_1 = comb_coeff_1[26];
                        value_156_1 = fma_f32x2_rn_noftz(make_float2(cc_159_1, cc_159_1), make_float2(rv_131_1[4], rv_131_1[5]), value_156_1);
                        float cc_160_1 = comb_coeff_1[30];
                        value_156_1 = fma_f32x2_rn_noftz(make_float2(cc_160_1, cc_160_1), make_float2(rv_131_1[6], rv_131_1[7]), value_156_1);
                        unsigned int word_161_1 = __as_u32(__float22bfloat162_rn(make_float2(value_156_1.x, value_156_1.y)));
                        residual_pairs[io_stage_1 * 8192 + 4096 + offset_128_1] = word_161_1;
                        words_1[11] = word_161_1;
                        float2 rounded_pair_162_1 = make_float2(__uint_as_float(word_161_1 << 16), __uint_as_float(word_161_1 & 4294901760u));
                        float2 squares_163_1 = fma_f32x2_rn_noftz(rounded_pair_162_1, rounded_pair_162_1, make_float2(sq_136_1[0], sq_136_1[1]));
                        sq_136_1[0] = squares_163_1.x;
                        sq_136_1[1] = squares_163_1.y;
                        float coeff_164_1 = post_coeff_1[7];
                        float2 _f32x2_mul_a_24 = make_float2(coeff_164_1, coeff_164_1);
                        float2 value_165_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&value_165_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&xv_130_1));
                        float cc_166_1 = comb_coeff_1[19];
                        value_165_1 = fma_f32x2_rn_noftz(make_float2(cc_166_1, cc_166_1), make_float2(rv_131_1[0], rv_131_1[1]), value_165_1);
                        float cc_167_1 = comb_coeff_1[23];
                        value_165_1 = fma_f32x2_rn_noftz(make_float2(cc_167_1, cc_167_1), make_float2(rv_131_1[2], rv_131_1[3]), value_165_1);
                        float cc_168_1 = comb_coeff_1[27];
                        value_165_1 = fma_f32x2_rn_noftz(make_float2(cc_168_1, cc_168_1), make_float2(rv_131_1[4], rv_131_1[5]), value_165_1);
                        float cc_169_1 = comb_coeff_1[31];
                        value_165_1 = fma_f32x2_rn_noftz(make_float2(cc_169_1, cc_169_1), make_float2(rv_131_1[6], rv_131_1[7]), value_165_1);
                        unsigned int word_170_1 = __as_u32(__float22bfloat162_rn(make_float2(value_165_1.x, value_165_1.y)));
                        residual_pairs[io_stage_1 * 8192 + 6144 + offset_128_1] = word_170_1;
                        words_1[15] = word_170_1;
                        float2 rounded_pair_171_1 = make_float2(__uint_as_float(word_170_1 << 16), __uint_as_float(word_170_1 & 4294901760u));
                        float2 squares_172_1 = fma_f32x2_rn_noftz(rounded_pair_171_1, rounded_pair_171_1, make_float2(sq_136_1[0], sq_136_1[1]));
                        sq_136_1[0] = squares_172_1.x;
                        sq_136_1[1] = squares_172_1.y;
                        float _add_25_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25_1) : "f"(sq_136_1[0]), "f"(sq_136_1[1]));
                        float _sub_9;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_9) : "f"(_add_25_1), "f"(hc_comp_1[1]));
                        float _add_11_2;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11_2) : "f"(hc_acc_1[1]), "f"(_sub_9));
                        float _sub_27_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27_1) : "f"(_add_11_2), "f"(hc_acc_1[1]));
                        float _sub_26_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26_1) : "f"(_sub_27_1), "f"(_sub_9));
                        hc_comp_1[1] = _sub_26_1;
                        hc_acc_1[1] = _add_11_2;
                        unsigned int ast_1 = a_seq_1 % 4;
                        unsigned int a_empty_phase_1 = a_seq_1 / 4 % 2 ^ 1;
                        mbarrier_wait(empty_a_addr + (ast_1) * 8, a_empty_phase_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x4.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"((unsigned int)tmem_a_tmem + ast_1 * 32), "r"(*reinterpret_cast<const uint32_t*>(&words_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&words_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&words_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&words_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&words_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&words_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&words_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&words_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&words_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&words_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&words_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&words_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&words_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&words_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&words_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&words_1[15])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(full_a_addr + (ast_1) * 8);
                        a_seq_1 += 2;
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(store_ready_addr + (io_stage_1) * 8);
                    }
                    io_seq_2 += 1;
                }
                mbarrier_wait(empty_stats_addr, stats_phase_1);
                stats_phase_1 ^= 1;
                unsigned int row_5 = first_row_1;
                float _sub_10;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_10) : "f"(hc_acc_1[0]), "f"(hc_comp_1[0]));
                float _shfl_xor_sync_28_1 = __shfl_xor_sync(0xFFFFFFFF, _sub_10, 2);
                float _add_12;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(_sub_10), "f"(_shfl_xor_sync_28_1));
                float _shfl_xor_sync_29_1 = __shfl_xor_sync(0xFFFFFFFF, _add_12, 1);
                float _add_13;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(_add_12), "f"(_shfl_xor_sync_29_1));
                if (lane_in_warp % 4 == 0) {
                    hc_sums[64 + row_5] = _add_13;
                }
                unsigned int row_0_1 = first_row_1 + 8;
                float _sub_11;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11) : "f"(hc_acc_1[1]), "f"(hc_comp_1[1]));
                float _shfl_xor_sync_30_1 = __shfl_xor_sync(0xFFFFFFFF, _sub_11, 2);
                float _add_14;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(_sub_11), "f"(_shfl_xor_sync_30_1));
                float _shfl_xor_sync_31_1 = __shfl_xor_sync(0xFFFFFFFF, _add_14, 1);
                float _add_15;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(_add_14), "f"(_shfl_xor_sync_31_1));
                if (lane_in_warp % 4 == 0) {
                    hc_sums[64 + row_0_1] = _add_15;
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(full_stats_addr);
                }
                __syncwarp();
            }
            asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
        }
    }
    // ---- Role: workspace ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // workspace_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
            unsigned int seq_1 = 0;
            unsigned int cseq_1 = 0;
            unsigned int first_row_2 = warp % 4 * 16 + lane_in_warp / 4;
            #pragma unroll 1
            for (unsigned int task_5 = (unsigned int)bid; task_5 < (num_tokens + 63) / 64 * num_splits; task_5 += num_ctas) {
                uint32_t _fast_div_q_8 = (split_div.divisor != 1) ? (__umulhi((uint32_t)(task_5), split_div.multiplier) >> split_div.shift_right) : (uint32_t)(task_5);
                uint32_t _fast_div_r_8 = (uint32_t)(task_5) - _fast_div_q_8 * (uint32_t)(split_div.divisor);
                if (warp % 4 == 0) {
                    unsigned long long base = (launch_epochs[bid] + 1) * 128;
                    {
                    unsigned long long _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_acquire_observed) : "l"(reinterpret_cast<unsigned long long*>(split_barriers + ((16384 + _fast_div_q_8) * 16))) : "memory");
                    } while (static_cast<unsigned long long>(_acquire_observed - static_cast<unsigned long long>(base)) >= static_cast<unsigned long long>((unsigned long long)num_splits));
                    }
                }
                float acc[12];
                float comp[12];
                acc[0] = 0.0f;
                comp[0] = 0.0f;
                acc[1] = 0.0f;
                comp[1] = 0.0f;
                acc[2] = 0.0f;
                comp[2] = 0.0f;
                acc[3] = 0.0f;
                comp[3] = 0.0f;
                acc[4] = 0.0f;
                comp[4] = 0.0f;
                acc[5] = 0.0f;
                comp[5] = 0.0f;
                acc[6] = 0.0f;
                comp[6] = 0.0f;
                acc[7] = 0.0f;
                comp[7] = 0.0f;
                acc[8] = 0.0f;
                comp[8] = 0.0f;
                acc[9] = 0.0f;
                comp[9] = 0.0f;
                acc[10] = 0.0f;
                comp[10] = 0.0f;
                acc[11] = 0.0f;
                comp[11] = 0.0f;
                #pragma unroll 1
                for (unsigned int chunk = 0; chunk < (_fast_div_q_0 + ((_fast_div_r_8 < _fast_div_r_0) ? 1 : 0)) * 2; chunk++) {
                    unsigned int cstage_1 = cseq_1 % 2;
                    unsigned int big_phase = cseq_1 / 2 % 2;
                    mbarrier_wait(big_full_addr + (cstage_1) * 8, big_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_0[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                        : "r"((unsigned int)tmem_acc_big + cstage_1 * 24));
                    float _tmem_load_1[4];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                        " {%0, %1, %2, %3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3]))
                        : "r"((unsigned int)tmem_acc_big + (cstage_1 * 24 + 16)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    if (elect_sync()) {
                        mbarrier_arrive(big_empty_addr + (cstage_1) * 8);
                    }
                    float _sub_12_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_12_2) : "f"(_tmem_load_0[0]), "f"(comp[0]));
                    float _add_16;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(acc[0]), "f"(_sub_12_2));
                    float _sub_1_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1_1) : "f"(_add_16), "f"(acc[0]));
                    float _sub_0_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_0_1) : "f"(_sub_1_1), "f"(_sub_12_2));
                    comp[0] = _sub_0_1;
                    acc[0] = _add_16;
                    float _sub_13_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13_2) : "f"(_tmem_load_0[1]), "f"(comp[1]));
                    float _add_17;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(acc[1]), "f"(_sub_13_2));
                    float _sub_3_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_3_1) : "f"(_add_17), "f"(acc[1]));
                    float _sub_2_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_2_1) : "f"(_sub_3_1), "f"(_sub_13_2));
                    comp[1] = _sub_2_1;
                    acc[1] = _add_17;
                    float _sub_14;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_14) : "f"(_tmem_load_0[2]), "f"(comp[2]));
                    float _add_18_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18_2) : "f"(acc[2]), "f"(_sub_14));
                    float _sub_5_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_5_3) : "f"(_add_18_2), "f"(acc[2]));
                    float _sub_4_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_4_1) : "f"(_sub_5_3), "f"(_sub_14));
                    comp[2] = _sub_4_1;
                    acc[2] = _add_18_2;
                    float _sub_15;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15) : "f"(_tmem_load_0[3]), "f"(comp[3]));
                    float _add_19;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(acc[3]), "f"(_sub_15));
                    float _sub_7_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_7_1) : "f"(_add_19), "f"(acc[3]));
                    float _sub_6_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6_3) : "f"(_sub_7_1), "f"(_sub_15));
                    comp[3] = _sub_6_3;
                    acc[3] = _add_19;
                    float _sub_16;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_16) : "f"(_tmem_load_0[4]), "f"(comp[4]));
                    float _add_20;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(acc[4]), "f"(_sub_16));
                    float _sub_9_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_9_1) : "f"(_add_20), "f"(acc[4]));
                    float _sub_8_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_8_1) : "f"(_sub_9_1), "f"(_sub_16));
                    comp[4] = _sub_8_1;
                    acc[4] = _add_20;
                    float _sub_17;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17) : "f"(_tmem_load_0[5]), "f"(comp[5]));
                    float _add_21;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(acc[5]), "f"(_sub_17));
                    float _sub_11_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11_1) : "f"(_add_21), "f"(acc[5]));
                    float _sub_10_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_10_1) : "f"(_sub_11_1), "f"(_sub_17));
                    comp[5] = _sub_10_1;
                    acc[5] = _add_21;
                    float _sub_18;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18) : "f"(_tmem_load_0[6]), "f"(comp[6]));
                    float _add_22;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(acc[6]), "f"(_sub_18));
                    float _sub_13_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13_3) : "f"(_add_22), "f"(acc[6]));
                    float _sub_12_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_12_3) : "f"(_sub_13_3), "f"(_sub_18));
                    comp[6] = _sub_12_3;
                    acc[6] = _add_22;
                    float _sub_19_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19_2) : "f"(_tmem_load_0[7]), "f"(comp[7]));
                    float _add_23;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23) : "f"(acc[7]), "f"(_sub_19_2));
                    float _sub_15_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15_1) : "f"(_add_23), "f"(acc[7]));
                    float _sub_14_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_14_1) : "f"(_sub_15_1), "f"(_sub_19_2));
                    comp[7] = _sub_14_1;
                    acc[7] = _add_23;
                    float _sub_20_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20_2) : "f"(_tmem_load_1[0]), "f"(comp[8]));
                    float _add_24;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(acc[8]), "f"(_sub_20_2));
                    float _sub_17_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17_1) : "f"(_add_24), "f"(acc[8]));
                    float _sub_16_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_16_1) : "f"(_sub_17_1), "f"(_sub_20_2));
                    comp[8] = _sub_16_1;
                    acc[8] = _add_24;
                    float _sub_21;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_21) : "f"(_tmem_load_1[1]), "f"(comp[9]));
                    float _add_25_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25_2) : "f"(acc[9]), "f"(_sub_21));
                    float _sub_19_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19_3) : "f"(_add_25_2), "f"(acc[9]));
                    float _sub_18_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18_1) : "f"(_sub_19_3), "f"(_sub_21));
                    comp[9] = _sub_18_1;
                    acc[9] = _add_25_2;
                    float _sub_22;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_22) : "f"(_tmem_load_1[2]), "f"(comp[10]));
                    float _add_26;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(acc[10]), "f"(_sub_22));
                    float _sub_21_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_21_1) : "f"(_add_26), "f"(acc[10]));
                    float _sub_20_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20_3) : "f"(_sub_21_1), "f"(_sub_22));
                    comp[10] = _sub_20_3;
                    acc[10] = _add_26;
                    float _sub_23;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23) : "f"(_tmem_load_1[3]), "f"(comp[11]));
                    float _add_27;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(acc[11]), "f"(_sub_23));
                    float _sub_23_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23_1) : "f"(_add_27), "f"(acc[11]));
                    float _sub_22_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_22_1) : "f"(_sub_23_1), "f"(_sub_23));
                    comp[11] = _sub_22_1;
                    acc[11] = _add_27;
                    cseq_1 += 1;
                }
                unsigned int ustage_1 = seq_1 % 2;
                unsigned int small_phase = seq_1 / 2 % 2;
                mbarrier_wait(small_full_addr + (ustage_1) * 8, small_phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_2[8];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7]))
                    : "r"((unsigned int)tmem_acc_small + ustage_1 * 24));
                float _tmem_load_3[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3]))
                    : "r"((unsigned int)tmem_acc_small + (ustage_1 * 24 + 16)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                if (elect_sync()) {
                    mbarrier_arrive(small_empty_addr + (ustage_1) * 8);
                }
                float _sub_24;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_24) : "f"(_tmem_load_2[0]), "f"(comp[0]));
                float _add_28;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(acc[0]), "f"(_sub_24));
                float _sub_25;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_25) : "f"(_add_28), "f"(acc[0]));
                float _sub_24_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_24_1) : "f"(_sub_25), "f"(_sub_24));
                comp[0] = _sub_24_1;
                acc[0] = _add_28;
                float _sub_25_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_25_1) : "f"(_tmem_load_2[1]), "f"(comp[1]));
                float _add_29;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(acc[1]), "f"(_sub_25_1));
                float _sub_27_2;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27_2) : "f"(_add_29), "f"(acc[1]));
                float _sub_26_2;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26_2) : "f"(_sub_27_2), "f"(_sub_25_1));
                comp[1] = _sub_26_2;
                acc[1] = _add_29;
                float _sub_26_3;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26_3) : "f"(_tmem_load_2[2]), "f"(comp[2]));
                float _add_30;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(acc[2]), "f"(_sub_26_3));
                float _sub_29;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_29) : "f"(_add_30), "f"(acc[2]));
                float _sub_28;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28) : "f"(_sub_29), "f"(_sub_26_3));
                comp[2] = _sub_28;
                acc[2] = _add_30;
                float _sub_27_3;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27_3) : "f"(_tmem_load_2[3]), "f"(comp[3]));
                float _add_31;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(acc[3]), "f"(_sub_27_3));
                float _sub_31;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31) : "f"(_add_31), "f"(acc[3]));
                float _sub_30;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_30) : "f"(_sub_31), "f"(_sub_27_3));
                comp[3] = _sub_30;
                acc[3] = _add_31;
                float _sub_28_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28_1) : "f"(_tmem_load_2[4]), "f"(comp[4]));
                float _add_32;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(acc[4]), "f"(_sub_28_1));
                float _sub_33;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_33) : "f"(_add_32), "f"(acc[4]));
                float _sub_32;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32) : "f"(_sub_33), "f"(_sub_28_1));
                comp[4] = _sub_32;
                acc[4] = _add_32;
                float _sub_29_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_29_1) : "f"(_tmem_load_2[5]), "f"(comp[5]));
                float _add_33;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(acc[5]), "f"(_sub_29_1));
                float _sub_35;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35) : "f"(_add_33), "f"(acc[5]));
                float _sub_34;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34) : "f"(_sub_35), "f"(_sub_29_1));
                comp[5] = _sub_34;
                acc[5] = _add_33;
                float _sub_30_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_30_1) : "f"(_tmem_load_2[6]), "f"(comp[6]));
                float _add_34;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(acc[6]), "f"(_sub_30_1));
                float _sub_37;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_37) : "f"(_add_34), "f"(acc[6]));
                float _sub_36;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36) : "f"(_sub_37), "f"(_sub_30_1));
                comp[6] = _sub_36;
                acc[6] = _add_34;
                float _sub_31_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31_1) : "f"(_tmem_load_2[7]), "f"(comp[7]));
                float _add_35;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35) : "f"(acc[7]), "f"(_sub_31_1));
                float _sub_39;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_39) : "f"(_add_35), "f"(acc[7]));
                float _sub_38;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_38) : "f"(_sub_39), "f"(_sub_31_1));
                comp[7] = _sub_38;
                acc[7] = _add_35;
                float _sub_32_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32_1) : "f"(_tmem_load_3[0]), "f"(comp[8]));
                float _add_36;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(acc[8]), "f"(_sub_32_1));
                float _sub_41;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41) : "f"(_add_36), "f"(acc[8]));
                float _sub_40;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_40) : "f"(_sub_41), "f"(_sub_32_1));
                comp[8] = _sub_40;
                acc[8] = _add_36;
                float _sub_33_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_33_1) : "f"(_tmem_load_3[1]), "f"(comp[9]));
                float _add_37;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(acc[9]), "f"(_sub_33_1));
                float _sub_43;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43) : "f"(_add_37), "f"(acc[9]));
                float _sub_42;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_42) : "f"(_sub_43), "f"(_sub_33_1));
                comp[9] = _sub_42;
                acc[9] = _add_37;
                float _sub_34_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34_1) : "f"(_tmem_load_3[2]), "f"(comp[10]));
                float _add_38;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(acc[10]), "f"(_sub_34_1));
                float _sub_45;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_45) : "f"(_add_38), "f"(acc[10]));
                float _sub_44;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44) : "f"(_sub_45), "f"(_sub_34_1));
                comp[10] = _sub_44;
                acc[10] = _add_38;
                float _sub_35_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35_1) : "f"(_tmem_load_3[3]), "f"(comp[11]));
                float _add_39;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(acc[11]), "f"(_sub_35_1));
                float _sub_47;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_47) : "f"(_add_39), "f"(acc[11]));
                float _sub_46;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_46) : "f"(_sub_47), "f"(_sub_35_1));
                comp[11] = _sub_46;
                acc[11] = _add_39;
                unsigned int row_6 = first_row_2;
                float values[2];
                float total;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total) : "f"(acc[0]), "f"(comp[0]));
                values[0] = ((_fast_div_q_8 * 64 + row_6 < num_tokens) ? total : 0.0f);
                float total_0;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_0) : "f"(acc[1]), "f"(comp[1]));
                values[1] = ((_fast_div_q_8 * 64 + row_6 < num_tokens) ? total_0 : 0.0f);
                {
                    float2 _v2 = make_float2(values[0 + 0], values[0 + 1]);
                    *reinterpret_cast<float2*>((scratch + ((unsigned long long)task_5 * 1536 + (unsigned long long)(row_6 * 24) + (unsigned long long)(lane_in_warp % 4 * 2))) + 0) = _v2;
                }
                unsigned int row_1_1 = first_row_2 + 8;
                float values_2[2];
                float total_3;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_3) : "f"(acc[2]), "f"(comp[2]));
                values_2[0] = ((_fast_div_q_8 * 64 + row_1_1 < num_tokens) ? total_3 : 0.0f);
                float total_4;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_4) : "f"(acc[3]), "f"(comp[3]));
                values_2[1] = ((_fast_div_q_8 * 64 + row_1_1 < num_tokens) ? total_4 : 0.0f);
                {
                    float2 _v2 = make_float2(values_2[0 + 0], values_2[0 + 1]);
                    *reinterpret_cast<float2*>((scratch + ((unsigned long long)task_5 * 1536 + (unsigned long long)(row_1_1 * 24) + (unsigned long long)(lane_in_warp % 4 * 2))) + 0) = _v2;
                }
                unsigned int row_5_1 = first_row_2;
                float values_6[2];
                float total_7;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_7) : "f"(acc[4]), "f"(comp[4]));
                values_6[0] = ((_fast_div_q_8 * 64 + row_5_1 < num_tokens) ? total_7 : 0.0f);
                float total_8;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_8) : "f"(acc[5]), "f"(comp[5]));
                values_6[1] = ((_fast_div_q_8 * 64 + row_5_1 < num_tokens) ? total_8 : 0.0f);
                {
                    float2 _v2 = make_float2(values_6[0 + 0], values_6[0 + 1]);
                    *reinterpret_cast<float2*>((scratch + ((unsigned long long)task_5 * 1536 + (unsigned long long)(row_5_1 * 24) + 8 + (unsigned long long)(lane_in_warp % 4 * 2))) + 0) = _v2;
                }
                unsigned int row_9 = first_row_2 + 8;
                float values_10[2];
                float total_11;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_11) : "f"(acc[6]), "f"(comp[6]));
                values_10[0] = ((_fast_div_q_8 * 64 + row_9 < num_tokens) ? total_11 : 0.0f);
                float total_12;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_12) : "f"(acc[7]), "f"(comp[7]));
                values_10[1] = ((_fast_div_q_8 * 64 + row_9 < num_tokens) ? total_12 : 0.0f);
                {
                    float2 _v2 = make_float2(values_10[0 + 0], values_10[0 + 1]);
                    *reinterpret_cast<float2*>((scratch + ((unsigned long long)task_5 * 1536 + (unsigned long long)(row_9 * 24) + 8 + (unsigned long long)(lane_in_warp % 4 * 2))) + 0) = _v2;
                }
                unsigned int row_13 = first_row_2;
                float values_14[2];
                float total_15;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_15) : "f"(acc[8]), "f"(comp[8]));
                values_14[0] = ((_fast_div_q_8 * 64 + row_13 < num_tokens) ? total_15 : 0.0f);
                float total_16;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_16) : "f"(acc[9]), "f"(comp[9]));
                values_14[1] = ((_fast_div_q_8 * 64 + row_13 < num_tokens) ? total_16 : 0.0f);
                {
                    float2 _v2 = make_float2(values_14[0 + 0], values_14[0 + 1]);
                    *reinterpret_cast<float2*>((scratch + ((unsigned long long)task_5 * 1536 + (unsigned long long)(row_13 * 24) + 16 + (unsigned long long)(lane_in_warp % 4 * 2))) + 0) = _v2;
                }
                unsigned int row_17 = first_row_2 + 8;
                float values_18[2];
                float total_19;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_19) : "f"(acc[10]), "f"(comp[10]));
                values_18[0] = ((_fast_div_q_8 * 64 + row_17 < num_tokens) ? total_19 : 0.0f);
                float total_20;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(total_20) : "f"(acc[11]), "f"(comp[11]));
                values_18[1] = ((_fast_div_q_8 * 64 + row_17 < num_tokens) ? total_20 : 0.0f);
                {
                    float2 _v2 = make_float2(values_18[0 + 0], values_18[0 + 1]);
                    *reinterpret_cast<float2*>((scratch + ((unsigned long long)task_5 * 1536 + (unsigned long long)(row_17 * 24) + 16 + (unsigned long long)(lane_in_warp % 4 * 2))) + 0) = _v2;
                }
                unsigned int stats_slot = 0;
                unsigned int stats_phase_2 = seq_1 % 2;
                mbarrier_wait(full_stats_addr + (stats_slot) * 8, stats_phase_2);
                if (lane_in_warp < 16) {
                    unsigned int row_0_2 = warp % 4 * 16 + lane_in_warp;
                    float _add_48;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(hc_sums[row_0_2]), "f"(hc_sums[64 + row_0_2]));
                    scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)task_5 * 64 + (unsigned long long)row_0_2] = _add_48;
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(empty_stats_addr);
                    mbarrier_arrive(mix_arrival_addr);
                }
                if (warp % 4 == 0) {
                    unsigned int mix_slot = 0;
                    unsigned int mix_phase = seq_1 % 2;
                    mbarrier_wait(mix_arrival_addr + (mix_slot) * 8, mix_phase);
                    if (elect_sync()) {
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
                        asm volatile("red.async.release.gpu.global.add.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(split_barriers + ((16384 + _fast_div_q_8) * 16))), "l"(static_cast<unsigned long long>(1)) : "memory");
                        #elif defined(__CUDA_ARCH__)
                        #error "GlobalRedAsyncReleaseAdd requires SM100 or newer"
                        #endif
                    }
                }
                seq_1 += 1;
            }
        }
    }
    // ---- Role: mix ----
    if (warp >= 16 && warp <= 19) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // mix_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            if (warp == 17) {
                if (elect_sync()) {
                    for (unsigned int mb = (unsigned int)bid; mb < (num_tokens + 63) / 64; mb += num_ctas) {
                        asm volatile("st.release.gpu.global.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(split_barriers + ((16384 + mb) * 16))), "l"(static_cast<unsigned long long>((launch_epochs[bid] + 1) * 128)) : "memory");
                    }
                }
            }
            asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
            #pragma unroll 1
            for (unsigned int token = (unsigned int)bid * 4 + warp % 4; token < num_tokens; token += num_ctas * 4) {
                unsigned int mb_1 = token / 64;
                unsigned long long target = (launch_epochs[bid] + 1) * 128 + (unsigned long long)num_splits;
                {
                unsigned long long _acquire_observed;
                do {
                asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_acquire_observed) : "l"(reinterpret_cast<unsigned long long*>(split_barriers + ((16384 + mb_1) * 16))) : "memory");
                } while (static_cast<unsigned long long>(_acquire_observed - static_cast<unsigned long long>(target)) >= static_cast<unsigned long long>(1));
                }
                unsigned int row_7 = token % 64;
                unsigned int first_task = mb_1 * num_splits;
                float sums[25];
                unsigned int split = lane_in_warp * 2;
                unsigned int safe_split = ((split < num_splits) ? split : 0);
                unsigned long long record = ((unsigned long long)first_task + (unsigned long long)safe_split) * 1536 + (unsigned long long)(row_7 * 24);
                float _vec_load_0[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(scratch + record + 0);
                    _vec_load_0[0 + 0] = _v4.x;
                    _vec_load_0[0 + 1] = _v4.y;
                    _vec_load_0[0 + 2] = _v4.z;
                    _vec_load_0[0 + 3] = _v4.w;
                }
                float value_2 = ((split < num_splits) ? _vec_load_0[0] : 0.0f);
                sums[0] = value_2;
                float value_0 = ((split < num_splits) ? _vec_load_0[1] : 0.0f);
                sums[1] = value_0;
                float value_1_1 = ((split < num_splits) ? _vec_load_0[2] : 0.0f);
                sums[2] = value_1_1;
                float value_2_1 = ((split < num_splits) ? _vec_load_0[3] : 0.0f);
                sums[3] = value_2_1;
                float _vec_load_1[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record + 4) + 0);
                    _vec_load_1[0 + 0] = _v4.x;
                    _vec_load_1[0 + 1] = _v4.y;
                    _vec_load_1[0 + 2] = _v4.z;
                    _vec_load_1[0 + 3] = _v4.w;
                }
                float value_3 = ((split < num_splits) ? _vec_load_1[0] : 0.0f);
                sums[4] = value_3;
                float value_4 = ((split < num_splits) ? _vec_load_1[1] : 0.0f);
                sums[5] = value_4;
                float value_5 = ((split < num_splits) ? _vec_load_1[2] : 0.0f);
                sums[6] = value_5;
                float value_6 = ((split < num_splits) ? _vec_load_1[3] : 0.0f);
                sums[7] = value_6;
                float _vec_load_2[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record + 8) + 0);
                    _vec_load_2[0 + 0] = _v4.x;
                    _vec_load_2[0 + 1] = _v4.y;
                    _vec_load_2[0 + 2] = _v4.z;
                    _vec_load_2[0 + 3] = _v4.w;
                }
                float value_7 = ((split < num_splits) ? _vec_load_2[0] : 0.0f);
                sums[8] = value_7;
                float value_8_2 = ((split < num_splits) ? _vec_load_2[1] : 0.0f);
                sums[9] = value_8_2;
                float value_9 = ((split < num_splits) ? _vec_load_2[2] : 0.0f);
                sums[10] = value_9;
                float value_10 = ((split < num_splits) ? _vec_load_2[3] : 0.0f);
                sums[11] = value_10;
                float _vec_load_3[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record + 12) + 0);
                    _vec_load_3[0 + 0] = _v4.x;
                    _vec_load_3[0 + 1] = _v4.y;
                    _vec_load_3[0 + 2] = _v4.z;
                    _vec_load_3[0 + 3] = _v4.w;
                }
                float value_11 = ((split < num_splits) ? _vec_load_3[0] : 0.0f);
                sums[12] = value_11;
                float value_12 = ((split < num_splits) ? _vec_load_3[1] : 0.0f);
                sums[13] = value_12;
                float value_13 = ((split < num_splits) ? _vec_load_3[2] : 0.0f);
                sums[14] = value_13;
                float value_14 = ((split < num_splits) ? _vec_load_3[3] : 0.0f);
                sums[15] = value_14;
                float _vec_load_4[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record + 16) + 0);
                    _vec_load_4[0 + 0] = _v4.x;
                    _vec_load_4[0 + 1] = _v4.y;
                    _vec_load_4[0 + 2] = _v4.z;
                    _vec_load_4[0 + 3] = _v4.w;
                }
                float value_15 = ((split < num_splits) ? _vec_load_4[0] : 0.0f);
                sums[16] = value_15;
                float value_16 = ((split < num_splits) ? _vec_load_4[1] : 0.0f);
                sums[17] = value_16;
                float value_17_2 = ((split < num_splits) ? _vec_load_4[2] : 0.0f);
                sums[18] = value_17_2;
                float value_18 = ((split < num_splits) ? _vec_load_4[3] : 0.0f);
                sums[19] = value_18;
                float _vec_load_5[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record + 20) + 0);
                    _vec_load_5[0 + 0] = _v4.x;
                    _vec_load_5[0 + 1] = _v4.y;
                    _vec_load_5[0 + 2] = _v4.z;
                    _vec_load_5[0 + 3] = _v4.w;
                }
                float value_19 = ((split < num_splits) ? _vec_load_5[0] : 0.0f);
                sums[20] = value_19;
                float value_20 = ((split < num_splits) ? _vec_load_5[1] : 0.0f);
                sums[21] = value_20;
                float value_21 = ((split < num_splits) ? _vec_load_5[2] : 0.0f);
                sums[22] = value_21;
                float value_22 = ((split < num_splits) ? _vec_load_5[3] : 0.0f);
                sums[23] = value_22;
                float q_word = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task * 64 + (unsigned long long)(safe_split * 64) + (unsigned long long)row_7];
                float q_value = ((split < num_splits) ? q_word : 0.0f);
                sums[24] = q_value;
                unsigned int split_23 = lane_in_warp * 2 + 1;
                unsigned int safe_split_24 = ((split_23 < num_splits) ? split_23 : 0);
                unsigned long long record_25 = ((unsigned long long)first_task + (unsigned long long)safe_split_24) * 1536 + (unsigned long long)(row_7 * 24);
                float _vec_load_6[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_25 + 0);
                    _vec_load_6[0 + 0] = _v4.x;
                    _vec_load_6[0 + 1] = _v4.y;
                    _vec_load_6[0 + 2] = _v4.z;
                    _vec_load_6[0 + 3] = _v4.w;
                }
                float value_26_2 = ((split_23 < num_splits) ? _vec_load_6[0] : 0.0f);
                float _add_7_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7_1) : "f"(sums[0]), "f"(value_26_2));
                sums[0] = _add_7_1;
                float value_27 = ((split_23 < num_splits) ? _vec_load_6[1] : 0.0f);
                float _add_8_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8_1) : "f"(sums[1]), "f"(value_27));
                sums[1] = _add_8_1;
                float value_28 = ((split_23 < num_splits) ? _vec_load_6[2] : 0.0f);
                float _add_9_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9_1) : "f"(sums[2]), "f"(value_28));
                sums[2] = _add_9_1;
                float value_29 = ((split_23 < num_splits) ? _vec_load_6[3] : 0.0f);
                float _add_10_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_10_1) : "f"(sums[3]), "f"(value_29));
                sums[3] = _add_10_1;
                float _vec_load_7[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25 + 4) + 0);
                    _vec_load_7[0 + 0] = _v4.x;
                    _vec_load_7[0 + 1] = _v4.y;
                    _vec_load_7[0 + 2] = _v4.z;
                    _vec_load_7[0 + 3] = _v4.w;
                }
                float value_30 = ((split_23 < num_splits) ? _vec_load_7[0] : 0.0f);
                float _add_12_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12_1) : "f"(sums[4]), "f"(value_30));
                sums[4] = _add_12_1;
                float value_31 = ((split_23 < num_splits) ? _vec_load_7[1] : 0.0f);
                float _add_13_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13_1) : "f"(sums[5]), "f"(value_31));
                sums[5] = _add_13_1;
                float value_32 = ((split_23 < num_splits) ? _vec_load_7[2] : 0.0f);
                float _add_14_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14_1) : "f"(sums[6]), "f"(value_32));
                sums[6] = _add_14_1;
                float value_33 = ((split_23 < num_splits) ? _vec_load_7[3] : 0.0f);
                float _add_15_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15_1) : "f"(sums[7]), "f"(value_33));
                sums[7] = _add_15_1;
                float _vec_load_8[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25 + 8) + 0);
                    _vec_load_8[0 + 0] = _v4.x;
                    _vec_load_8[0 + 1] = _v4.y;
                    _vec_load_8[0 + 2] = _v4.z;
                    _vec_load_8[0 + 3] = _v4.w;
                }
                float value_34 = ((split_23 < num_splits) ? _vec_load_8[0] : 0.0f);
                float _add_17_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17_1) : "f"(sums[8]), "f"(value_34));
                sums[8] = _add_17_1;
                float value_35 = ((split_23 < num_splits) ? _vec_load_8[1] : 0.0f);
                float _add_18_3;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18_3) : "f"(sums[9]), "f"(value_35));
                sums[9] = _add_18_3;
                float value_36 = ((split_23 < num_splits) ? _vec_load_8[2] : 0.0f);
                float _add_19_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19_1) : "f"(sums[10]), "f"(value_36));
                sums[10] = _add_19_1;
                float value_37 = ((split_23 < num_splits) ? _vec_load_8[3] : 0.0f);
                float _add_20_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20_1) : "f"(sums[11]), "f"(value_37));
                sums[11] = _add_20_1;
                float _vec_load_9[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25 + 12) + 0);
                    _vec_load_9[0 + 0] = _v4.x;
                    _vec_load_9[0 + 1] = _v4.y;
                    _vec_load_9[0 + 2] = _v4.z;
                    _vec_load_9[0 + 3] = _v4.w;
                }
                float value_38 = ((split_23 < num_splits) ? _vec_load_9[0] : 0.0f);
                float _add_22_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22_1) : "f"(sums[12]), "f"(value_38));
                sums[12] = _add_22_1;
                float value_39 = ((split_23 < num_splits) ? _vec_load_9[1] : 0.0f);
                float _add_23_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23_1) : "f"(sums[13]), "f"(value_39));
                sums[13] = _add_23_1;
                float value_40 = ((split_23 < num_splits) ? _vec_load_9[2] : 0.0f);
                float _add_24_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24_1) : "f"(sums[14]), "f"(value_40));
                sums[14] = _add_24_1;
                float value_41 = ((split_23 < num_splits) ? _vec_load_9[3] : 0.0f);
                float _add_25_3;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25_3) : "f"(sums[15]), "f"(value_41));
                sums[15] = _add_25_3;
                float _vec_load_10[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25 + 16) + 0);
                    _vec_load_10[0 + 0] = _v4.x;
                    _vec_load_10[0 + 1] = _v4.y;
                    _vec_load_10[0 + 2] = _v4.z;
                    _vec_load_10[0 + 3] = _v4.w;
                }
                float value_42 = ((split_23 < num_splits) ? _vec_load_10[0] : 0.0f);
                float _add_27_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27_1) : "f"(sums[16]), "f"(value_42));
                sums[16] = _add_27_1;
                float value_43 = ((split_23 < num_splits) ? _vec_load_10[1] : 0.0f);
                float _add_28_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28_1) : "f"(sums[17]), "f"(value_43));
                sums[17] = _add_28_1;
                float value_44 = ((split_23 < num_splits) ? _vec_load_10[2] : 0.0f);
                float _add_29_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29_1) : "f"(sums[18]), "f"(value_44));
                sums[18] = _add_29_1;
                float value_45_2 = ((split_23 < num_splits) ? _vec_load_10[3] : 0.0f);
                float _add_30_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30_1) : "f"(sums[19]), "f"(value_45_2));
                sums[19] = _add_30_1;
                float _vec_load_11[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(scratch + (record_25 + 20) + 0);
                    _vec_load_11[0 + 0] = _v4.x;
                    _vec_load_11[0 + 1] = _v4.y;
                    _vec_load_11[0 + 2] = _v4.z;
                    _vec_load_11[0 + 3] = _v4.w;
                }
                float value_46 = ((split_23 < num_splits) ? _vec_load_11[0] : 0.0f);
                float _add_32_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32_1) : "f"(sums[20]), "f"(value_46));
                sums[20] = _add_32_1;
                float value_47 = ((split_23 < num_splits) ? _vec_load_11[1] : 0.0f);
                float _add_33_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33_1) : "f"(sums[21]), "f"(value_47));
                sums[21] = _add_33_1;
                float value_48 = ((split_23 < num_splits) ? _vec_load_11[2] : 0.0f);
                float _add_34_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34_1) : "f"(sums[22]), "f"(value_48));
                sums[22] = _add_34_1;
                float value_49 = ((split_23 < num_splits) ? _vec_load_11[3] : 0.0f);
                float _add_35_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35_1) : "f"(sums[23]), "f"(value_49));
                sums[23] = _add_35_1;
                float q_word_50 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task * 64 + (unsigned long long)(safe_split_24 * 64) + (unsigned long long)row_7];
                float q_value_51 = ((split_23 < num_splits) ? q_word_50 : 0.0f);
                float _add_36_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36_1) : "f"(sums[24]), "f"(q_value_51));
                sums[24] = _add_36_1;
                float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, sums[0], 1);
                float _add_37_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37_1) : "f"(sums[0]), "f"(_shfl_xor_sync_38));
                sums[0] = _add_37_1;
                float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, sums[1], 1);
                float _add_39_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39_1) : "f"(sums[1]), "f"(_shfl_xor_sync_40));
                sums[1] = _add_39_1;
                float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, sums[2], 1);
                float _add_41;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(sums[2]), "f"(_shfl_xor_sync_42));
                sums[2] = _add_41;
                float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, sums[3], 1);
                float _add_43;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(sums[3]), "f"(_shfl_xor_sync_44));
                sums[3] = _add_43;
                float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, sums[4], 1);
                float _add_45;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(sums[4]), "f"(_shfl_xor_sync_46));
                sums[4] = _add_45;
                float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, sums[5], 1);
                float _add_47;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(sums[5]), "f"(_shfl_xor_sync_48));
                sums[5] = _add_47;
                float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, sums[6], 1);
                float _add_49;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(sums[6]), "f"(_shfl_xor_sync_50));
                sums[6] = _add_49;
                float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, sums[7], 1);
                float _add_51;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(sums[7]), "f"(_shfl_xor_sync_52));
                sums[7] = _add_51;
                float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, sums[8], 1);
                float _add_53;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(sums[8]), "f"(_shfl_xor_sync_54));
                sums[8] = _add_53;
                float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, sums[9], 1);
                float _add_55;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_55) : "f"(sums[9]), "f"(_shfl_xor_sync_56));
                sums[9] = _add_55;
                float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, sums[10], 1);
                float _add_57;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(sums[10]), "f"(_shfl_xor_sync_58));
                sums[10] = _add_57;
                float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, sums[11], 1);
                float _add_59;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(sums[11]), "f"(_shfl_xor_sync_60));
                sums[11] = _add_59;
                float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, sums[12], 1);
                float _add_61;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_61) : "f"(sums[12]), "f"(_shfl_xor_sync_62));
                sums[12] = _add_61;
                float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, sums[13], 1);
                float _add_63;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_63) : "f"(sums[13]), "f"(_shfl_xor_sync_64));
                sums[13] = _add_63;
                float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, sums[14], 1);
                float _add_65;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_65) : "f"(sums[14]), "f"(_shfl_xor_sync_66));
                sums[14] = _add_65;
                float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, sums[15], 1);
                float _add_67;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_67) : "f"(sums[15]), "f"(_shfl_xor_sync_68));
                sums[15] = _add_67;
                float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, sums[16], 1);
                float _add_69;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(sums[16]), "f"(_shfl_xor_sync_70));
                sums[16] = _add_69;
                float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, sums[17], 1);
                float _add_71;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_71) : "f"(sums[17]), "f"(_shfl_xor_sync_72));
                sums[17] = _add_71;
                float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, sums[18], 1);
                float _add_73;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(sums[18]), "f"(_shfl_xor_sync_74));
                sums[18] = _add_73;
                float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, sums[19], 1);
                float _add_75;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(sums[19]), "f"(_shfl_xor_sync_76));
                sums[19] = _add_75;
                float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, sums[20], 1);
                float _add_77;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(sums[20]), "f"(_shfl_xor_sync_78));
                sums[20] = _add_77;
                float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, sums[21], 1);
                float _add_79;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(sums[21]), "f"(_shfl_xor_sync_80));
                sums[21] = _add_79;
                float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, sums[22], 1);
                float _add_81;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_81) : "f"(sums[22]), "f"(_shfl_xor_sync_82));
                sums[22] = _add_81;
                float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, sums[23], 1);
                float _add_83;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(sums[23]), "f"(_shfl_xor_sync_84));
                sums[23] = _add_83;
                float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, sums[24], 1);
                float _add_85;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_85) : "f"(sums[24]), "f"(_shfl_xor_sync_86));
                sums[24] = _add_85;
                float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, sums[0], 2);
                float _add_87;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(sums[0]), "f"(_shfl_xor_sync_88));
                sums[0] = _add_87;
                float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, sums[1], 2);
                float _add_89;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_89) : "f"(sums[1]), "f"(_shfl_xor_sync_90));
                sums[1] = _add_89;
                float _shfl_xor_sync_92 = __shfl_xor_sync(0xFFFFFFFF, sums[2], 2);
                float _add_91;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_91) : "f"(sums[2]), "f"(_shfl_xor_sync_92));
                sums[2] = _add_91;
                float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, sums[3], 2);
                float _add_93;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(sums[3]), "f"(_shfl_xor_sync_94));
                sums[3] = _add_93;
                float _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, sums[4], 2);
                float _add_95;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_95) : "f"(sums[4]), "f"(_shfl_xor_sync_96));
                sums[4] = _add_95;
                float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, sums[5], 2);
                float _add_97;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_97) : "f"(sums[5]), "f"(_shfl_xor_sync_98));
                sums[5] = _add_97;
                float _shfl_xor_sync_100 = __shfl_xor_sync(0xFFFFFFFF, sums[6], 2);
                float _add_99;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_99) : "f"(sums[6]), "f"(_shfl_xor_sync_100));
                sums[6] = _add_99;
                float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, sums[7], 2);
                float _add_101;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_101) : "f"(sums[7]), "f"(_shfl_xor_sync_102));
                sums[7] = _add_101;
                float _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, sums[8], 2);
                float _add_103;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(sums[8]), "f"(_shfl_xor_sync_104));
                sums[8] = _add_103;
                float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, sums[9], 2);
                float _add_105;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_105) : "f"(sums[9]), "f"(_shfl_xor_sync_106));
                sums[9] = _add_105;
                float _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, sums[10], 2);
                float _add_107;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_107) : "f"(sums[10]), "f"(_shfl_xor_sync_108));
                sums[10] = _add_107;
                float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, sums[11], 2);
                float _add_109;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_109) : "f"(sums[11]), "f"(_shfl_xor_sync_110));
                sums[11] = _add_109;
                float _shfl_xor_sync_112 = __shfl_xor_sync(0xFFFFFFFF, sums[12], 2);
                float _add_111;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_111) : "f"(sums[12]), "f"(_shfl_xor_sync_112));
                sums[12] = _add_111;
                float _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, sums[13], 2);
                float _add_113;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(sums[13]), "f"(_shfl_xor_sync_114));
                sums[13] = _add_113;
                float _shfl_xor_sync_116 = __shfl_xor_sync(0xFFFFFFFF, sums[14], 2);
                float _add_115;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_115) : "f"(sums[14]), "f"(_shfl_xor_sync_116));
                sums[14] = _add_115;
                float _shfl_xor_sync_118 = __shfl_xor_sync(0xFFFFFFFF, sums[15], 2);
                float _add_117;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_117) : "f"(sums[15]), "f"(_shfl_xor_sync_118));
                sums[15] = _add_117;
                float _shfl_xor_sync_120 = __shfl_xor_sync(0xFFFFFFFF, sums[16], 2);
                float _add_119;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_119) : "f"(sums[16]), "f"(_shfl_xor_sync_120));
                sums[16] = _add_119;
                float _shfl_xor_sync_122 = __shfl_xor_sync(0xFFFFFFFF, sums[17], 2);
                float _add_121;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_121) : "f"(sums[17]), "f"(_shfl_xor_sync_122));
                sums[17] = _add_121;
                float _shfl_xor_sync_124 = __shfl_xor_sync(0xFFFFFFFF, sums[18], 2);
                float _add_123;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_123) : "f"(sums[18]), "f"(_shfl_xor_sync_124));
                sums[18] = _add_123;
                float _shfl_xor_sync_126 = __shfl_xor_sync(0xFFFFFFFF, sums[19], 2);
                float _add_125;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_125) : "f"(sums[19]), "f"(_shfl_xor_sync_126));
                sums[19] = _add_125;
                float _shfl_xor_sync_128 = __shfl_xor_sync(0xFFFFFFFF, sums[20], 2);
                float _add_127;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_127) : "f"(sums[20]), "f"(_shfl_xor_sync_128));
                sums[20] = _add_127;
                float _shfl_xor_sync_130 = __shfl_xor_sync(0xFFFFFFFF, sums[21], 2);
                float _add_129;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_129) : "f"(sums[21]), "f"(_shfl_xor_sync_130));
                sums[21] = _add_129;
                float _shfl_xor_sync_132 = __shfl_xor_sync(0xFFFFFFFF, sums[22], 2);
                float _add_131;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_131) : "f"(sums[22]), "f"(_shfl_xor_sync_132));
                sums[22] = _add_131;
                float _shfl_xor_sync_134 = __shfl_xor_sync(0xFFFFFFFF, sums[23], 2);
                float _add_133;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_133) : "f"(sums[23]), "f"(_shfl_xor_sync_134));
                sums[23] = _add_133;
                float _shfl_xor_sync_136 = __shfl_xor_sync(0xFFFFFFFF, sums[24], 2);
                float _add_135;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_135) : "f"(sums[24]), "f"(_shfl_xor_sync_136));
                sums[24] = _add_135;
                float _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, sums[0], 4);
                float _add_137;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_137) : "f"(sums[0]), "f"(_shfl_xor_sync_138));
                sums[0] = _add_137;
                float _shfl_xor_sync_140 = __shfl_xor_sync(0xFFFFFFFF, sums[1], 4);
                float _add_139;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_139) : "f"(sums[1]), "f"(_shfl_xor_sync_140));
                sums[1] = _add_139;
                float _shfl_xor_sync_142 = __shfl_xor_sync(0xFFFFFFFF, sums[2], 4);
                float _add_141;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_141) : "f"(sums[2]), "f"(_shfl_xor_sync_142));
                sums[2] = _add_141;
                float _shfl_xor_sync_144 = __shfl_xor_sync(0xFFFFFFFF, sums[3], 4);
                float _add_143;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_143) : "f"(sums[3]), "f"(_shfl_xor_sync_144));
                sums[3] = _add_143;
                float _shfl_xor_sync_146 = __shfl_xor_sync(0xFFFFFFFF, sums[4], 4);
                float _add_145;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_145) : "f"(sums[4]), "f"(_shfl_xor_sync_146));
                sums[4] = _add_145;
                float _shfl_xor_sync_148 = __shfl_xor_sync(0xFFFFFFFF, sums[5], 4);
                float _add_147;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_147) : "f"(sums[5]), "f"(_shfl_xor_sync_148));
                sums[5] = _add_147;
                float _shfl_xor_sync_150 = __shfl_xor_sync(0xFFFFFFFF, sums[6], 4);
                float _add_149;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_149) : "f"(sums[6]), "f"(_shfl_xor_sync_150));
                sums[6] = _add_149;
                float _shfl_xor_sync_152 = __shfl_xor_sync(0xFFFFFFFF, sums[7], 4);
                float _add_151;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_151) : "f"(sums[7]), "f"(_shfl_xor_sync_152));
                sums[7] = _add_151;
                float _shfl_xor_sync_154 = __shfl_xor_sync(0xFFFFFFFF, sums[8], 4);
                float _add_153;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_153) : "f"(sums[8]), "f"(_shfl_xor_sync_154));
                sums[8] = _add_153;
                float _shfl_xor_sync_156 = __shfl_xor_sync(0xFFFFFFFF, sums[9], 4);
                float _add_155;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_155) : "f"(sums[9]), "f"(_shfl_xor_sync_156));
                sums[9] = _add_155;
                float _shfl_xor_sync_158 = __shfl_xor_sync(0xFFFFFFFF, sums[10], 4);
                float _add_157;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_157) : "f"(sums[10]), "f"(_shfl_xor_sync_158));
                sums[10] = _add_157;
                float _shfl_xor_sync_160 = __shfl_xor_sync(0xFFFFFFFF, sums[11], 4);
                float _add_159;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_159) : "f"(sums[11]), "f"(_shfl_xor_sync_160));
                sums[11] = _add_159;
                float _shfl_xor_sync_162 = __shfl_xor_sync(0xFFFFFFFF, sums[12], 4);
                float _add_161;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_161) : "f"(sums[12]), "f"(_shfl_xor_sync_162));
                sums[12] = _add_161;
                float _shfl_xor_sync_164 = __shfl_xor_sync(0xFFFFFFFF, sums[13], 4);
                float _add_163;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_163) : "f"(sums[13]), "f"(_shfl_xor_sync_164));
                sums[13] = _add_163;
                float _shfl_xor_sync_166 = __shfl_xor_sync(0xFFFFFFFF, sums[14], 4);
                float _add_165;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_165) : "f"(sums[14]), "f"(_shfl_xor_sync_166));
                sums[14] = _add_165;
                float _shfl_xor_sync_168 = __shfl_xor_sync(0xFFFFFFFF, sums[15], 4);
                float _add_167;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_167) : "f"(sums[15]), "f"(_shfl_xor_sync_168));
                sums[15] = _add_167;
                float _shfl_xor_sync_170 = __shfl_xor_sync(0xFFFFFFFF, sums[16], 4);
                float _add_169;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_169) : "f"(sums[16]), "f"(_shfl_xor_sync_170));
                sums[16] = _add_169;
                float _shfl_xor_sync_172 = __shfl_xor_sync(0xFFFFFFFF, sums[17], 4);
                float _add_171;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_171) : "f"(sums[17]), "f"(_shfl_xor_sync_172));
                sums[17] = _add_171;
                float _shfl_xor_sync_174 = __shfl_xor_sync(0xFFFFFFFF, sums[18], 4);
                float _add_173;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_173) : "f"(sums[18]), "f"(_shfl_xor_sync_174));
                sums[18] = _add_173;
                float _shfl_xor_sync_176 = __shfl_xor_sync(0xFFFFFFFF, sums[19], 4);
                float _add_175;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_175) : "f"(sums[19]), "f"(_shfl_xor_sync_176));
                sums[19] = _add_175;
                float _shfl_xor_sync_178 = __shfl_xor_sync(0xFFFFFFFF, sums[20], 4);
                float _add_177;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_177) : "f"(sums[20]), "f"(_shfl_xor_sync_178));
                sums[20] = _add_177;
                float _shfl_xor_sync_180 = __shfl_xor_sync(0xFFFFFFFF, sums[21], 4);
                float _add_179;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_179) : "f"(sums[21]), "f"(_shfl_xor_sync_180));
                sums[21] = _add_179;
                float _shfl_xor_sync_182 = __shfl_xor_sync(0xFFFFFFFF, sums[22], 4);
                float _add_181;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_181) : "f"(sums[22]), "f"(_shfl_xor_sync_182));
                sums[22] = _add_181;
                float _shfl_xor_sync_184 = __shfl_xor_sync(0xFFFFFFFF, sums[23], 4);
                float _add_183;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_183) : "f"(sums[23]), "f"(_shfl_xor_sync_184));
                sums[23] = _add_183;
                float _shfl_xor_sync_186 = __shfl_xor_sync(0xFFFFFFFF, sums[24], 4);
                float _add_185;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_185) : "f"(sums[24]), "f"(_shfl_xor_sync_186));
                sums[24] = _add_185;
                float _shfl_xor_sync_188 = __shfl_xor_sync(0xFFFFFFFF, sums[0], 8);
                float _add_187;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_187) : "f"(sums[0]), "f"(_shfl_xor_sync_188));
                sums[0] = _add_187;
                float _shfl_xor_sync_190 = __shfl_xor_sync(0xFFFFFFFF, sums[1], 8);
                float _add_189;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_189) : "f"(sums[1]), "f"(_shfl_xor_sync_190));
                sums[1] = _add_189;
                float _shfl_xor_sync_192 = __shfl_xor_sync(0xFFFFFFFF, sums[2], 8);
                float _add_191;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_191) : "f"(sums[2]), "f"(_shfl_xor_sync_192));
                sums[2] = _add_191;
                float _shfl_xor_sync_194 = __shfl_xor_sync(0xFFFFFFFF, sums[3], 8);
                float _add_193;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_193) : "f"(sums[3]), "f"(_shfl_xor_sync_194));
                sums[3] = _add_193;
                float _shfl_xor_sync_196 = __shfl_xor_sync(0xFFFFFFFF, sums[4], 8);
                float _add_195;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_195) : "f"(sums[4]), "f"(_shfl_xor_sync_196));
                sums[4] = _add_195;
                float _shfl_xor_sync_198 = __shfl_xor_sync(0xFFFFFFFF, sums[5], 8);
                float _add_197;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_197) : "f"(sums[5]), "f"(_shfl_xor_sync_198));
                sums[5] = _add_197;
                float _shfl_xor_sync_200 = __shfl_xor_sync(0xFFFFFFFF, sums[6], 8);
                float _add_199;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_199) : "f"(sums[6]), "f"(_shfl_xor_sync_200));
                sums[6] = _add_199;
                float _shfl_xor_sync_202 = __shfl_xor_sync(0xFFFFFFFF, sums[7], 8);
                float _add_201;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_201) : "f"(sums[7]), "f"(_shfl_xor_sync_202));
                sums[7] = _add_201;
                float _shfl_xor_sync_204 = __shfl_xor_sync(0xFFFFFFFF, sums[8], 8);
                float _add_203;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_203) : "f"(sums[8]), "f"(_shfl_xor_sync_204));
                sums[8] = _add_203;
                float _shfl_xor_sync_206 = __shfl_xor_sync(0xFFFFFFFF, sums[9], 8);
                float _add_205;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_205) : "f"(sums[9]), "f"(_shfl_xor_sync_206));
                sums[9] = _add_205;
                float _shfl_xor_sync_208 = __shfl_xor_sync(0xFFFFFFFF, sums[10], 8);
                float _add_207;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_207) : "f"(sums[10]), "f"(_shfl_xor_sync_208));
                sums[10] = _add_207;
                float _shfl_xor_sync_210 = __shfl_xor_sync(0xFFFFFFFF, sums[11], 8);
                float _add_209;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_209) : "f"(sums[11]), "f"(_shfl_xor_sync_210));
                sums[11] = _add_209;
                float _shfl_xor_sync_212 = __shfl_xor_sync(0xFFFFFFFF, sums[12], 8);
                float _add_211;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_211) : "f"(sums[12]), "f"(_shfl_xor_sync_212));
                sums[12] = _add_211;
                float _shfl_xor_sync_214 = __shfl_xor_sync(0xFFFFFFFF, sums[13], 8);
                float _add_213;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_213) : "f"(sums[13]), "f"(_shfl_xor_sync_214));
                sums[13] = _add_213;
                float _shfl_xor_sync_216 = __shfl_xor_sync(0xFFFFFFFF, sums[14], 8);
                float _add_215;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_215) : "f"(sums[14]), "f"(_shfl_xor_sync_216));
                sums[14] = _add_215;
                float _shfl_xor_sync_218 = __shfl_xor_sync(0xFFFFFFFF, sums[15], 8);
                float _add_217;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_217) : "f"(sums[15]), "f"(_shfl_xor_sync_218));
                sums[15] = _add_217;
                float _shfl_xor_sync_220 = __shfl_xor_sync(0xFFFFFFFF, sums[16], 8);
                float _add_219;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_219) : "f"(sums[16]), "f"(_shfl_xor_sync_220));
                sums[16] = _add_219;
                float _shfl_xor_sync_222 = __shfl_xor_sync(0xFFFFFFFF, sums[17], 8);
                float _add_221;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_221) : "f"(sums[17]), "f"(_shfl_xor_sync_222));
                sums[17] = _add_221;
                float _shfl_xor_sync_224 = __shfl_xor_sync(0xFFFFFFFF, sums[18], 8);
                float _add_223;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_223) : "f"(sums[18]), "f"(_shfl_xor_sync_224));
                sums[18] = _add_223;
                float _shfl_xor_sync_226 = __shfl_xor_sync(0xFFFFFFFF, sums[19], 8);
                float _add_225;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_225) : "f"(sums[19]), "f"(_shfl_xor_sync_226));
                sums[19] = _add_225;
                float _shfl_xor_sync_228 = __shfl_xor_sync(0xFFFFFFFF, sums[20], 8);
                float _add_227;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_227) : "f"(sums[20]), "f"(_shfl_xor_sync_228));
                sums[20] = _add_227;
                float _shfl_xor_sync_230 = __shfl_xor_sync(0xFFFFFFFF, sums[21], 8);
                float _add_229;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_229) : "f"(sums[21]), "f"(_shfl_xor_sync_230));
                sums[21] = _add_229;
                float _shfl_xor_sync_232 = __shfl_xor_sync(0xFFFFFFFF, sums[22], 8);
                float _add_231;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_231) : "f"(sums[22]), "f"(_shfl_xor_sync_232));
                sums[22] = _add_231;
                float _shfl_xor_sync_234 = __shfl_xor_sync(0xFFFFFFFF, sums[23], 8);
                float _add_233;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_233) : "f"(sums[23]), "f"(_shfl_xor_sync_234));
                sums[23] = _add_233;
                float _shfl_xor_sync_236 = __shfl_xor_sync(0xFFFFFFFF, sums[24], 8);
                float _add_235;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_235) : "f"(sums[24]), "f"(_shfl_xor_sync_236));
                sums[24] = _add_235;
                float _shfl_xor_sync_238 = __shfl_xor_sync(0xFFFFFFFF, sums[0], 16);
                float _add_237;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_237) : "f"(sums[0]), "f"(_shfl_xor_sync_238));
                sums[0] = _add_237;
                float _shfl_xor_sync_240 = __shfl_xor_sync(0xFFFFFFFF, sums[1], 16);
                float _add_239;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_239) : "f"(sums[1]), "f"(_shfl_xor_sync_240));
                sums[1] = _add_239;
                float _shfl_xor_sync_242 = __shfl_xor_sync(0xFFFFFFFF, sums[2], 16);
                float _add_241;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_241) : "f"(sums[2]), "f"(_shfl_xor_sync_242));
                sums[2] = _add_241;
                float _shfl_xor_sync_244 = __shfl_xor_sync(0xFFFFFFFF, sums[3], 16);
                float _add_243;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_243) : "f"(sums[3]), "f"(_shfl_xor_sync_244));
                sums[3] = _add_243;
                float _shfl_xor_sync_246 = __shfl_xor_sync(0xFFFFFFFF, sums[4], 16);
                float _add_245;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_245) : "f"(sums[4]), "f"(_shfl_xor_sync_246));
                sums[4] = _add_245;
                float _shfl_xor_sync_248 = __shfl_xor_sync(0xFFFFFFFF, sums[5], 16);
                float _add_247;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_247) : "f"(sums[5]), "f"(_shfl_xor_sync_248));
                sums[5] = _add_247;
                float _shfl_xor_sync_250 = __shfl_xor_sync(0xFFFFFFFF, sums[6], 16);
                float _add_249;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_249) : "f"(sums[6]), "f"(_shfl_xor_sync_250));
                sums[6] = _add_249;
                float _shfl_xor_sync_252 = __shfl_xor_sync(0xFFFFFFFF, sums[7], 16);
                float _add_251;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_251) : "f"(sums[7]), "f"(_shfl_xor_sync_252));
                sums[7] = _add_251;
                float _shfl_xor_sync_254 = __shfl_xor_sync(0xFFFFFFFF, sums[8], 16);
                float _add_253;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_253) : "f"(sums[8]), "f"(_shfl_xor_sync_254));
                sums[8] = _add_253;
                float _shfl_xor_sync_256 = __shfl_xor_sync(0xFFFFFFFF, sums[9], 16);
                float _add_255;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_255) : "f"(sums[9]), "f"(_shfl_xor_sync_256));
                sums[9] = _add_255;
                float _shfl_xor_sync_258 = __shfl_xor_sync(0xFFFFFFFF, sums[10], 16);
                float _add_257;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_257) : "f"(sums[10]), "f"(_shfl_xor_sync_258));
                sums[10] = _add_257;
                float _shfl_xor_sync_260 = __shfl_xor_sync(0xFFFFFFFF, sums[11], 16);
                float _add_259;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_259) : "f"(sums[11]), "f"(_shfl_xor_sync_260));
                sums[11] = _add_259;
                float _shfl_xor_sync_262 = __shfl_xor_sync(0xFFFFFFFF, sums[12], 16);
                float _add_261;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_261) : "f"(sums[12]), "f"(_shfl_xor_sync_262));
                sums[12] = _add_261;
                float _shfl_xor_sync_264 = __shfl_xor_sync(0xFFFFFFFF, sums[13], 16);
                float _add_263;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_263) : "f"(sums[13]), "f"(_shfl_xor_sync_264));
                sums[13] = _add_263;
                float _shfl_xor_sync_266 = __shfl_xor_sync(0xFFFFFFFF, sums[14], 16);
                float _add_265;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_265) : "f"(sums[14]), "f"(_shfl_xor_sync_266));
                sums[14] = _add_265;
                float _shfl_xor_sync_268 = __shfl_xor_sync(0xFFFFFFFF, sums[15], 16);
                float _add_267;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_267) : "f"(sums[15]), "f"(_shfl_xor_sync_268));
                sums[15] = _add_267;
                float _shfl_xor_sync_270 = __shfl_xor_sync(0xFFFFFFFF, sums[16], 16);
                float _add_269;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_269) : "f"(sums[16]), "f"(_shfl_xor_sync_270));
                sums[16] = _add_269;
                float _shfl_xor_sync_272 = __shfl_xor_sync(0xFFFFFFFF, sums[17], 16);
                float _add_271;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_271) : "f"(sums[17]), "f"(_shfl_xor_sync_272));
                sums[17] = _add_271;
                float _shfl_xor_sync_274 = __shfl_xor_sync(0xFFFFFFFF, sums[18], 16);
                float _add_273;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_273) : "f"(sums[18]), "f"(_shfl_xor_sync_274));
                sums[18] = _add_273;
                float _shfl_xor_sync_276 = __shfl_xor_sync(0xFFFFFFFF, sums[19], 16);
                float _add_275;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_275) : "f"(sums[19]), "f"(_shfl_xor_sync_276));
                sums[19] = _add_275;
                float _shfl_xor_sync_278 = __shfl_xor_sync(0xFFFFFFFF, sums[20], 16);
                float _add_277;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_277) : "f"(sums[20]), "f"(_shfl_xor_sync_278));
                sums[20] = _add_277;
                float _shfl_xor_sync_280 = __shfl_xor_sync(0xFFFFFFFF, sums[21], 16);
                float _add_279;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_279) : "f"(sums[21]), "f"(_shfl_xor_sync_280));
                sums[21] = _add_279;
                float _shfl_xor_sync_282 = __shfl_xor_sync(0xFFFFFFFF, sums[22], 16);
                float _add_281;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_281) : "f"(sums[22]), "f"(_shfl_xor_sync_282));
                sums[22] = _add_281;
                float _shfl_xor_sync_284 = __shfl_xor_sync(0xFFFFFFFF, sums[23], 16);
                float _add_283;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_283) : "f"(sums[23]), "f"(_shfl_xor_sync_284));
                sums[23] = _add_283;
                float _shfl_xor_sync_286 = __shfl_xor_sync(0xFFFFFFFF, sums[24], 16);
                float _add_285;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_285) : "f"(sums[24]), "f"(_shfl_xor_sync_286));
                sums[24] = _add_285;
                float s = 0.0f;
                s = ((lane_in_warp == 0) ? sums[0] : s);
                s = ((lane_in_warp == 1) ? sums[1] : s);
                s = ((lane_in_warp == 2) ? sums[2] : s);
                s = ((lane_in_warp == 3) ? sums[3] : s);
                s = ((lane_in_warp == 4) ? sums[4] : s);
                s = ((lane_in_warp == 5) ? sums[5] : s);
                s = ((lane_in_warp == 6) ? sums[6] : s);
                s = ((lane_in_warp == 7) ? sums[7] : s);
                s = ((lane_in_warp == 8) ? sums[8] : s);
                s = ((lane_in_warp == 9) ? sums[9] : s);
                s = ((lane_in_warp == 10) ? sums[10] : s);
                s = ((lane_in_warp == 11) ? sums[11] : s);
                s = ((lane_in_warp == 12) ? sums[12] : s);
                s = ((lane_in_warp == 13) ? sums[13] : s);
                s = ((lane_in_warp == 14) ? sums[14] : s);
                s = ((lane_in_warp == 15) ? sums[15] : s);
                s = ((lane_in_warp == 16) ? sums[16] : s);
                s = ((lane_in_warp == 17) ? sums[17] : s);
                s = ((lane_in_warp == 18) ? sums[18] : s);
                s = ((lane_in_warp == 19) ? sums[19] : s);
                s = ((lane_in_warp == 20) ? sums[20] : s);
                s = ((lane_in_warp == 21) ? sums[21] : s);
                s = ((lane_in_warp == 22) ? sums[22] : s);
                s = ((lane_in_warp == 23) ? sums[23] : s);
                float q = sums[24];
                float _fma_288;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_288) : "f"(q), "f"(6.103515625e-05f), "f"(hc_norm_eps));
                float _sqrt_287;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_287) : "f"(_fma_288));
                float r = __frcp_rn(_sqrt_287);
                float z;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(z) : "f"(r), "f"(s));
                float sc = ((lane_in_warp < 4) ? mix_scales[0] : ((lane_in_warp < 8) ? mix_scales[1] : mix_scales[2]));
                float bs = mix_bases[((lane_in_warp < 24) ? lane_in_warp : 23)];
                float a_map;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a_map) : "f"(z), "f"(sc), "f"(bs));
                float a = ((lane_in_warp < 24) ? a_map : 0.0f);
                float _sub_289;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_289) : "f"(0.0f), "f"(fabsf(a)));
                float _exp_0 = __nv_expf(_sub_289);
                float _add_40;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(1.0f), "f"(_exp_0));
                float _fdiv_rn_0 = __fdiv_rn(1.0f, _add_40);
                float _fdiv_rn_1 = __fdiv_rn(_exp_0, _add_40);
                float sig = ((a >= 0.0f) ? _fdiv_rn_0 : _fdiv_rn_1);
                float sneg = ((a >= 0.0f) ? _fdiv_rn_1 : _fdiv_rn_0);
                float _add_41_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41_1) : "f"(sig), "f"(hc_pre_eps));
                float _mul_0;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(hc_post_scale), "f"(sig));
                float logit = ((lane_in_warp >= 8 && lane_in_warp < 24) ? a : 0.0f);
                float _shfl_xor_sync_290 = __shfl_xor_sync(0xFFFFFFFF, logit, 1);
                float _max_0 = fmaxf(logit, _shfl_xor_sync_290);
                float _shfl_xor_sync_292 = __shfl_xor_sync(0xFFFFFFFF, _max_0, 2);
                float _sub_291;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_291) : "f"(logit), "f"(fmaxf(_max_0, _shfl_xor_sync_292)));
                float _exp_1 = __nv_expf(_sub_291);
                float _shfl_xor_sync_293 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 1);
                float _add_42;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(_exp_1), "f"(_shfl_xor_sync_293));
                float _shfl_xor_sync_296 = __shfl_xor_sync(0xFFFFFFFF, _add_42, 2);
                float _add_295;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_295) : "f"(_add_42), "f"(_shfl_xor_sync_296));
                float _fdiv_294 = __fdiv_rn(_exp_1, _add_295);
                float C;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(C) : "f"(_fdiv_294), "f"(sinkhorn_eps));
                float _shfl_xor_sync_297 = __shfl_xor_sync(0xFFFFFFFF, C, 4);
                float _add_43_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_43_1) : "f"(C), "f"(_shfl_xor_sync_297));
                float _shfl_xor_sync_301 = __shfl_xor_sync(0xFFFFFFFF, _add_43_1, 24);
                float _add_300;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_300) : "f"(_add_43_1), "f"(_shfl_xor_sync_301));
                float _add_299;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_299) : "f"(_add_300), "f"(sinkhorn_eps));
                float _fdiv_298 = __fdiv_rn(C, _add_299);
                C = _fdiv_298;
                #pragma unroll 1
                for (unsigned int iteration = 1; iteration < num_sinkhorn_iters; iteration++) {
                    float _shfl_xor_sync_302 = __shfl_xor_sync(0xFFFFFFFF, C, 1);
                    float _add_44;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(C), "f"(_shfl_xor_sync_302));
                    float _shfl_xor_sync_306 = __shfl_xor_sync(0xFFFFFFFF, _add_44, 2);
                    float _add_305;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_305) : "f"(_add_44), "f"(_shfl_xor_sync_306));
                    float _add_304;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_304) : "f"(_add_305), "f"(sinkhorn_eps));
                    float _fdiv_303 = __fdiv_rn(C, _add_304);
                    C = _fdiv_303;
                    float _shfl_xor_sync_307 = __shfl_xor_sync(0xFFFFFFFF, C, 4);
                    float _add_45_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45_1) : "f"(C), "f"(_shfl_xor_sync_307));
                    float _shfl_xor_sync_311 = __shfl_xor_sync(0xFFFFFFFF, _add_45_1, 24);
                    float _add_310;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_310) : "f"(_add_45_1), "f"(_shfl_xor_sync_311));
                    float _add_309;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_309) : "f"(_add_310), "f"(sinkhorn_eps));
                    float _fdiv_308 = __fdiv_rn(C, _add_309);
                    C = _fdiv_308;
                }
                if (lane_in_warp >= 8) {
                    if (lane_in_warp < 24) {
                        new_comb_res_mix[(unsigned long long)token * 16 + (unsigned long long)(lane_in_warp - 8)] = C;
                    }
                } else if (lane_in_warp >= 4) {
                    new_post_mix[(unsigned long long)token * 4 + (unsigned long long)(lane_in_warp - 4)] = _mul_0;
                }
                __syncwarp();
            }
        }
    }
    // ---- Role: norm ----
    if (warp >= 20 && warp <= 23) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // norm_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
        }
    }

    // Kernel teardown ops
    asm volatile("tcgen05.fence::before_thread_sync;");
    asm volatile("barrier.sync %0, 128;" :: "r"(9 + warp / 4) : "memory");
    asm volatile("setmaxnreg.inc.sync.aligned.u32 80;");
    unsigned int ticket = 0;
    unsigned int token_1 = 0;
    #pragma unroll 1
    for (unsigned int attempt = 0; attempt < num_tokens + 1; attempt++) {
        if (warp % 4 == 0) {
            if (lane == 0) {
                unsigned int _atomic_fetch_add_0 = atomicAdd(queue, 1);
                ticket = _atomic_fetch_add_0;
            }
            unsigned int _shfl_sync_1 = __shfl_sync(0xFFFFFFFF, ticket, 0);
            ticket = _shfl_sync_1;
            token_1 = (unsigned int)bid + ticket * num_ctas;
            if (lane_in_warp == 0) {
                tail_control[24 + warp / 4] = token_1;
            }
            if (token_1 < num_tokens) {
                unsigned int mb_2 = token_1 / 64;
                unsigned long long target_1 = (launch_epochs[bid] + 1) * 128 + (unsigned long long)num_splits;
                {
                unsigned long long _acquire_observed;
                do {
                asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_acquire_observed) : "l"(reinterpret_cast<unsigned long long*>(split_barriers + ((16384 + mb_2) * 16))) : "memory");
                } while (static_cast<unsigned long long>(_acquire_observed - static_cast<unsigned long long>(target_1)) >= static_cast<unsigned long long>(1));
                }
                unsigned int row_8 = token_1 % 64;
                unsigned int first_task_1 = mb_2 * num_splits;
                float sums_1[5];
                unsigned int split_1 = lane_in_warp * 2;
                unsigned int safe_split_1 = ((split_1 < num_splits) ? split_1 : 0);
                unsigned long long record_1 = ((unsigned long long)first_task_1 + (unsigned long long)safe_split_1) * 1536 + (unsigned long long)(row_8 * 24);
                float _vec_load_12[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_1 + 0);
                    _vec_load_12[0 + 0] = _v4.x;
                    _vec_load_12[0 + 1] = _v4.y;
                    _vec_load_12[0 + 2] = _v4.z;
                    _vec_load_12[0 + 3] = _v4.w;
                }
                float value_23 = ((split_1 < num_splits) ? _vec_load_12[0] : 0.0f);
                sums_1[0] = value_23;
                float value_0_1 = ((split_1 < num_splits) ? _vec_load_12[1] : 0.0f);
                sums_1[1] = value_0_1;
                float value_1_2 = ((split_1 < num_splits) ? _vec_load_12[2] : 0.0f);
                sums_1[2] = value_1_2;
                float value_2_2 = ((split_1 < num_splits) ? _vec_load_12[3] : 0.0f);
                sums_1[3] = value_2_2;
                float q_word_1 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_1 * 64 + (unsigned long long)(safe_split_1 * 64) + (unsigned long long)row_8];
                float q_value_1 = ((split_1 < num_splits) ? q_word_1 : 0.0f);
                sums_1[4] = q_value_1;
                unsigned int split_3 = lane_in_warp * 2 + 1;
                unsigned int safe_split_4 = ((split_3 < num_splits) ? split_3 : 0);
                unsigned long long record_5 = ((unsigned long long)first_task_1 + (unsigned long long)safe_split_4) * 1536 + (unsigned long long)(row_8 * 24);
                float _vec_load_13[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(scratch + record_5 + 0);
                    _vec_load_13[0 + 0] = _v4.x;
                    _vec_load_13[0 + 1] = _v4.y;
                    _vec_load_13[0 + 2] = _v4.z;
                    _vec_load_13[0 + 3] = _v4.w;
                }
                float value_6_1 = ((split_3 < num_splits) ? _vec_load_13[0] : 0.0f);
                float _add_4_3;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4_3) : "f"(sums_1[0]), "f"(value_6_1));
                sums_1[0] = _add_4_3;
                float value_7_1 = ((split_3 < num_splits) ? _vec_load_13[1] : 0.0f);
                float _add_5_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_5_1) : "f"(sums_1[1]), "f"(value_7_1));
                sums_1[1] = _add_5_1;
                float value_8_3 = ((split_3 < num_splits) ? _vec_load_13[2] : 0.0f);
                float _add_6_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_6_1) : "f"(sums_1[2]), "f"(value_8_3));
                sums_1[2] = _add_6_1;
                float value_9_1 = ((split_3 < num_splits) ? _vec_load_13[3] : 0.0f);
                float _add_7_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7_2) : "f"(sums_1[3]), "f"(value_9_1));
                sums_1[3] = _add_7_2;
                float q_word_10 = scratch[(unsigned long long)((num_tokens + 63) / 64 * num_splits) * 1536 + (unsigned long long)first_task_1 * 64 + (unsigned long long)(safe_split_4 * 64) + (unsigned long long)row_8];
                float q_value_11 = ((split_3 < num_splits) ? q_word_10 : 0.0f);
                float _add_8_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8_2) : "f"(sums_1[4]), "f"(q_value_11));
                sums_1[4] = _add_8_2;
                float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, sums_1[0], 1);
                float _add_9_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9_2) : "f"(sums_1[0]), "f"(_shfl_xor_sync_10));
                sums_1[0] = _add_9_2;
                float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, sums_1[1], 1);
                float _add_11_3;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11_3) : "f"(sums_1[1]), "f"(_shfl_xor_sync_12));
                sums_1[1] = _add_11_3;
                float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, sums_1[2], 1);
                float _add_13_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13_2) : "f"(sums_1[2]), "f"(_shfl_xor_sync_14));
                sums_1[2] = _add_13_2;
                float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, sums_1[3], 1);
                float _add_15_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15_2) : "f"(sums_1[3]), "f"(_shfl_xor_sync_16));
                sums_1[3] = _add_15_2;
                float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, sums_1[4], 1);
                float _add_17_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17_2) : "f"(sums_1[4]), "f"(_shfl_xor_sync_18));
                sums_1[4] = _add_17_2;
                float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, sums_1[0], 2);
                float _add_19_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19_2) : "f"(sums_1[0]), "f"(_shfl_xor_sync_20));
                sums_1[0] = _add_19_2;
                float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, sums_1[1], 2);
                float _add_21_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21_1) : "f"(sums_1[1]), "f"(_shfl_xor_sync_22));
                sums_1[1] = _add_21_1;
                float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, sums_1[2], 2);
                float _add_23_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23_2) : "f"(sums_1[2]), "f"(_shfl_xor_sync_24));
                sums_1[2] = _add_23_2;
                float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, sums_1[3], 2);
                float _add_25_4;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25_4) : "f"(sums_1[3]), "f"(_shfl_xor_sync_26));
                sums_1[3] = _add_25_4;
                float _shfl_xor_sync_28_2 = __shfl_xor_sync(0xFFFFFFFF, sums_1[4], 2);
                float _add_27_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27_2) : "f"(sums_1[4]), "f"(_shfl_xor_sync_28_2));
                sums_1[4] = _add_27_2;
                float _shfl_xor_sync_30_2 = __shfl_xor_sync(0xFFFFFFFF, sums_1[0], 4);
                float _add_29_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29_2) : "f"(sums_1[0]), "f"(_shfl_xor_sync_30_2));
                sums_1[0] = _add_29_2;
                float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, sums_1[1], 4);
                float _add_31_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_31_1) : "f"(sums_1[1]), "f"(_shfl_xor_sync_32));
                sums_1[1] = _add_31_1;
                float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, sums_1[2], 4);
                float _add_33_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33_2) : "f"(sums_1[2]), "f"(_shfl_xor_sync_34));
                sums_1[2] = _add_33_2;
                float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, sums_1[3], 4);
                float _add_35_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35_2) : "f"(sums_1[3]), "f"(_shfl_xor_sync_36));
                sums_1[3] = _add_35_2;
                float _shfl_xor_sync_38_1 = __shfl_xor_sync(0xFFFFFFFF, sums_1[4], 4);
                float _add_37_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37_2) : "f"(sums_1[4]), "f"(_shfl_xor_sync_38_1));
                sums_1[4] = _add_37_2;
                float _shfl_xor_sync_40_1 = __shfl_xor_sync(0xFFFFFFFF, sums_1[0], 8);
                float _add_39_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39_2) : "f"(sums_1[0]), "f"(_shfl_xor_sync_40_1));
                sums_1[0] = _add_39_2;
                float _shfl_xor_sync_42_1 = __shfl_xor_sync(0xFFFFFFFF, sums_1[1], 8);
                float _add_41_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41_2) : "f"(sums_1[1]), "f"(_shfl_xor_sync_42_1));
                sums_1[1] = _add_41_2;
                float _shfl_xor_sync_44_1 = __shfl_xor_sync(0xFFFFFFFF, sums_1[2], 8);
                float _add_43_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_43_2) : "f"(sums_1[2]), "f"(_shfl_xor_sync_44_1));
                sums_1[2] = _add_43_2;
                float _shfl_xor_sync_46_1 = __shfl_xor_sync(0xFFFFFFFF, sums_1[3], 8);
                float _add_45_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45_2) : "f"(sums_1[3]), "f"(_shfl_xor_sync_46_1));
                sums_1[3] = _add_45_2;
                float _shfl_xor_sync_48_1 = __shfl_xor_sync(0xFFFFFFFF, sums_1[4], 8);
                float _add_47_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_47_1) : "f"(sums_1[4]), "f"(_shfl_xor_sync_48_1));
                sums_1[4] = _add_47_1;
                float _shfl_xor_sync_50_1 = __shfl_xor_sync(0xFFFFFFFF, sums_1[0], 16);
                float _add_49_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49_1) : "f"(sums_1[0]), "f"(_shfl_xor_sync_50_1));
                sums_1[0] = _add_49_1;
                float _shfl_xor_sync_52_1 = __shfl_xor_sync(0xFFFFFFFF, sums_1[1], 16);
                float _add_51_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51_1) : "f"(sums_1[1]), "f"(_shfl_xor_sync_52_1));
                sums_1[1] = _add_51_1;
                float _shfl_xor_sync_54_1 = __shfl_xor_sync(0xFFFFFFFF, sums_1[2], 16);
                float _add_53_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53_1) : "f"(sums_1[2]), "f"(_shfl_xor_sync_54_1));
                sums_1[2] = _add_53_1;
                float _shfl_xor_sync_56_1 = __shfl_xor_sync(0xFFFFFFFF, sums_1[3], 16);
                float _add_55_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_55_1) : "f"(sums_1[3]), "f"(_shfl_xor_sync_56_1));
                sums_1[3] = _add_55_1;
                float _shfl_xor_sync_58_1 = __shfl_xor_sync(0xFFFFFFFF, sums_1[4], 16);
                float _add_57_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57_1) : "f"(sums_1[4]), "f"(_shfl_xor_sync_58_1));
                sums_1[4] = _add_57_1;
                float s_1 = 0.0f;
                s_1 = ((lane_in_warp == 0) ? sums_1[0] : s_1);
                s_1 = ((lane_in_warp == 1) ? sums_1[1] : s_1);
                s_1 = ((lane_in_warp == 2) ? sums_1[2] : s_1);
                s_1 = ((lane_in_warp == 3) ? sums_1[3] : s_1);
                float q_1 = sums_1[4];
                float _fma_60;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(q_1), "f"(6.103515625e-05f), "f"(hc_norm_eps));
                float _sqrt_59;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_59) : "f"(_fma_60));
                float r_1 = __frcp_rn(_sqrt_59);
                float z_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(z_1) : "f"(r_1), "f"(s_1));
                float sc_1 = ((lane_in_warp < 4) ? mix_scales[0] : ((lane_in_warp < 8) ? mix_scales[1] : mix_scales[2]));
                float bs_1 = mix_bases[((lane_in_warp < 24) ? lane_in_warp : 23)];
                float a_map_1;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a_map_1) : "f"(z_1), "f"(sc_1), "f"(bs_1));
                float a_1 = ((lane_in_warp < 24) ? a_map_1 : 0.0f);
                float _sub_61;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_61) : "f"(0.0f), "f"(fabsf(a_1)));
                float _exp_2 = __nv_expf(_sub_61);
                float _add_46;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(1.0f), "f"(_exp_2));
                float _fdiv_rn_2 = __fdiv_rn(1.0f, _add_46);
                float _fdiv_rn_3 = __fdiv_rn(_exp_2, _add_46);
                float sig_1 = ((a_1 >= 0.0f) ? _fdiv_rn_2 : _fdiv_rn_3);
                float sneg_1 = ((a_1 >= 0.0f) ? _fdiv_rn_3 : _fdiv_rn_2);
                float pre;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(pre) : "f"(sig_1), "f"(hc_pre_eps));
                __syncwarp();
                if (lane_in_warp < 4) {
                    tail_scratch[30 + warp / 4 * 4 + lane_in_warp] = pre;
                }
            }
        }
        asm volatile("barrier.sync %0, 128;" :: "r"(9 + warp / 4) : "memory");
        token_1 = tail_control[24 + warp / 4];
        if (token_1 < num_tokens) {
            float coeff_2 = ((lane_in_warp < 4) ? tail_scratch[30 + warp / 4 * 4 + lane_in_warp] : 0.0f);
            unsigned int retained[16];
            float sums_2[2];
            sums_2[0] = 0.0f;
            sums_2[1] = 0.0f;
            unsigned int loaded[32];
            unsigned int hidden_pack = warp % 4 * 4;
            unsigned long long off = (unsigned long long)token_1 * 16384 + (unsigned long long)(hidden_pack * 256) + (unsigned long long)(lane_in_warp * 8);
            {
                const uint4* _vptr_62 = reinterpret_cast<const uint4*>(new_residual + off + 0);
                uint4* _vdst_62 = reinterpret_cast<uint4*>(&loaded[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_62[_blk].x), "=r"(_vdst_62[_blk].y), "=r"(_vdst_62[_blk].z), "=r"(_vdst_62[_blk].w) : "l"((const void*)(_vptr_62 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_63 = reinterpret_cast<const uint4*>(new_residual + (off + 4096) + 0);
                uint4* _vdst_63 = reinterpret_cast<uint4*>(&loaded[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_63[_blk].x), "=r"(_vdst_63[_blk].y), "=r"(_vdst_63[_blk].z), "=r"(_vdst_63[_blk].w) : "l"((const void*)(_vptr_63 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_64 = reinterpret_cast<const uint4*>(new_residual + (off + 8192) + 0);
                uint4* _vdst_64 = reinterpret_cast<uint4*>(&loaded[8]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_64[_blk].x), "=r"(_vdst_64[_blk].y), "=r"(_vdst_64[_blk].z), "=r"(_vdst_64[_blk].w) : "l"((const void*)(_vptr_64 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_65 = reinterpret_cast<const uint4*>(new_residual + (off + 12288) + 0);
                uint4* _vdst_65 = reinterpret_cast<uint4*>(&loaded[12]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_65[_blk].x), "=r"(_vdst_65[_blk].y), "=r"(_vdst_65[_blk].z), "=r"(_vdst_65[_blk].w) : "l"((const void*)(_vptr_65 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            unsigned int hidden_pack_0 = warp % 4 * 4 + 1;
            unsigned long long off_1 = (unsigned long long)token_1 * 16384 + (unsigned long long)(hidden_pack_0 * 256) + (unsigned long long)(lane_in_warp * 8);
            {
                const uint4* _vptr_66 = reinterpret_cast<const uint4*>(new_residual + off_1 + 0);
                uint4* _vdst_66 = reinterpret_cast<uint4*>(&loaded[16]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_66[_blk].x), "=r"(_vdst_66[_blk].y), "=r"(_vdst_66[_blk].z), "=r"(_vdst_66[_blk].w) : "l"((const void*)(_vptr_66 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_67 = reinterpret_cast<const uint4*>(new_residual + (off_1 + 4096) + 0);
                uint4* _vdst_67 = reinterpret_cast<uint4*>(&loaded[20]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_67[_blk].x), "=r"(_vdst_67[_blk].y), "=r"(_vdst_67[_blk].z), "=r"(_vdst_67[_blk].w) : "l"((const void*)(_vptr_67 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_68 = reinterpret_cast<const uint4*>(new_residual + (off_1 + 8192) + 0);
                uint4* _vdst_68 = reinterpret_cast<uint4*>(&loaded[24]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_68[_blk].x), "=r"(_vdst_68[_blk].y), "=r"(_vdst_68[_blk].z), "=r"(_vdst_68[_blk].w) : "l"((const void*)(_vptr_68 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_69 = reinterpret_cast<const uint4*>(new_residual + (off_1 + 12288) + 0);
                uint4* _vdst_69 = reinterpret_cast<uint4*>(&loaded[28]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_69[_blk].x), "=r"(_vdst_69[_blk].y), "=r"(_vdst_69[_blk].z), "=r"(_vdst_69[_blk].w) : "l"((const void*)(_vptr_69 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            float values_1[8];
            float _shfl_0 = __shfl_sync(0xFFFFFFFF, coeff_2, 0);
            unsigned int word_4 = loaded[0];
            float2 _f32x2_mul_a_70 = make_float2(_shfl_0, _shfl_0);
            float2 _f32x2_mul_b_71 = make_float2(__uint_as_float(word_4 << 16), __uint_as_float(word_4 & 4294901760u));
            float2 pair_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&_f32x2_mul_b_71));
            values_1[0] = pair_2.x;
            values_1[1] = pair_2.y;
            unsigned int word_2_2 = loaded[1];
            float2 _f32x2_mul_a_72 = make_float2(_shfl_0, _shfl_0);
            float2 _f32x2_mul_b_73 = make_float2(__uint_as_float(word_2_2 << 16), __uint_as_float(word_2_2 & 4294901760u));
            float2 pair_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_72), "l"(*(const unsigned long long*)&_f32x2_mul_b_73));
            values_1[2] = pair_3.x;
            values_1[3] = pair_3.y;
            unsigned int word_4_1 = loaded[2];
            float2 _f32x2_mul_a_74 = make_float2(_shfl_0, _shfl_0);
            float2 _f32x2_mul_b_75 = make_float2(__uint_as_float(word_4_1 << 16), __uint_as_float(word_4_1 & 4294901760u));
            float2 pair_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_74), "l"(*(const unsigned long long*)&_f32x2_mul_b_75));
            values_1[4] = pair_5.x;
            values_1[5] = pair_5.y;
            unsigned int word_6_2 = loaded[3];
            float2 _f32x2_mul_a_76 = make_float2(_shfl_0, _shfl_0);
            float2 _f32x2_mul_b_77 = make_float2(__uint_as_float(word_6_2 << 16), __uint_as_float(word_6_2 & 4294901760u));
            float2 pair_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_76), "l"(*(const unsigned long long*)&_f32x2_mul_b_77));
            values_1[6] = pair_7.x;
            values_1[7] = pair_7.y;
            float _shfl_1 = __shfl_sync(0xFFFFFFFF, coeff_2, 1);
            unsigned int word_8 = loaded[4];
            float2 pair_9 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(__uint_as_float(word_8 << 16), __uint_as_float(word_8 & 4294901760u)), make_float2(values_1[0], values_1[1]));
            values_1[0] = pair_9.x;
            values_1[1] = pair_9.y;
            unsigned int word_10 = loaded[5];
            float2 pair_11 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(__uint_as_float(word_10 << 16), __uint_as_float(word_10 & 4294901760u)), make_float2(values_1[2], values_1[3]));
            values_1[2] = pair_11.x;
            values_1[3] = pair_11.y;
            unsigned int word_12 = loaded[6];
            float2 pair_13 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(__uint_as_float(word_12 << 16), __uint_as_float(word_12 & 4294901760u)), make_float2(values_1[4], values_1[5]));
            values_1[4] = pair_13.x;
            values_1[5] = pair_13.y;
            unsigned int word_14 = loaded[7];
            float2 pair_15 = fma_f32x2_rn_noftz(make_float2(_shfl_1, _shfl_1), make_float2(__uint_as_float(word_14 << 16), __uint_as_float(word_14 & 4294901760u)), make_float2(values_1[6], values_1[7]));
            values_1[6] = pair_15.x;
            values_1[7] = pair_15.y;
            float _shfl_2 = __shfl_sync(0xFFFFFFFF, coeff_2, 2);
            unsigned int word_16 = loaded[8];
            float2 pair_17 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(__uint_as_float(word_16 << 16), __uint_as_float(word_16 & 4294901760u)), make_float2(values_1[0], values_1[1]));
            values_1[0] = pair_17.x;
            values_1[1] = pair_17.y;
            unsigned int word_18 = loaded[9];
            float2 pair_19 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(__uint_as_float(word_18 << 16), __uint_as_float(word_18 & 4294901760u)), make_float2(values_1[2], values_1[3]));
            values_1[2] = pair_19.x;
            values_1[3] = pair_19.y;
            unsigned int word_20 = loaded[10];
            float2 pair_21 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(__uint_as_float(word_20 << 16), __uint_as_float(word_20 & 4294901760u)), make_float2(values_1[4], values_1[5]));
            values_1[4] = pair_21.x;
            values_1[5] = pair_21.y;
            unsigned int word_22_2 = loaded[11];
            float2 pair_23 = fma_f32x2_rn_noftz(make_float2(_shfl_2, _shfl_2), make_float2(__uint_as_float(word_22_2 << 16), __uint_as_float(word_22_2 & 4294901760u)), make_float2(values_1[6], values_1[7]));
            values_1[6] = pair_23.x;
            values_1[7] = pair_23.y;
            float _shfl_3 = __shfl_sync(0xFFFFFFFF, coeff_2, 3);
            unsigned int word_24 = loaded[12];
            float2 pair_25 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(__uint_as_float(word_24 << 16), __uint_as_float(word_24 & 4294901760u)), make_float2(values_1[0], values_1[1]));
            values_1[0] = pair_25.x;
            values_1[1] = pair_25.y;
            unsigned int word_26 = loaded[13];
            float2 pair_27 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(__uint_as_float(word_26 << 16), __uint_as_float(word_26 & 4294901760u)), make_float2(values_1[2], values_1[3]));
            values_1[2] = pair_27.x;
            values_1[3] = pair_27.y;
            unsigned int word_28 = loaded[14];
            float2 pair_29 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(__uint_as_float(word_28 << 16), __uint_as_float(word_28 & 4294901760u)), make_float2(values_1[4], values_1[5]));
            values_1[4] = pair_29.x;
            values_1[5] = pair_29.y;
            unsigned int word_30 = loaded[15];
            float2 pair_31 = fma_f32x2_rn_noftz(make_float2(_shfl_3, _shfl_3), make_float2(__uint_as_float(word_30 << 16), __uint_as_float(word_30 & 4294901760u)), make_float2(values_1[6], values_1[7]));
            values_1[6] = pair_31.x;
            values_1[7] = pair_31.y;
            float2 w_pair = make_float2(values_1[0], values_1[1]);
            float2 squares_2 = fma_f32x2_rn_noftz(w_pair, w_pair, make_float2(sums_2[0], sums_2[1]));
            sums_2[0] = squares_2.x;
            sums_2[1] = squares_2.y;
            float2 w_pair_32 = make_float2(values_1[2], values_1[3]);
            float2 squares_33_2 = fma_f32x2_rn_noftz(w_pair_32, w_pair_32, make_float2(sums_2[0], sums_2[1]));
            sums_2[0] = squares_33_2.x;
            sums_2[1] = squares_33_2.y;
            float2 w_pair_34 = make_float2(values_1[4], values_1[5]);
            float2 squares_35 = fma_f32x2_rn_noftz(w_pair_34, w_pair_34, make_float2(sums_2[0], sums_2[1]));
            sums_2[0] = squares_35.x;
            sums_2[1] = squares_35.y;
            float2 w_pair_36 = make_float2(values_1[6], values_1[7]);
            float2 squares_37 = fma_f32x2_rn_noftz(w_pair_36, w_pair_36, make_float2(sums_2[0], sums_2[1]));
            sums_2[0] = squares_37.x;
            sums_2[1] = squares_37.y;
            retained[0] = __as_u32(__float22bfloat162_rn(make_float2(values_1[0], values_1[1])));
            retained[1] = __as_u32(__float22bfloat162_rn(make_float2(values_1[2], values_1[3])));
            retained[2] = __as_u32(__float22bfloat162_rn(make_float2(values_1[4], values_1[5])));
            retained[3] = __as_u32(__float22bfloat162_rn(make_float2(values_1[6], values_1[7])));
            float values_38[8];
            float _shfl_4 = __shfl_sync(0xFFFFFFFF, coeff_2, 0);
            unsigned int word_39_2 = loaded[16];
            float2 _f32x2_mul_a_78 = make_float2(_shfl_4, _shfl_4);
            float2 _f32x2_mul_b_79 = make_float2(__uint_as_float(word_39_2 << 16), __uint_as_float(word_39_2 & 4294901760u));
            float2 pair_40;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_40) : "l"(*(const unsigned long long*)&_f32x2_mul_a_78), "l"(*(const unsigned long long*)&_f32x2_mul_b_79));
            values_38[0] = pair_40.x;
            values_38[1] = pair_40.y;
            unsigned int word_41_2 = loaded[17];
            float2 _f32x2_mul_a_80 = make_float2(_shfl_4, _shfl_4);
            float2 _f32x2_mul_b_81 = make_float2(__uint_as_float(word_41_2 << 16), __uint_as_float(word_41_2 & 4294901760u));
            float2 pair_42;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_42) : "l"(*(const unsigned long long*)&_f32x2_mul_a_80), "l"(*(const unsigned long long*)&_f32x2_mul_b_81));
            values_38[2] = pair_42.x;
            values_38[3] = pair_42.y;
            unsigned int word_43 = loaded[18];
            float2 _f32x2_mul_a_82 = make_float2(_shfl_4, _shfl_4);
            float2 _f32x2_mul_b_83 = make_float2(__uint_as_float(word_43 << 16), __uint_as_float(word_43 & 4294901760u));
            float2 pair_44;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_44) : "l"(*(const unsigned long long*)&_f32x2_mul_a_82), "l"(*(const unsigned long long*)&_f32x2_mul_b_83));
            values_38[4] = pair_44.x;
            values_38[5] = pair_44.y;
            unsigned int word_45 = loaded[19];
            float2 _f32x2_mul_a_84 = make_float2(_shfl_4, _shfl_4);
            float2 _f32x2_mul_b_85 = make_float2(__uint_as_float(word_45 << 16), __uint_as_float(word_45 & 4294901760u));
            float2 pair_46;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_46) : "l"(*(const unsigned long long*)&_f32x2_mul_a_84), "l"(*(const unsigned long long*)&_f32x2_mul_b_85));
            values_38[6] = pair_46.x;
            values_38[7] = pair_46.y;
            float _shfl_5 = __shfl_sync(0xFFFFFFFF, coeff_2, 1);
            unsigned int word_47 = loaded[20];
            float2 pair_48 = fma_f32x2_rn_noftz(make_float2(_shfl_5, _shfl_5), make_float2(__uint_as_float(word_47 << 16), __uint_as_float(word_47 & 4294901760u)), make_float2(values_38[0], values_38[1]));
            values_38[0] = pair_48.x;
            values_38[1] = pair_48.y;
            unsigned int word_49 = loaded[21];
            float2 pair_50 = fma_f32x2_rn_noftz(make_float2(_shfl_5, _shfl_5), make_float2(__uint_as_float(word_49 << 16), __uint_as_float(word_49 & 4294901760u)), make_float2(values_38[2], values_38[3]));
            values_38[2] = pair_50.x;
            values_38[3] = pair_50.y;
            unsigned int word_51 = loaded[22];
            float2 pair_52 = fma_f32x2_rn_noftz(make_float2(_shfl_5, _shfl_5), make_float2(__uint_as_float(word_51 << 16), __uint_as_float(word_51 & 4294901760u)), make_float2(values_38[4], values_38[5]));
            values_38[4] = pair_52.x;
            values_38[5] = pair_52.y;
            unsigned int word_53 = loaded[23];
            float2 pair_54 = fma_f32x2_rn_noftz(make_float2(_shfl_5, _shfl_5), make_float2(__uint_as_float(word_53 << 16), __uint_as_float(word_53 & 4294901760u)), make_float2(values_38[6], values_38[7]));
            values_38[6] = pair_54.x;
            values_38[7] = pair_54.y;
            float _shfl_6 = __shfl_sync(0xFFFFFFFF, coeff_2, 2);
            unsigned int word_55 = loaded[24];
            float2 pair_56 = fma_f32x2_rn_noftz(make_float2(_shfl_6, _shfl_6), make_float2(__uint_as_float(word_55 << 16), __uint_as_float(word_55 & 4294901760u)), make_float2(values_38[0], values_38[1]));
            values_38[0] = pair_56.x;
            values_38[1] = pair_56.y;
            unsigned int word_57 = loaded[25];
            float2 pair_58 = fma_f32x2_rn_noftz(make_float2(_shfl_6, _shfl_6), make_float2(__uint_as_float(word_57 << 16), __uint_as_float(word_57 & 4294901760u)), make_float2(values_38[2], values_38[3]));
            values_38[2] = pair_58.x;
            values_38[3] = pair_58.y;
            unsigned int word_59_2 = loaded[26];
            float2 pair_60 = fma_f32x2_rn_noftz(make_float2(_shfl_6, _shfl_6), make_float2(__uint_as_float(word_59_2 << 16), __uint_as_float(word_59_2 & 4294901760u)), make_float2(values_38[4], values_38[5]));
            values_38[4] = pair_60.x;
            values_38[5] = pair_60.y;
            unsigned int word_61 = loaded[27];
            float2 pair_62 = fma_f32x2_rn_noftz(make_float2(_shfl_6, _shfl_6), make_float2(__uint_as_float(word_61 << 16), __uint_as_float(word_61 & 4294901760u)), make_float2(values_38[6], values_38[7]));
            values_38[6] = pair_62.x;
            values_38[7] = pair_62.y;
            float _shfl_7 = __shfl_sync(0xFFFFFFFF, coeff_2, 3);
            unsigned int word_63 = loaded[28];
            float2 pair_64 = fma_f32x2_rn_noftz(make_float2(_shfl_7, _shfl_7), make_float2(__uint_as_float(word_63 << 16), __uint_as_float(word_63 & 4294901760u)), make_float2(values_38[0], values_38[1]));
            values_38[0] = pair_64.x;
            values_38[1] = pair_64.y;
            unsigned int word_65 = loaded[29];
            float2 pair_66 = fma_f32x2_rn_noftz(make_float2(_shfl_7, _shfl_7), make_float2(__uint_as_float(word_65 << 16), __uint_as_float(word_65 & 4294901760u)), make_float2(values_38[2], values_38[3]));
            values_38[2] = pair_66.x;
            values_38[3] = pair_66.y;
            unsigned int word_67 = loaded[30];
            float2 pair_68 = fma_f32x2_rn_noftz(make_float2(_shfl_7, _shfl_7), make_float2(__uint_as_float(word_67 << 16), __uint_as_float(word_67 & 4294901760u)), make_float2(values_38[4], values_38[5]));
            values_38[4] = pair_68.x;
            values_38[5] = pair_68.y;
            unsigned int word_69 = loaded[31];
            float2 pair_70 = fma_f32x2_rn_noftz(make_float2(_shfl_7, _shfl_7), make_float2(__uint_as_float(word_69 << 16), __uint_as_float(word_69 & 4294901760u)), make_float2(values_38[6], values_38[7]));
            values_38[6] = pair_70.x;
            values_38[7] = pair_70.y;
            float2 w_pair_71 = make_float2(values_38[0], values_38[1]);
            float2 squares_72 = fma_f32x2_rn_noftz(w_pair_71, w_pair_71, make_float2(sums_2[0], sums_2[1]));
            sums_2[0] = squares_72.x;
            sums_2[1] = squares_72.y;
            float2 w_pair_73 = make_float2(values_38[2], values_38[3]);
            float2 squares_74 = fma_f32x2_rn_noftz(w_pair_73, w_pair_73, make_float2(sums_2[0], sums_2[1]));
            sums_2[0] = squares_74.x;
            sums_2[1] = squares_74.y;
            float2 w_pair_75 = make_float2(values_38[4], values_38[5]);
            float2 squares_76 = fma_f32x2_rn_noftz(w_pair_75, w_pair_75, make_float2(sums_2[0], sums_2[1]));
            sums_2[0] = squares_76.x;
            sums_2[1] = squares_76.y;
            float2 w_pair_77 = make_float2(values_38[6], values_38[7]);
            float2 squares_78 = fma_f32x2_rn_noftz(w_pair_77, w_pair_77, make_float2(sums_2[0], sums_2[1]));
            sums_2[0] = squares_78.x;
            sums_2[1] = squares_78.y;
            retained[4] = __as_u32(__float22bfloat162_rn(make_float2(values_38[0], values_38[1])));
            retained[5] = __as_u32(__float22bfloat162_rn(make_float2(values_38[2], values_38[3])));
            retained[6] = __as_u32(__float22bfloat162_rn(make_float2(values_38[4], values_38[5])));
            retained[7] = __as_u32(__float22bfloat162_rn(make_float2(values_38[6], values_38[7])));
            unsigned int loaded_79[32];
            unsigned int hidden_pack_80 = warp % 4 * 4 + 2;
            unsigned long long off_81 = (unsigned long long)token_1 * 16384 + (unsigned long long)(hidden_pack_80 * 256) + (unsigned long long)(lane_in_warp * 8);
            {
                const uint4* _vptr_86 = reinterpret_cast<const uint4*>(new_residual + off_81 + 0);
                uint4* _vdst_86 = reinterpret_cast<uint4*>(&loaded_79[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_86[_blk].x), "=r"(_vdst_86[_blk].y), "=r"(_vdst_86[_blk].z), "=r"(_vdst_86[_blk].w) : "l"((const void*)(_vptr_86 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_87 = reinterpret_cast<const uint4*>(new_residual + (off_81 + 4096) + 0);
                uint4* _vdst_87 = reinterpret_cast<uint4*>(&loaded_79[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_87[_blk].x), "=r"(_vdst_87[_blk].y), "=r"(_vdst_87[_blk].z), "=r"(_vdst_87[_blk].w) : "l"((const void*)(_vptr_87 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_88 = reinterpret_cast<const uint4*>(new_residual + (off_81 + 8192) + 0);
                uint4* _vdst_88 = reinterpret_cast<uint4*>(&loaded_79[8]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_88[_blk].x), "=r"(_vdst_88[_blk].y), "=r"(_vdst_88[_blk].z), "=r"(_vdst_88[_blk].w) : "l"((const void*)(_vptr_88 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_89 = reinterpret_cast<const uint4*>(new_residual + (off_81 + 12288) + 0);
                uint4* _vdst_89 = reinterpret_cast<uint4*>(&loaded_79[12]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_89[_blk].x), "=r"(_vdst_89[_blk].y), "=r"(_vdst_89[_blk].z), "=r"(_vdst_89[_blk].w) : "l"((const void*)(_vptr_89 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            unsigned int hidden_pack_82 = warp % 4 * 4 + 3;
            unsigned long long off_83 = (unsigned long long)token_1 * 16384 + (unsigned long long)(hidden_pack_82 * 256) + (unsigned long long)(lane_in_warp * 8);
            {
                const uint4* _vptr_90 = reinterpret_cast<const uint4*>(new_residual + off_83 + 0);
                uint4* _vdst_90 = reinterpret_cast<uint4*>(&loaded_79[16]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_90[_blk].x), "=r"(_vdst_90[_blk].y), "=r"(_vdst_90[_blk].z), "=r"(_vdst_90[_blk].w) : "l"((const void*)(_vptr_90 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_91 = reinterpret_cast<const uint4*>(new_residual + (off_83 + 4096) + 0);
                uint4* _vdst_91 = reinterpret_cast<uint4*>(&loaded_79[20]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_91[_blk].x), "=r"(_vdst_91[_blk].y), "=r"(_vdst_91[_blk].z), "=r"(_vdst_91[_blk].w) : "l"((const void*)(_vptr_91 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_92 = reinterpret_cast<const uint4*>(new_residual + (off_83 + 8192) + 0);
                uint4* _vdst_92 = reinterpret_cast<uint4*>(&loaded_79[24]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_92[_blk].x), "=r"(_vdst_92[_blk].y), "=r"(_vdst_92[_blk].z), "=r"(_vdst_92[_blk].w) : "l"((const void*)(_vptr_92 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_93 = reinterpret_cast<const uint4*>(new_residual + (off_83 + 12288) + 0);
                uint4* _vdst_93 = reinterpret_cast<uint4*>(&loaded_79[28]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_93[_blk].x), "=r"(_vdst_93[_blk].y), "=r"(_vdst_93[_blk].z), "=r"(_vdst_93[_blk].w) : "l"((const void*)(_vptr_93 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            float values_84[8];
            float _shfl_8 = __shfl_sync(0xFFFFFFFF, coeff_2, 0);
            unsigned int word_85 = loaded_79[0];
            float2 _f32x2_mul_a_94 = make_float2(_shfl_8, _shfl_8);
            float2 _f32x2_mul_b_95 = make_float2(__uint_as_float(word_85 << 16), __uint_as_float(word_85 & 4294901760u));
            float2 pair_86;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_86) : "l"(*(const unsigned long long*)&_f32x2_mul_a_94), "l"(*(const unsigned long long*)&_f32x2_mul_b_95));
            values_84[0] = pair_86.x;
            values_84[1] = pair_86.y;
            unsigned int word_87_2 = loaded_79[1];
            float2 _f32x2_mul_a_96 = make_float2(_shfl_8, _shfl_8);
            float2 _f32x2_mul_b_97 = make_float2(__uint_as_float(word_87_2 << 16), __uint_as_float(word_87_2 & 4294901760u));
            float2 pair_88;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_88) : "l"(*(const unsigned long long*)&_f32x2_mul_a_96), "l"(*(const unsigned long long*)&_f32x2_mul_b_97));
            values_84[2] = pair_88.x;
            values_84[3] = pair_88.y;
            unsigned int word_89_2 = loaded_79[2];
            float2 _f32x2_mul_a_98 = make_float2(_shfl_8, _shfl_8);
            float2 _f32x2_mul_b_99 = make_float2(__uint_as_float(word_89_2 << 16), __uint_as_float(word_89_2 & 4294901760u));
            float2 pair_90;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_90) : "l"(*(const unsigned long long*)&_f32x2_mul_a_98), "l"(*(const unsigned long long*)&_f32x2_mul_b_99));
            values_84[4] = pair_90.x;
            values_84[5] = pair_90.y;
            unsigned int word_91 = loaded_79[3];
            float2 _f32x2_mul_a_100 = make_float2(_shfl_8, _shfl_8);
            float2 _f32x2_mul_b_101 = make_float2(__uint_as_float(word_91 << 16), __uint_as_float(word_91 & 4294901760u));
            float2 pair_92;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_92) : "l"(*(const unsigned long long*)&_f32x2_mul_a_100), "l"(*(const unsigned long long*)&_f32x2_mul_b_101));
            values_84[6] = pair_92.x;
            values_84[7] = pair_92.y;
            float _shfl_9 = __shfl_sync(0xFFFFFFFF, coeff_2, 1);
            unsigned int word_93 = loaded_79[4];
            float2 pair_94 = fma_f32x2_rn_noftz(make_float2(_shfl_9, _shfl_9), make_float2(__uint_as_float(word_93 << 16), __uint_as_float(word_93 & 4294901760u)), make_float2(values_84[0], values_84[1]));
            values_84[0] = pair_94.x;
            values_84[1] = pair_94.y;
            unsigned int word_95 = loaded_79[5];
            float2 pair_96 = fma_f32x2_rn_noftz(make_float2(_shfl_9, _shfl_9), make_float2(__uint_as_float(word_95 << 16), __uint_as_float(word_95 & 4294901760u)), make_float2(values_84[2], values_84[3]));
            values_84[2] = pair_96.x;
            values_84[3] = pair_96.y;
            unsigned int word_97_2 = loaded_79[6];
            float2 pair_98 = fma_f32x2_rn_noftz(make_float2(_shfl_9, _shfl_9), make_float2(__uint_as_float(word_97_2 << 16), __uint_as_float(word_97_2 & 4294901760u)), make_float2(values_84[4], values_84[5]));
            values_84[4] = pair_98.x;
            values_84[5] = pair_98.y;
            unsigned int word_99 = loaded_79[7];
            float2 pair_100 = fma_f32x2_rn_noftz(make_float2(_shfl_9, _shfl_9), make_float2(__uint_as_float(word_99 << 16), __uint_as_float(word_99 & 4294901760u)), make_float2(values_84[6], values_84[7]));
            values_84[6] = pair_100.x;
            values_84[7] = pair_100.y;
            float _shfl_10 = __shfl_sync(0xFFFFFFFF, coeff_2, 2);
            unsigned int word_101 = loaded_79[8];
            float2 pair_102 = fma_f32x2_rn_noftz(make_float2(_shfl_10, _shfl_10), make_float2(__uint_as_float(word_101 << 16), __uint_as_float(word_101 & 4294901760u)), make_float2(values_84[0], values_84[1]));
            values_84[0] = pair_102.x;
            values_84[1] = pair_102.y;
            unsigned int word_103 = loaded_79[9];
            float2 pair_104 = fma_f32x2_rn_noftz(make_float2(_shfl_10, _shfl_10), make_float2(__uint_as_float(word_103 << 16), __uint_as_float(word_103 & 4294901760u)), make_float2(values_84[2], values_84[3]));
            values_84[2] = pair_104.x;
            values_84[3] = pair_104.y;
            unsigned int word_105 = loaded_79[10];
            float2 pair_106 = fma_f32x2_rn_noftz(make_float2(_shfl_10, _shfl_10), make_float2(__uint_as_float(word_105 << 16), __uint_as_float(word_105 & 4294901760u)), make_float2(values_84[4], values_84[5]));
            values_84[4] = pair_106.x;
            values_84[5] = pair_106.y;
            unsigned int word_107 = loaded_79[11];
            float2 pair_108 = fma_f32x2_rn_noftz(make_float2(_shfl_10, _shfl_10), make_float2(__uint_as_float(word_107 << 16), __uint_as_float(word_107 & 4294901760u)), make_float2(values_84[6], values_84[7]));
            values_84[6] = pair_108.x;
            values_84[7] = pair_108.y;
            float _shfl_11 = __shfl_sync(0xFFFFFFFF, coeff_2, 3);
            unsigned int word_109 = loaded_79[12];
            float2 pair_110 = fma_f32x2_rn_noftz(make_float2(_shfl_11, _shfl_11), make_float2(__uint_as_float(word_109 << 16), __uint_as_float(word_109 & 4294901760u)), make_float2(values_84[0], values_84[1]));
            values_84[0] = pair_110.x;
            values_84[1] = pair_110.y;
            unsigned int word_111 = loaded_79[13];
            float2 pair_112 = fma_f32x2_rn_noftz(make_float2(_shfl_11, _shfl_11), make_float2(__uint_as_float(word_111 << 16), __uint_as_float(word_111 & 4294901760u)), make_float2(values_84[2], values_84[3]));
            values_84[2] = pair_112.x;
            values_84[3] = pair_112.y;
            unsigned int word_113 = loaded_79[14];
            float2 pair_114 = fma_f32x2_rn_noftz(make_float2(_shfl_11, _shfl_11), make_float2(__uint_as_float(word_113 << 16), __uint_as_float(word_113 & 4294901760u)), make_float2(values_84[4], values_84[5]));
            values_84[4] = pair_114.x;
            values_84[5] = pair_114.y;
            unsigned int word_115_2 = loaded_79[15];
            float2 pair_116 = fma_f32x2_rn_noftz(make_float2(_shfl_11, _shfl_11), make_float2(__uint_as_float(word_115_2 << 16), __uint_as_float(word_115_2 & 4294901760u)), make_float2(values_84[6], values_84[7]));
            values_84[6] = pair_116.x;
            values_84[7] = pair_116.y;
            float2 w_pair_117 = make_float2(values_84[0], values_84[1]);
            float2 squares_118 = fma_f32x2_rn_noftz(w_pair_117, w_pair_117, make_float2(sums_2[0], sums_2[1]));
            sums_2[0] = squares_118.x;
            sums_2[1] = squares_118.y;
            float2 w_pair_119 = make_float2(values_84[2], values_84[3]);
            float2 squares_120 = fma_f32x2_rn_noftz(w_pair_119, w_pair_119, make_float2(sums_2[0], sums_2[1]));
            sums_2[0] = squares_120.x;
            sums_2[1] = squares_120.y;
            float2 w_pair_121 = make_float2(values_84[4], values_84[5]);
            float2 squares_122 = fma_f32x2_rn_noftz(w_pair_121, w_pair_121, make_float2(sums_2[0], sums_2[1]));
            sums_2[0] = squares_122.x;
            sums_2[1] = squares_122.y;
            float2 w_pair_123 = make_float2(values_84[6], values_84[7]);
            float2 squares_124 = fma_f32x2_rn_noftz(w_pair_123, w_pair_123, make_float2(sums_2[0], sums_2[1]));
            sums_2[0] = squares_124.x;
            sums_2[1] = squares_124.y;
            retained[8] = __as_u32(__float22bfloat162_rn(make_float2(values_84[0], values_84[1])));
            retained[9] = __as_u32(__float22bfloat162_rn(make_float2(values_84[2], values_84[3])));
            retained[10] = __as_u32(__float22bfloat162_rn(make_float2(values_84[4], values_84[5])));
            retained[11] = __as_u32(__float22bfloat162_rn(make_float2(values_84[6], values_84[7])));
            float values_125[8];
            float _shfl_12 = __shfl_sync(0xFFFFFFFF, coeff_2, 0);
            unsigned int word_126 = loaded_79[16];
            float2 _f32x2_mul_a_102 = make_float2(_shfl_12, _shfl_12);
            float2 _f32x2_mul_b_103 = make_float2(__uint_as_float(word_126 << 16), __uint_as_float(word_126 & 4294901760u));
            float2 pair_127;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_127) : "l"(*(const unsigned long long*)&_f32x2_mul_a_102), "l"(*(const unsigned long long*)&_f32x2_mul_b_103));
            values_125[0] = pair_127.x;
            values_125[1] = pair_127.y;
            unsigned int word_128 = loaded_79[17];
            float2 _f32x2_mul_a_104 = make_float2(_shfl_12, _shfl_12);
            float2 _f32x2_mul_b_105 = make_float2(__uint_as_float(word_128 << 16), __uint_as_float(word_128 & 4294901760u));
            float2 pair_129;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_129) : "l"(*(const unsigned long long*)&_f32x2_mul_a_104), "l"(*(const unsigned long long*)&_f32x2_mul_b_105));
            values_125[2] = pair_129.x;
            values_125[3] = pair_129.y;
            unsigned int word_130 = loaded_79[18];
            float2 _f32x2_mul_a_106 = make_float2(_shfl_12, _shfl_12);
            float2 _f32x2_mul_b_107 = make_float2(__uint_as_float(word_130 << 16), __uint_as_float(word_130 & 4294901760u));
            float2 pair_131;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_131) : "l"(*(const unsigned long long*)&_f32x2_mul_a_106), "l"(*(const unsigned long long*)&_f32x2_mul_b_107));
            values_125[4] = pair_131.x;
            values_125[5] = pair_131.y;
            unsigned int word_132_2 = loaded_79[19];
            float2 _f32x2_mul_a_108 = make_float2(_shfl_12, _shfl_12);
            float2 _f32x2_mul_b_109 = make_float2(__uint_as_float(word_132_2 << 16), __uint_as_float(word_132_2 & 4294901760u));
            float2 pair_133;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pair_133) : "l"(*(const unsigned long long*)&_f32x2_mul_a_108), "l"(*(const unsigned long long*)&_f32x2_mul_b_109));
            values_125[6] = pair_133.x;
            values_125[7] = pair_133.y;
            float _shfl_13 = __shfl_sync(0xFFFFFFFF, coeff_2, 1);
            unsigned int word_134_2 = loaded_79[20];
            float2 pair_135 = fma_f32x2_rn_noftz(make_float2(_shfl_13, _shfl_13), make_float2(__uint_as_float(word_134_2 << 16), __uint_as_float(word_134_2 & 4294901760u)), make_float2(values_125[0], values_125[1]));
            values_125[0] = pair_135.x;
            values_125[1] = pair_135.y;
            unsigned int word_136 = loaded_79[21];
            float2 pair_137 = fma_f32x2_rn_noftz(make_float2(_shfl_13, _shfl_13), make_float2(__uint_as_float(word_136 << 16), __uint_as_float(word_136 & 4294901760u)), make_float2(values_125[2], values_125[3]));
            values_125[2] = pair_137.x;
            values_125[3] = pair_137.y;
            unsigned int word_138 = loaded_79[22];
            float2 pair_139 = fma_f32x2_rn_noftz(make_float2(_shfl_13, _shfl_13), make_float2(__uint_as_float(word_138 << 16), __uint_as_float(word_138 & 4294901760u)), make_float2(values_125[4], values_125[5]));
            values_125[4] = pair_139.x;
            values_125[5] = pair_139.y;
            unsigned int word_140 = loaded_79[23];
            float2 pair_141 = fma_f32x2_rn_noftz(make_float2(_shfl_13, _shfl_13), make_float2(__uint_as_float(word_140 << 16), __uint_as_float(word_140 & 4294901760u)), make_float2(values_125[6], values_125[7]));
            values_125[6] = pair_141.x;
            values_125[7] = pair_141.y;
            float _shfl_14 = __shfl_sync(0xFFFFFFFF, coeff_2, 2);
            unsigned int word_142 = loaded_79[24];
            float2 pair_143 = fma_f32x2_rn_noftz(make_float2(_shfl_14, _shfl_14), make_float2(__uint_as_float(word_142 << 16), __uint_as_float(word_142 & 4294901760u)), make_float2(values_125[0], values_125[1]));
            values_125[0] = pair_143.x;
            values_125[1] = pair_143.y;
            unsigned int word_144 = loaded_79[25];
            float2 pair_145 = fma_f32x2_rn_noftz(make_float2(_shfl_14, _shfl_14), make_float2(__uint_as_float(word_144 << 16), __uint_as_float(word_144 & 4294901760u)), make_float2(values_125[2], values_125[3]));
            values_125[2] = pair_145.x;
            values_125[3] = pair_145.y;
            unsigned int word_146 = loaded_79[26];
            float2 pair_147 = fma_f32x2_rn_noftz(make_float2(_shfl_14, _shfl_14), make_float2(__uint_as_float(word_146 << 16), __uint_as_float(word_146 & 4294901760u)), make_float2(values_125[4], values_125[5]));
            values_125[4] = pair_147.x;
            values_125[5] = pair_147.y;
            unsigned int word_148 = loaded_79[27];
            float2 pair_149 = fma_f32x2_rn_noftz(make_float2(_shfl_14, _shfl_14), make_float2(__uint_as_float(word_148 << 16), __uint_as_float(word_148 & 4294901760u)), make_float2(values_125[6], values_125[7]));
            values_125[6] = pair_149.x;
            values_125[7] = pair_149.y;
            float _shfl_15 = __shfl_sync(0xFFFFFFFF, coeff_2, 3);
            unsigned int word_150 = loaded_79[28];
            float2 pair_151 = fma_f32x2_rn_noftz(make_float2(_shfl_15, _shfl_15), make_float2(__uint_as_float(word_150 << 16), __uint_as_float(word_150 & 4294901760u)), make_float2(values_125[0], values_125[1]));
            values_125[0] = pair_151.x;
            values_125[1] = pair_151.y;
            unsigned int word_152_2 = loaded_79[29];
            float2 pair_153 = fma_f32x2_rn_noftz(make_float2(_shfl_15, _shfl_15), make_float2(__uint_as_float(word_152_2 << 16), __uint_as_float(word_152_2 & 4294901760u)), make_float2(values_125[2], values_125[3]));
            values_125[2] = pair_153.x;
            values_125[3] = pair_153.y;
            unsigned int word_154 = loaded_79[30];
            float2 pair_155 = fma_f32x2_rn_noftz(make_float2(_shfl_15, _shfl_15), make_float2(__uint_as_float(word_154 << 16), __uint_as_float(word_154 & 4294901760u)), make_float2(values_125[4], values_125[5]));
            values_125[4] = pair_155.x;
            values_125[5] = pair_155.y;
            unsigned int word_156 = loaded_79[31];
            float2 pair_157 = fma_f32x2_rn_noftz(make_float2(_shfl_15, _shfl_15), make_float2(__uint_as_float(word_156 << 16), __uint_as_float(word_156 & 4294901760u)), make_float2(values_125[6], values_125[7]));
            values_125[6] = pair_157.x;
            values_125[7] = pair_157.y;
            float2 w_pair_158 = make_float2(values_125[0], values_125[1]);
            float2 squares_159 = fma_f32x2_rn_noftz(w_pair_158, w_pair_158, make_float2(sums_2[0], sums_2[1]));
            sums_2[0] = squares_159.x;
            sums_2[1] = squares_159.y;
            float2 w_pair_160 = make_float2(values_125[2], values_125[3]);
            float2 squares_161 = fma_f32x2_rn_noftz(w_pair_160, w_pair_160, make_float2(sums_2[0], sums_2[1]));
            sums_2[0] = squares_161.x;
            sums_2[1] = squares_161.y;
            float2 w_pair_162 = make_float2(values_125[4], values_125[5]);
            float2 squares_163_2 = fma_f32x2_rn_noftz(w_pair_162, w_pair_162, make_float2(sums_2[0], sums_2[1]));
            sums_2[0] = squares_163_2.x;
            sums_2[1] = squares_163_2.y;
            float2 w_pair_164 = make_float2(values_125[6], values_125[7]);
            float2 squares_165 = fma_f32x2_rn_noftz(w_pair_164, w_pair_164, make_float2(sums_2[0], sums_2[1]));
            sums_2[0] = squares_165.x;
            sums_2[1] = squares_165.y;
            retained[12] = __as_u32(__float22bfloat162_rn(make_float2(values_125[0], values_125[1])));
            retained[13] = __as_u32(__float22bfloat162_rn(make_float2(values_125[2], values_125[3])));
            retained[14] = __as_u32(__float22bfloat162_rn(make_float2(values_125[4], values_125[5])));
            retained[15] = __as_u32(__float22bfloat162_rn(make_float2(values_125[6], values_125[7])));
            float _add_110;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_110) : "f"(sums_2[0]), "f"(sums_2[1]));
            float _warp_reduce_0 = _add_110;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
            if (lane_in_warp == 0) {
                tail_scratch[warp] = _warp_reduce_0;
            }
            asm volatile("barrier.sync %0, 128;" :: "r"(9 + warp / 4) : "memory");
            float row_sum = ((lane_in_warp < 4) ? tail_scratch[warp / 4 * 4 + lane_in_warp] : 0.0f);
            float _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, row_sum, 2);
            float _add_47_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_47_2) : "f"(row_sum), "f"(_shfl_xor_sync_111));
            float _shfl_xor_sync_113 = __shfl_xor_sync(0xFFFFFFFF, _add_47_2, 1);
            float _add_112;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(_add_47_2), "f"(_shfl_xor_sync_113));
            row_sum = _add_112;
            float _shfl_sync_114 = __shfl_sync(0xFFFFFFFF, row_sum, 0);
            row_sum = _shfl_sync_114;
            float _fma_117;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_117) : "f"(row_sum), "f"(0.000244140625f), "f"(rmsnorm_eps));
            float _sqrt_116;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_116) : "f"(_fma_117));
            float _rcp_115 = __frcp_rn(_sqrt_116);
            float rstd;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(rstd) : "f"(_rcp_115), "f"(rmsnorm_scale));
            unsigned int first_hidden_pack = warp % 4 * 4;
            unsigned int weight_cache[4];
            {
                const uint4* _vptr_118 = reinterpret_cast<const uint4*>(rmsnorm_weight + (first_hidden_pack * 256 + lane_in_warp * 8) + 0);
                uint4* _vdst_118 = reinterpret_cast<uint4*>(&weight_cache[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_118[_blk] = _vptr_118[_blk];
                }
            }
            #pragma unroll
            for (int pack = 0; pack < 4; pack++) {
                unsigned int hidden_pack_1 = warp % 4 * 4 + (unsigned int)pack;
                unsigned int next_weight[4];
                if (pack + 1 < 4) {
                    {
                        const uint4* _vptr_119 = reinterpret_cast<const uint4*>(rmsnorm_weight + ((hidden_pack_1 + 1) * 256 + lane_in_warp * 8) + 0);
                        uint4* _vdst_119 = reinterpret_cast<uint4*>(&next_weight[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_119[_blk] = _vptr_119[_blk];
                        }
                    }
                }
                unsigned int normalized_words[4];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    unsigned int word_0_2 = retained[pack * 4 + j];
                    float lo = __uint_as_float(word_0_2 << 16);
                    float hi = __uint_as_float(word_0_2 & 4294901760u);
                    float weight_lo = __uint_as_float(weight_cache[j] << 16);
                    float weight_hi = __uint_as_float(weight_cache[j] & 4294901760u);
                    float2 _f32x2_mul_a_120 = make_float2(lo, hi);
                    float2 _f32x2_mul_b_121 = make_float2(rstd, rstd);
                    float2 scaled;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled) : "l"(*(const unsigned long long*)&_f32x2_mul_a_120), "l"(*(const unsigned long long*)&_f32x2_mul_b_121));
                    float2 _f32x2_mul_b_122 = make_float2(weight_lo, weight_hi);
                    float2 weighted;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&weighted) : "l"(*(const unsigned long long*)&scaled), "l"(*(const unsigned long long*)&_f32x2_mul_b_122));
                    normalized_words[j] = __as_u32(__float22bfloat162_rn(make_float2(weighted.x, weighted.y)));
                }
                unsigned long long off_2 = (unsigned long long)token_1 * 4096 + (unsigned long long)(hidden_pack_1 * 256) + (unsigned long long)(lane_in_warp * 8);
                {
                    int4 _iv4 = make_int4(normalized_words[0 + 0], normalized_words[0 + 1], normalized_words[0 + 2], normalized_words[0 + 3]);
                    *reinterpret_cast<int4*>(y_bf16 + off_2 + 0) = _iv4;
                }
                if (pack + 1 < 4) {
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 4; j_1++) {
                        weight_cache[j_1] = next_weight[j_1];
                    }
                }
            }
        }
        if (token_1 >= num_tokens) {
            break;
        }
    }
    asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
    asm volatile("tcgen05.fence::before_thread_sync;");
    asm volatile("barrier.sync 0, 768;" ::: "memory");
    if (warp == 0) {
        if (elect_sync()) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
            asm volatile("red.async.release.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(launch_epochs) + (bid))), "l"(static_cast<unsigned long long>(1)) : "memory");
            #elif defined(__CUDA_ARCH__)
            #error "GlobalRedAsyncReleaseAdd requires SM100 or newer"
            #endif
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 3) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
