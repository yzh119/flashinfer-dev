/*
 * Copyright (c) 2023 by FlashInfer team.
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
#ifndef FLASHINFER_MATH_CUH_
#define FLASHINFER_MATH_CUH_

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cute/arch/simd_sm100.hpp>

namespace flashinfer {
namespace math {

// log2(e)
constexpr float log2e = 1.44269504088896340736f;

constexpr float loge2 = 0.693147180559945309417f;

constexpr float inf = 5e4;

__forceinline__ __device__ half2 uint32_as_half2(uint32_t x) { return *(half2*)&x; }

__forceinline__ __device__ uint32_t half2_as_uint32(half2 x) { return *(uint32_t*)&x; }

/*!
 * \brief Wrapper of PTX ex2.approx instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX lg2.approx instruction, which computes log2(x)
 * \param x input
 */
__forceinline__ __device__ float ptx_log2(float x) {
  float y;
  asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX ex2.approx.f16x2 instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ half2 ptx_exp2(half2 x) {
  uint32_t y_u32;
  uint32_t x_u32 = half2_as_uint32(x);
  asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
  return uint32_as_half2(y_u32);
}

/*!
 * \brief Wrapper of PTX ex2.approx.f16 instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ half ptx_exp2(half x) {
  ushort y_u16;
  asm volatile("ex2.approx.f16 %0, %1;" : "=h"(y_u16) : "h"(__half_as_ushort(x)));
  return __ushort_as_half(y_u16);
}

/*!
 * \brief Wrapper of PTX rcp.approx instruction, which computes 1/x
 * \param x input
 */
__forceinline__ __device__ float ptx_rcp(float x) {
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction, which performs a butterfly shuffle
 *   between threads in a warp.
 * \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with: y[i] <- x[i ^ delta]
 */
__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
  float y;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
               : "=f"(y)
               : "f"(x), "r"(lane_mask));
  return y;
}

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction on half2, which performs a butterfly
 *   shuffle between threads in a warp.
 * \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with: y[i] <- x[i ^ lane_mask]
 */
__forceinline__ __device__ half2 shfl_xor_sync(half2 x, int lane_mask) {
  return __shfl_xor_sync(0xffffffff, x, lane_mask);
}

/*!
 * \brief Wrapper of PTX rsqrt approximation instruction, which computes 1/sqrt(x)
 * \param x input
 */
__forceinline__ __device__ float rsqrt(float x) {
  float y;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX tanh.approx.f32 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ float tanh(float x) {
  float y;
  asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX tanh.approx.f16x2 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ half2 tanh(half2 x) {
  uint32_t y_u32;
  uint32_t x_u32 = half2_as_uint32(x);
  asm volatile("tanh.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
  return uint32_as_half2(y_u32);
}

/*!
 * \brief Wrapper of PTX tanh.approx.f16 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ half tanh(half x) {
  ushort y_u16;
  asm volatile("tanh.approx.f16 %0, %1;" : "=h"(y_u16) : "h"(__half_as_ushort(x)));
  return __ushort_as_half(y_u16);
}

#include <cuda_runtime.h>

#include <cute/arch/simd_sm100.hpp>

using namespace cute;

// constexpr float poly_ex2_deg3[4] = {
//     1.0f,
//     0.695146143436431884765625f,
//     0.227564394474029541015625f,
//     0.077119089663028717041015625f
// };

// Evaluate polynomial for single float with template degree
template <int DEG>
__device__ __forceinline__ float evaluate_polynomial(
    float x) {  //, constexpr float (&poly)[DEG+1]) {
  constexpr float poly_ex2_deg3[4] = {1.0f, 0.695146143436431884765625f,
                                      0.227564394474029541015625f, 0.077119089663028717041015625f};
  float out = poly_ex2_deg3[DEG];
#pragma unroll
  for (int i = DEG - 1; i >= 0; i--) {
    out = out * x + poly_ex2_deg3[i];
  }
  return out;
}

// Evaluate polynomial for float2 (vectorized version) with template degree
template <int DEG>
__device__ __forceinline__ float2
evaluate_polynomial_2(float2 xy) {  //}, constexpr float (&poly)[DEG+1]) {
  constexpr float poly_ex2_deg3[4] = {1.0f, 0.695146143436431884765625f,
                                      0.227564394474029541015625f, 0.077119089663028717041015625f};
  float2 out = make_float2(poly_ex2_deg3[DEG], poly_ex2_deg3[DEG]);
#pragma unroll
  for (int i = DEG - 1; i >= 0; i--) {
    float2 poly_broadcast = make_float2(poly_ex2_deg3[i], poly_ex2_deg3[i]);
    float2 temp;
    cute::fma(temp, out, xy, poly_broadcast);
    out = temp;
  }
  return out;
}

// Add with round down mode using PTX inline assembly
__device__ __forceinline__ float add_round_down(float x, float y) {
  float result;
  asm volatile("add.rm.ftz.f32 %0, %1, %2;" : "=f"(result) : "f"(x), "f"(y));
  return result;
}

// Combine integer and fractional parts for ex2 computation
__device__ __forceinline__ float combine_int_frac_ex2(float x_rounded, float frac_ex2) {
  float result;
  asm volatile(
      "{\n\t"
      ".reg .s32 x_rounded_i, frac_ex_i, x_rounded_e, out_i;\n\t"
      "mov.b32 x_rounded_i, %1;\n\t"
      "mov.b32 frac_ex_i, %2;\n\t"
      "shl.b32 x_rounded_e, x_rounded_i, 23;\n\t"
      "add.s32 out_i, x_rounded_e, frac_ex_i;\n\t"
      "mov.b32 %0, out_i;\n\t"
      "}\n"
      : "=f"(result)
      : "f"(x_rounded), "f"(frac_ex2));
  return result;
}

// Single float ex2 emulation
__device__ __forceinline__ float ex2_emulation(float x) {
  // Polynomial coefficients for ex2 approximation (degree 3)
  constexpr float poly_ex2_deg3[4] = {1.0f, 0.695146143436431884765625f,
                                      0.227564394474029541015625f, 0.077119089663028717041015625f};

  const float fp32_round_int = float(1 << 23) + float(1 << 22);  // 2^23 + 2^22

  // Clamp x to prevent overflow
  float x_clamped = fmaxf(x, -127.0f);

  // Round down to get integer part
  float x_rounded = add_round_down(x_clamped, fp32_round_int);

  // Extract fractional part
  float x_rounded_back = x_rounded - fp32_round_int;
  float x_frac = x_clamped - x_rounded_back;

  // Evaluate polynomial on fractional part
  float x_frac_ex2 = evaluate_polynomial<3>(x_frac);  //, poly_ex2_deg3);

  // Combine integer and fractional parts
  return combine_int_frac_ex2(x_rounded, x_frac_ex2);
}

// Vectorized float2 ex2 emulation using cute SIMD operations
__device__ __forceinline__ float2 ex2_emulation_2(float2 xy) {
  // Polynomial coefficients for ex2 approximation (degree 3)

  const float fp32_round_int = float(1 << 23) + float(1 << 22);  // 2^23 + 2^22
  const float2 fp32_round_int_vec = make_float2(fp32_round_int, fp32_round_int);
  const float2 clamp_val = make_float2(-127.0f, -127.0f);

  // Clamp x and y to prevent overflow
  float2 xy_clamped = make_float2(fmaxf(xy.x, -127.0f), fmaxf(xy.y, -127.0f));

  // Round down using cute SIMD operations with round-down mode
  float2 xy_rounded;
  // Note: cute::add with rounding mode might need specific implementation
  // For now, using component-wise round down
  xy_rounded.x = add_round_down(xy_clamped.x, fp32_round_int);
  xy_rounded.y = add_round_down(xy_clamped.y, fp32_round_int);

  // Extract fractional parts using cute SIMD operations
  float2 xy_rounded_back;
  float2 neg_round_int = make_float2(-fp32_round_int, -fp32_round_int);
  cute::add(xy_rounded_back, xy_rounded, neg_round_int);

  float2 xy_frac;
  float2 neg_rounded_back = make_float2(-xy_rounded_back.x, -xy_rounded_back.y);
  cute::add(xy_frac, xy_clamped, neg_rounded_back);

  // Evaluate polynomial on fractional parts (vectorized)
  float2 xy_frac_ex2 = evaluate_polynomial_2<3>(xy_frac);  //, poly_ex2_deg3);

  // Combine integer and fractional parts for both components
  float x_out = combine_int_frac_ex2(xy_rounded.x, xy_frac_ex2.x);
  float y_out = combine_int_frac_ex2(xy_rounded.y, xy_frac_ex2.y);

  return make_float2(x_out, y_out);
}

}  // namespace math
}  // namespace flashinfer
#endif  // FLASHINFER_MATH_CUH_
