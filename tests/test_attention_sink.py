import math

import einops
import pytest
import torch

import flashinfer
from flashinfer.jit.utils import filename_safe_dtype_map

attention_sink_decl = r"""
struct AttentionSink : AttentionVariantBase {
  static constexpr bool use_softmax = true;

  uint32_t window_left, qo_len, kv_len;
  float sm_scale_log2;

  // Create closure
  template <typename Params>
  __device__ __host__ AttentionSink(const Params& params, uint32_t batch_idx,
                                   uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
    sm_scale_log2 = params.sm_scale * math::log2e;
  }

  REGISTER_OUTPUT_TRANSFORM(params, output, batch_idx, qo_idx, qo_head_idx, m, d, {
    float log_sink = math::ptx_log2(params.sink[qo_head_idx]);
    float m_all = (log_sink > m) ? log_sink : m;
    float scale = math::ptx_exp2(m - m_all);
    float d_all = d * scale;
    float denom = math::ptx_exp2(log_sink - m_all) + d_all;
    return (output * scale) * math::ptx_rcp(denom);
  });
};
"""


def sink_softmax(logits, sink):
    sink = einops.repeat(sink, "h -> b h m 1", b=logits.shape[0], m=logits.shape[2])
    # (b, h, m, (n + 1))
    logits = torch.cat([logits, torch.log(sink)], dim=-1)
    # (s_1, s_2, ..., s_n)
    # (s_1, s_2, ..., s_n, log(sink))
    # (exp(s_1), exp(s_2), ..., exp(s_n), sink)
    # (exp(s_1) / (exp(s_1) + exp(s_2) + ... + exp(s_n) + sink),
    #  exp(s_2) / (exp(s_1) + exp(s_2) + ... + exp(s_n) + sink),
    #  ...,
    #  exp(s_n) / (exp(s_1) + exp(s_2) + ... + exp(s_n) + sink))
    #  sink / (exp(s_1) + exp(s_2) + ... + exp(s_n) + sink)
    score = torch.softmax(logits, dim=-1)[..., :-1].contiguous()
    return score


def sink_attention_ref(
    batch_size,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    qo_len = q.shape[0] // batch_size
    kv_len = k.shape[0] // batch_size
    num_qo_heads = q.shape[1]
    head_dim_qk = q.shape[2]
    head_dim_vo = v.shape[2]
    logits = (
        torch.einsum(
            "bmhd,bnhd->bhmn",
            q.view(batch_size, qo_len, num_qo_heads, head_dim_qk).float(),
            k.view(batch_size, kv_len, num_qo_heads, head_dim_qk).float(),
        )
        * sm_scale
    )

    if causal:
        mask = torch.arange(kv_len - qo_len, kv_len, device=q.device).unsqueeze(
            1
        ) >= torch.arange(0, kv_len, device=q.device).unsqueeze(0)
    else:
        mask = torch.ones(qo_len, kv_len, device=q.device)

    logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))

    p = sink_softmax(logits, sink)
    o_ref = (
        torch.einsum(
            "bhmn,bnhd->bmhd",
            p,
            v.view(batch_size, kv_len, num_qo_heads, head_dim_vo).float(),
        )
        .contiguous()
        .view(batch_size * qo_len, num_qo_heads, head_dim_vo)
        .to(q)
    )

    return o_ref


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # , torch.bfloat16])
@pytest.mark.parametrize("causal", [False])  # [True, False])
def test_attention_sink(dtype, causal):
    jit_args = (
        f"batch_prefill_attention_sink_{filename_safe_dtype_map[dtype]}",  # uri
        dtype,  # dtype_q
        dtype,  # dtype_kv
        dtype,  # dtype_o
        torch.int32,  # idtype
        128,  # hidden_dim_qk
        128,  # hidden_dim_vo
        ["sink"],  # additional_tensor_names
        ["float"],  # additional_tensor_dtypes
        ["sm_scale"],  # additional_scalar_names
        ["double"],  # additional_scalar_dtypes
        "AttentionSink",
        attention_sink_decl,
    )
    sm_scale = 1.0 / math.sqrt(128)
    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
    )
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        float_workspace_buffer, kv_layout="NHD", backend="fa2", jit_args=jit_args
    )
    batch_size = 128
    seq_len_per_request = 1024
    qo_indptr_host = torch.arange(
        0, batch_size * seq_len_per_request + 1, seq_len_per_request, dtype=torch.int32
    )
    kv_indptr_host = torch.arange(
        0, batch_size * seq_len_per_request + 1, seq_len_per_request, dtype=torch.int32
    )

    num_qo_heads = 32
    num_kv_heads = 32
    head_dim = 128

    wrapper.plan(
        qo_indptr_host,
        kv_indptr_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    q = torch.randn(
        batch_size * seq_len_per_request,
        num_qo_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
    )
    k = torch.randn(
        batch_size * seq_len_per_request,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
    )
    v = torch.randn(
        batch_size * seq_len_per_request,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
    )

    sink = torch.rand(num_qo_heads, device="cuda", dtype=torch.float32) * 100

    o = wrapper.run(q, k, v, sink, sm_scale)
    o_ref = sink_attention_ref(
        batch_size, q, k, v, sink, causal=causal, sm_scale=sm_scale
    )
    if dtype == torch.float16:
        torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)

    wrapper_paged = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        float_workspace_buffer, kv_layout="NHD", backend="fa2", jit_args=jit_args
    )
    kv_indices_host = torch.arange(
        0,
        batch_size * seq_len_per_request,
        dtype=torch.int32,
    )
    paged_kv_last_page_len_host = torch.full((batch_size,), 1, dtype=torch.int32)
    wrapper_paged.plan(
        qo_indptr_host,
        kv_indptr_host,
        kv_indices_host,
        paged_kv_last_page_len_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        1,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o_paged = wrapper_paged.run(q, (k, v), sink, sm_scale)
    if dtype == torch.float16:
        torch.testing.assert_close(o_paged, o_ref, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(o_paged, o_ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_attention_sink(torch.float16, True)
