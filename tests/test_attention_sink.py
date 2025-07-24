import math

import einops
import pytest
import torch

import flashinfer
from flashinfer.jit.utils import filename_safe_dtype_map
from typing import Optional

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
    window_left = (params.window_left >= 0) ? params.window_left : kv_len;
    sm_scale_log2 = params.sm_scale * math::log2e;
  }

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return (kv_idx + qo_len + window_left >= kv_len + qo_idx);
  })

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


def sink_attention_unified(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    window_left: int,
    causal: bool,
    sm_scale: float,
    batch_size: Optional[int] = None,
    mode: str = "auto",
    qo_indptr: Optional[torch.Tensor] = None,
    kv_indptr: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Unified sink attention implementation supporting prefill, incremental, chunk prefill, and variable-length scenarios.

    Args:
        q: Query tensor. Format depends on mode:
           - Regular Prefill: [total_q_len, num_qo_heads, head_dim] where q_len == kv_len
           - Incremental: [batch_size, num_qo_heads, head_dim] where q_len == 1
           - Chunk Prefill: [total_q_len, num_qo_heads, head_dim] where q_len != kv_len and q_len > 1
           - Variable Length: [total_q_len, num_qo_heads, head_dim] with different q_len per request
        k: Key tensor. Format depends on mode:
           - Regular Prefill: [total_kv_len, num_kv_heads, head_dim]
           - Incremental: [batch_size, kv_len, num_kv_heads, head_dim]
           - Chunk Prefill: [total_kv_len, num_kv_heads, head_dim]
           - Variable Length: [total_kv_len, num_kv_heads, head_dim]
        v: Value tensor, same format as k
        sink: Sink values [num_qo_heads]
        window_left: Sliding window size (-1 for no window)
        causal: Whether to apply causal masking
        sm_scale: Scaling factor for attention
        batch_size: Required for prefill/chunk modes, auto-detected for incremental
        mode: Processing mode:
            - "auto": Auto-detect based on tensor shapes and dimensions
            - "prefill": Regular prefill (q_len == kv_len)
            - "incremental": Incremental generation (q_len == 1)
            - "chunk": Chunk prefill (q_len != kv_len and q_len > 1)
            - "varlen": Variable length sequences within batch
        qo_indptr: Optional[torch.Tensor] - Query sequence length pointers for variable length mode.
                  Shape: [batch_size + 1]. qo_indptr[i+1] - qo_indptr[i] gives the query length for request i.
                  Only used when mode="varlen".
        kv_indptr: Optional[torch.Tensor] - Key/Value sequence length pointers for variable length mode.
                  Shape: [batch_size + 1]. kv_indptr[i+1] - kv_indptr[i] gives the kv length for request i.
                  Only used when mode="varlen".

    Returns:
        Output tensor. Format depends on mode:
        - Regular Prefill: [total_q_len, num_qo_heads, head_dim]
        - Incremental: [batch_size, num_qo_heads, head_dim]
        - Chunk Prefill: [total_q_len, num_qo_heads, head_dim]
        - Variable Length: [total_q_len, num_qo_heads, head_dim]
    """

    # Auto-detect mode if not specified
    if mode == "auto":
        # Check if variable length mode is indicated by presence of indptr
        if qo_indptr is not None or kv_indptr is not None:
            mode = "varlen"
        elif len(q.shape) == 3 and len(k.shape) == 4:
            # q: [batch_size, num_heads, head_dim], k: [batch_size, kv_len, num_heads, head_dim]
            # This is incremental mode
            mode = "incremental"
        elif len(q.shape) == 3 and len(k.shape) == 3:
            # Both q and k are flattened: [total_len, num_heads, head_dim]
            if batch_size is None:
                raise ValueError("batch_size is required for auto-detection in prefill/chunk modes")

            qo_len = q.shape[0] // batch_size
            kv_len = k.shape[0] // batch_size

            if qo_len == kv_len:
                mode = "prefill"
            elif qo_len == 1:
                mode = "incremental"  # Special case: single token with flattened format
            elif qo_len > 1 and qo_len != kv_len:
                mode = "chunk"
            else:
                raise ValueError(f"Cannot auto-detect mode: qo_len={qo_len}, kv_len={kv_len}")
        else:
            raise ValueError(f"Cannot auto-detect mode from tensor shapes: q={q.shape}, k={k.shape}")

    # Process based on detected/specified mode
    if mode == "incremental":
        # Incremental generation mode: q_len=1, kv_len from cache
        batch_size = q.shape[0]
        qo_len = 1
        kv_len = k.shape[1]
        num_qo_heads = q.shape[1]
        num_kv_heads = k.shape[2]

        # Handle GQA
        if num_qo_heads != num_kv_heads:
            k = torch.repeat_interleave(k, num_qo_heads // num_kv_heads, dim=2)
            v = torch.repeat_interleave(v, num_qo_heads // num_kv_heads, dim=2)
            num_kv_heads = num_qo_heads

        head_dim_qk = q.shape[2]
        head_dim_vo = v.shape[3]

        # Compute logits: [batch_size, num_heads, 1, kv_len]
        logits = (
            torch.einsum(
                "bhd,blhd->bhl",
                q.float(),
                k.float(),
            ).unsqueeze(2)  # Add seq_len=1 dimension
            * sm_scale
        )

    elif mode in ["prefill", "chunk"]:
        # Prefill or Chunk prefill mode: q and k are flattened tensors
        if batch_size is None:
            raise ValueError(f"batch_size is required for {mode} mode")

        qo_len = q.shape[0] // batch_size
        kv_len = k.shape[0] // batch_size
        num_qo_heads = q.shape[1]
        num_kv_heads = k.shape[1]

        # Handle GQA
        if num_qo_heads != num_kv_heads:
            k = torch.repeat_interleave(k, num_qo_heads // num_kv_heads, dim=1)
            v = torch.repeat_interleave(v, num_qo_heads // num_kv_heads, dim=1)

        head_dim_qk = q.shape[2]
        head_dim_vo = v.shape[2]

        # Compute logits: [batch_size, qo_len, num_heads, kv_len]
        logits = (
            torch.einsum(
                "bmhd,bnhd->bhmn",
                q.view(batch_size, qo_len, num_qo_heads, head_dim_qk).float(),
                k.view(batch_size, kv_len, num_qo_heads, head_dim_qk).float(),
            )
            * sm_scale
        )

    elif mode == "varlen":
        # Variable length sequences mode
        if qo_indptr is None or kv_indptr is None:
            raise ValueError("qo_indptr and kv_indptr are required for varlen mode")

        batch_size = qo_indptr.shape[0] - 1
        num_qo_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        head_dim_qk = q.shape[2]
        head_dim_vo = v.shape[2]

        # Handle GQA
        if num_qo_heads != num_kv_heads:
            k = torch.repeat_interleave(k, num_qo_heads // num_kv_heads, dim=1).contiguous()
            v = torch.repeat_interleave(v, num_qo_heads // num_kv_heads, dim=1).contiguous()
            num_kv_heads = num_qo_heads

        # Process each request in the batch separately
        output_list = []

        for i in range(batch_size):
            # Extract tensors for current request
            qo_start, qo_end = qo_indptr[i].item(), qo_indptr[i + 1].item()
            kv_start, kv_end = kv_indptr[i].item(), kv_indptr[i + 1].item()

            q_i = q[qo_start:qo_end]  # [qo_len_i, num_heads, head_dim]
            k_i = k[kv_start:kv_end]  # [kv_len_i, num_heads, head_dim]
            v_i = v[kv_start:kv_end]  # [kv_len_i, num_heads, head_dim]

            qo_len_i = qo_end - qo_start
            kv_len_i = kv_end - kv_start

            # Compute logits for current request: [1, num_heads, qo_len_i, kv_len_i]
            logits_i = (
                torch.einsum(
                    "qhd,khd->hqk",
                    q_i.float(),
                    k_i.float(),
                ).unsqueeze(0)  # Add batch dimension
                * sm_scale
            )

            # Build attention mask for current request
            if causal:
                # Create causal mask for this specific request
                row_idx = torch.arange(qo_len_i, dtype=torch.int32, device=q.device)[:, None]
                col_idx = torch.arange(kv_len_i, dtype=torch.int32, device=q.device)[None, :]

                # Default causal mask: position i can attend to positions 0 to i in the kv sequence
                # Assuming queries correspond to the last qo_len_i positions in the kv sequence
                query_positions = kv_len_i - qo_len_i + row_idx
                mask_i = query_positions >= col_idx

                if window_left >= 0:
                    mask_i &= query_positions - window_left <= col_idx
            else:
                # Non-causal mask
                mask_i = torch.ones(qo_len_i, kv_len_i, device=q.device, dtype=torch.bool)
                if window_left >= 0:
                    row_idx = torch.arange(qo_len_i, dtype=torch.int32, device=q.device)[:, None]
                    col_idx = torch.arange(kv_len_i, dtype=torch.int32, device=q.device)[None, :]
                    query_positions = kv_len_i - qo_len_i + row_idx
                    mask_i = query_positions - window_left <= col_idx

            # Apply mask
            logits_i = logits_i.masked_fill(mask_i.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))

            # Apply sink softmax
            p_i = sink_softmax(logits_i, sink)  # [1, num_heads, qo_len_i, kv_len_i]

            # Compute output for current request
            o_i = (
                torch.einsum(
                    "bhmn,nhd->bmhd",
                    p_i,  # [1, num_heads, qo_len_i, kv_len_i]
                    v_i.float(),  # [kv_len_i, num_heads, head_dim]
                )
                .contiguous()
                .view(qo_len_i, num_qo_heads, head_dim_vo)
                .to(q)
            )

            output_list.append(o_i)

        # Concatenate outputs from all requests
        o_ref = torch.cat(output_list, dim=0)

        return o_ref

    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes: 'auto', 'prefill', 'incremental', 'chunk', 'varlen'")

    # Build attention mask (unified for all modes)
    if causal:
        if mode == "incremental":
            # For incremental: new token can attend to all previous tokens
            mask = torch.ones(1, kv_len, device=q.device, dtype=torch.bool)
            if window_left >= 0:
                col_idx = torch.arange(kv_len, dtype=torch.int32, device=q.device)
                mask = (kv_len - 1 - window_left) <= col_idx
        elif mode == "prefill":
            # For regular prefill: standard causal mask
            mask = torch.arange(kv_len - qo_len, kv_len, device=q.device).unsqueeze(1) >= \
                   torch.arange(0, kv_len, device=q.device).unsqueeze(0)
            if window_left >= 0:
                row_idx = torch.arange(qo_len, dtype=torch.int32, device=q.device)[:, None]
                col_idx = torch.arange(kv_len, dtype=torch.int32, device=q.device)[None, :]
                mask &= row_idx - window_left <= col_idx
        elif mode == "chunk":
            # For chunk prefill: each query position can attend to all previous KV positions
            # Current chunk positions are at the end: [kv_len - qo_len : kv_len]
            current_chunk_start = kv_len - qo_len
            row_idx = torch.arange(qo_len, dtype=torch.int32, device=q.device)[:, None]  # Positions within chunk
            col_idx = torch.arange(kv_len, dtype=torch.int32, device=q.device)[None, :]  # All KV positions

            # Each position can attend to: all historical + positions up to itself in current chunk
            abs_row_positions = current_chunk_start + row_idx  # Absolute positions in full sequence
            mask = abs_row_positions >= col_idx  # Standard causal mask

            if window_left >= 0:
                mask &= abs_row_positions - window_left <= col_idx
    else:
        # Non-causal mask
        if mode == "incremental":
            mask = torch.ones(1, kv_len, device=q.device, dtype=torch.bool)
            if window_left >= 0:
                col_idx = torch.arange(kv_len, dtype=torch.int32, device=q.device)
                mask = (kv_len - 1 - window_left) <= col_idx
        else:  # prefill or chunk
            mask = torch.ones(qo_len, kv_len, device=q.device, dtype=torch.bool)
            if window_left >= 0:
                if mode == "chunk":
                    # For chunk mode, apply window relative to absolute positions
                    current_chunk_start = kv_len - qo_len
                    row_idx = torch.arange(qo_len, dtype=torch.int32, device=q.device)[:, None]
                    col_idx = torch.arange(kv_len, dtype=torch.int32, device=q.device)[None, :]
                    abs_row_positions = current_chunk_start + row_idx
                    mask = abs_row_positions - window_left <= col_idx
                else:  # prefill
                    row_idx = torch.arange(qo_len, dtype=torch.int32, device=q.device)[:, None]
                    col_idx = torch.arange(kv_len, dtype=torch.int32, device=q.device)[None, :]
                    mask = row_idx - window_left <= col_idx

    # Apply mask
    logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))

    # Apply sink softmax
    p = sink_softmax(logits, sink)

    # Compute output
    if mode == "incremental":
        # Incremental mode output
        o_ref = (
            torch.einsum(
                "bhml,blhd->bhd",
                p,  # [batch_size, num_heads, 1, kv_len]
                v.float(),  # [batch_size, kv_len, num_heads, head_dim]
            )
            .contiguous()
            .to(q)
        )
    else:  # prefill or chunk mode
        # Prefill/Chunk mode output
        o_ref = (
            torch.einsum(
                "bhmn,bnhd->bmhd",
                p,  # [batch_size, num_heads, qo_len, kv_len]
                v.view(batch_size, kv_len, num_qo_heads, head_dim_vo).float(),
            )
            .contiguous()
            .view(batch_size * qo_len, num_qo_heads, head_dim_vo)
            .to(q)
        )

    return o_ref


# Wrapper functions for backward compatibility
def sink_attention_ref(
    batch_size: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    window_left: int,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    """Backward compatible wrapper for prefill mode."""
    return sink_attention_unified(
        q, k, v, sink, window_left, causal, sm_scale,
        batch_size=batch_size, mode="prefill"
    )


def sink_attention_incremental_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sink: torch.Tensor,
    window_left: int,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    """Backward compatible wrapper for incremental mode."""
    return sink_attention_unified(
        q, k_cache, v_cache, sink, window_left, causal, sm_scale,
        mode="incremental"
    )


def sink_attention_chunk_ref(
    batch_size: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    window_left: int,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    """Wrapper for chunk prefill mode."""
    return sink_attention_unified(
        q, k, v, sink, window_left, causal, sm_scale,
        batch_size=batch_size, mode="chunk"
    )


def sink_attention_varlen_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    window_left: int,
    causal: bool,
    sm_scale: float,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
) -> torch.Tensor:
    """Wrapper for variable length sequences mode."""
    return sink_attention_unified(
        q, k, v, sink, window_left, causal, sm_scale,
        mode="varlen", qo_indptr=qo_indptr, kv_indptr=kv_indptr
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # , torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 4, 16, 128])
@pytest.mark.parametrize("seq_len", [1, 4, 16, 128, 1024])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("window_left", [-1, 127, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_attention_sink(
    dtype, batch_size, seq_len, num_qo_heads, num_kv_heads, window_left, causal
):
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
    qo_indptr_host = torch.arange(
        0, batch_size * seq_len + 1, seq_len, dtype=torch.int32
    )
    kv_indptr_host = torch.arange(
        0, batch_size * seq_len + 1, seq_len, dtype=torch.int32
    )

    head_dim = 128

    wrapper.plan(
        qo_indptr_host,
        kv_indptr_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        window_left=window_left,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    q = torch.randn(
        batch_size * seq_len,
        num_qo_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
    )
    k = torch.randn(
        batch_size * seq_len,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
    )
    v = torch.randn(
        batch_size * seq_len,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
    )

    sink = torch.rand(num_qo_heads, device="cuda", dtype=torch.float32) * 100

    o = wrapper.run(q, k, v, sink, sm_scale)
    o_ref = sink_attention_ref(
        batch_size, q, k, v, sink, window_left, causal, sm_scale
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
        batch_size * seq_len,
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
        window_left=window_left,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o_paged = wrapper_paged.run(q, (k, v), sink, sm_scale)
    if dtype == torch.float16:
        torch.testing.assert_close(o_paged, o_ref, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(o_paged, o_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("initial_seq_len", [32, 128, 512])
@pytest.mark.parametrize("num_generation_steps", [1, 2, 4])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("window_left", [-1, 127, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_attention_sink_incremental_generation(
    dtype, batch_size, initial_seq_len, num_generation_steps,
    num_qo_heads, num_kv_heads, window_left, causal
):
    """
    Test incremental generation scenario: q_len=1, kv_len grows gradually
    Simulate the token-by-token generation process in real large model inference
    """
    head_dim = 128
    sm_scale = 1.0 / math.sqrt(head_dim)

    # Create JIT arguments
    jit_args = (
        f"single_prefill_attention_sink_{filename_safe_dtype_map[dtype]}",
        dtype, dtype, dtype, torch.int32,
        head_dim, head_dim,
        ["sink"], ["float"],
        ["sm_scale"], ["double"],
        "AttentionSink",
        attention_sink_decl,
    )

    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
    )

    # Initialize KV cache - simulate state after prefill phase
    k_cache = torch.randn(
        batch_size, initial_seq_len, num_kv_heads, head_dim,
        dtype=dtype, device="cuda"
    )
    v_cache = torch.randn(
        batch_size, initial_seq_len, num_kv_heads, head_dim,
        dtype=dtype, device="cuda"
    )

    sink = torch.rand(num_qo_heads, device="cuda", dtype=torch.float32) * 100

    # Simulate incremental generation process
    for step in range(num_generation_steps):
        current_kv_len = initial_seq_len + step

        # Current generated new token (q_len=1)
        q_new = torch.randn(
            batch_size, num_qo_heads, head_dim,
            dtype=dtype, device="cuda"
        )

        # K,V for newly generated token
        k_new = torch.randn(
            batch_size, 1, num_kv_heads, head_dim,
            dtype=dtype, device="cuda"
        )
        v_new = torch.randn(
            batch_size, 1, num_kv_heads, head_dim,
            dtype=dtype, device="cuda"
        )

        # Update KV cache
        if step == 0:
            k_cache_current = k_cache
            v_cache_current = v_cache
        else:
            k_cache_current = torch.cat([k_cache, k_accumulated], dim=1)
            v_cache_current = torch.cat([v_cache, v_accumulated], dim=1)

        # Calculate reference result
        o_ref = sink_attention_incremental_ref(
            q_new, k_cache_current, v_cache_current,
            sink, window_left, causal, sm_scale
        )

        # Use flashinfer to calculate result (need format conversion to adapt to existing API)
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            float_workspace_buffer, kv_layout="NHD", backend="fa2", jit_args=jit_args
        )

        # Set correct indptr: q_len=1 for each batch, kv_len=current_kv_len for each batch
        qo_indptr_host = torch.arange(0, batch_size + 1, dtype=torch.int32)  # [0, 1, 2, ..., batch_size]
        kv_indptr_host = torch.arange(
            0, batch_size * current_kv_len + 1, current_kv_len, dtype=torch.int32
        )

        wrapper.plan(
            qo_indptr_host,
            kv_indptr_host,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            causal=causal,
            window_left=window_left,
            q_data_type=dtype,
            kv_data_type=dtype,
        )

        # Convert to format expected by flashinfer [total_q_len, num_heads, head_dim]
        q_flashinfer = q_new.view(batch_size, num_qo_heads, head_dim)  # [batch_size, num_heads, head_dim]
        k_flashinfer = k_cache_current.view(batch_size * current_kv_len, num_kv_heads, head_dim)
        v_flashinfer = v_cache_current.view(batch_size * current_kv_len, num_kv_heads, head_dim)

        o = wrapper.run(q_flashinfer, k_flashinfer, v_flashinfer, sink, sm_scale)

        # Verify results
        if dtype == torch.float16:
            torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
        else:
            torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)

        # Also test with BatchPrefillWithPagedKVCacheWrapper
        wrapper_paged = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer, kv_layout="NHD", backend="fa2", jit_args=jit_args
        )
        kv_indices_host = torch.arange(
            0,
            batch_size * current_kv_len,
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
            window_left=window_left,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        o_paged = wrapper_paged.run(q_flashinfer, (k_flashinfer, v_flashinfer), sink, sm_scale)
        if dtype == torch.float16:
            torch.testing.assert_close(o_paged, o_ref, rtol=1e-3, atol=1e-3)
        else:
            torch.testing.assert_close(o_paged, o_ref, rtol=1e-2, atol=1e-2)

        # Accumulate new K,V for next step
        if step == 0:
            k_accumulated = k_new
            v_accumulated = v_new
        else:
            k_accumulated = torch.cat([k_accumulated, k_new], dim=1)
            v_accumulated = torch.cat([v_accumulated, v_new], dim=1)

        print(f"Step {step}: q_len=1, kv_len={current_kv_len}, both RaggedKV and PagedKV wrappers passed!")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("chunk_size", [128, 256, 512])
@pytest.mark.parametrize("historical_len", [256, 512, 1024])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("window_left", [-1, 127, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_attention_sink_chunk_prefill(
    dtype, batch_size, chunk_size, historical_len,
    num_qo_heads, num_kv_heads, window_left, causal
):
    """
    Test chunk prefill scenario: q_len != kv_len and q_len > 1
    Simulate chunk-based processing of long sequences where current chunk
    attends to all historical tokens plus current chunk tokens
    """
    # Skip invalid combinations
    if chunk_size >= historical_len:
        pytest.skip("chunk_size should be smaller than historical_len for meaningful chunk prefill test")

    head_dim = 128
    sm_scale = 1.0 / math.sqrt(head_dim)
    total_kv_len = historical_len + chunk_size

    # Create JIT arguments
    jit_args = (
        f"chunk_prefill_attention_sink_{filename_safe_dtype_map[dtype]}",
        dtype, dtype, dtype, torch.int32,
        head_dim, head_dim,
        ["sink"], ["float"],
        ["sm_scale"], ["double"],
        "AttentionSink",
        attention_sink_decl,
    )

    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
    )

    # Create input tensors for chunk prefill scenario
    # q represents current chunk: [batch_size * chunk_size, num_heads, head_dim]
    q_chunk = torch.randn(
        batch_size * chunk_size, num_qo_heads, head_dim,
        dtype=dtype, device="cuda"
    )

    # k, v represent all tokens (historical + current chunk)
    k_all = torch.randn(
        batch_size * total_kv_len, num_kv_heads, head_dim,
        dtype=dtype, device="cuda"
    )
    v_all = torch.randn(
        batch_size * total_kv_len, num_kv_heads, head_dim,
        dtype=dtype, device="cuda"
    )

    sink = torch.rand(num_qo_heads, device="cuda", dtype=torch.float32) * 100

    # Calculate reference result using chunk prefill mode
    o_ref = sink_attention_chunk_ref(
        batch_size, q_chunk, k_all, v_all,
        sink, window_left, causal, sm_scale
    )

    # Test with flashinfer
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        float_workspace_buffer, kv_layout="NHD", backend="fa2", jit_args=jit_args
    )

    # Set up indices for chunk prefill
    qo_indptr_host = torch.arange(
        0, batch_size * chunk_size + 1, chunk_size, dtype=torch.int32
    )
    kv_indptr_host = torch.arange(
        0, batch_size * total_kv_len + 1, total_kv_len, dtype=torch.int32
    )

    wrapper.plan(
        qo_indptr_host,
        kv_indptr_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        window_left=window_left,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    o = wrapper.run(q_chunk, k_all, v_all, sink, sm_scale)

    # Verify results
    if dtype == torch.float16:
        torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)

    # Also test with BatchPrefillWithPagedKVCacheWrapper
    wrapper_paged = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        float_workspace_buffer, kv_layout="NHD", backend="fa2", jit_args=jit_args
    )
    kv_indices_host = torch.arange(
        0,
        batch_size * total_kv_len,
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
        window_left=window_left,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o_paged = wrapper_paged.run(q_chunk, (k_all, v_all), sink, sm_scale)
    if dtype == torch.float16:
        torch.testing.assert_close(o_paged, o_ref, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(o_paged, o_ref, rtol=1e-2, atol=1e-2)

    print(f"Chunk prefill test passed: q_len={chunk_size}, kv_len={total_kv_len}, "
          f"batch_size={batch_size}, causal={causal}")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("indptr_config", [
    # (qo_indptr, kv_indptr, description)
    ([0, 32, 64, 128, 256], [0, 128, 256, 512, 1024], "4 requests: prefill-like scenarios"),
    ([0, 1, 2, 3, 4], [0, 128, 256, 384, 512], "4 requests: incremental generation"),
    ([0, 50, 150, 200], [0, 200, 600, 800], "3 requests: mixed lengths"),
    ([0, 100, 200, 400, 600, 1000], [0, 300, 600, 1200, 1800, 3000], "5 requests: large sequences"),
    ([0, 16, 32, 96, 128], [0, 64, 128, 384, 512], "4 requests: chunk prefill-like"),
])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("window_left", [-1, 127])
@pytest.mark.parametrize("causal", [True, False])
def test_attention_sink_varlen(
    dtype, indptr_config, num_qo_heads, num_kv_heads, window_left, causal
):
    """
    Test variable length sequences within a batch.
    Each request in the batch can have different query and key/value lengths.
    """
    # Unpack the indptr configuration
    qo_indptr, kv_indptr, description = indptr_config

    # Validate that qo_indptr and kv_indptr have same batch size
    if len(qo_indptr) != len(kv_indptr):
        pytest.skip(f"qo_indptr and kv_indptr must have same batch size for {description}")

    batch_size = len(qo_indptr) - 1
    total_qo_len = qo_indptr[-1]
    total_kv_len = kv_indptr[-1]
    head_dim = 128
    sm_scale = 1.0 / math.sqrt(head_dim)

    # Check if any request has qo_len > kv_len for causal case
    if causal:
        for i in range(batch_size):
            qo_len_i = qo_indptr[i + 1] - qo_indptr[i]
            kv_len_i = kv_indptr[i + 1] - kv_indptr[i]
            if qo_len_i > kv_len_i:
                pytest.skip("qo_len > kv_len not supported for causal attention in varlen mode")

    # Create input tensors
    q = torch.randn(
        total_qo_len, num_qo_heads, head_dim,
        dtype=dtype, device="cuda"
    )
    k = torch.randn(
        total_kv_len, num_kv_heads, head_dim,
        dtype=dtype, device="cuda"
    )
    v = torch.randn(
        total_kv_len, num_kv_heads, head_dim,
        dtype=dtype, device="cuda"
    )

    qo_indptr_tensor = torch.tensor(qo_indptr, dtype=torch.int32, device="cuda")
    kv_indptr_tensor = torch.tensor(kv_indptr, dtype=torch.int32, device="cuda")

    sink = torch.rand(num_qo_heads, device="cuda", dtype=torch.float32) * 100

    # Test the variable length reference implementation
    o_ref = sink_attention_varlen_ref(
        q, k, v, sink, window_left, causal, sm_scale,
        qo_indptr_tensor, kv_indptr_tensor
    )

    # Verify output shape
    assert o_ref.shape == (total_qo_len, num_qo_heads, head_dim), \
        f"Expected shape ({total_qo_len}, {num_qo_heads}, {head_dim}), got {o_ref.shape}"

    # Test against FlashInfer kernel for verification
    # Create JIT arguments for attention sink
    jit_args = (
        f"varlen_prefill_attention_sink_{filename_safe_dtype_map[dtype]}",  # uri
        dtype,  # dtype_q
        dtype,  # dtype_kv
        dtype,  # dtype_o
        torch.int32,  # idtype
        head_dim,  # hidden_dim_qk
        head_dim,  # hidden_dim_vo
        ["sink"],  # additional_tensor_names
        ["float"],  # additional_tensor_dtypes
        ["sm_scale"],  # additional_scalar_names
        ["double"],  # additional_scalar_dtypes
        "AttentionSink",
        attention_sink_decl,
    )

    # Create workspace buffer
    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
    )

    # Test with BatchPrefillWithRaggedKVCacheWrapper
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        float_workspace_buffer, kv_layout="NHD", backend="fa2", jit_args=jit_args
    )

    wrapper.plan(
        qo_indptr_tensor,
        kv_indptr_tensor,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        window_left=window_left,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    o = wrapper.run(q, k, v, sink, sm_scale)

    # Compare varlen reference result with FlashInfer kernel result
    if dtype == torch.float16:
        torch.testing.assert_close(o_ref, o, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(o_ref, o, rtol=1e-2, atol=1e-2)

    # Also test with BatchPrefillWithPagedKVCacheWrapper
    wrapper_paged = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        float_workspace_buffer, kv_layout="NHD", backend="fa2", jit_args=jit_args
    )
    kv_indices_host = torch.arange(
        0,
        total_kv_len,
        dtype=torch.int32,
        device="cuda"
    )
    paged_kv_last_page_len_host = torch.full((batch_size,), 1, dtype=torch.int32, device="cuda")
    wrapper_paged.plan(
        qo_indptr_tensor,
        kv_indptr_tensor,
        kv_indices_host,
        paged_kv_last_page_len_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        1,
        causal=causal,
        window_left=window_left,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o_paged = wrapper_paged.run(q, (k, v), sink, sm_scale)
    if dtype == torch.float16:
        torch.testing.assert_close(o_ref, o_paged, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(o_ref, o_paged, rtol=1e-2, atol=1e-2)

    print(f"Variable length test passed: {description}, batch_size={batch_size}, "
          f"qo_lens={[qo_indptr[i+1]-qo_indptr[i] for i in range(batch_size)]}, "
          f"kv_lens={[kv_indptr[i+1]-kv_indptr[i] for i in range(batch_size)]}, "
          f"causal={causal}")


if __name__ == "__main__":
    test_attention_sink(
        torch.float16,
        batch_size=128,
        seq_len=1024,
        num_qo_heads=32,
        num_kv_heads=32,
        window_left=128,
        causal=False,
    )
