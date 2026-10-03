# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CAKE DeepSeek-V4 NVFP4 (384-byte cache) sparse-MLA prefill on SM100/SM103.

Every case runs :func:`flashinfer.mla.cake_sparse_mla_sm100_dsv4_nvfp4_prefill`
on caches and a query packed by
:func:`flashinfer.mla.nvfp4_quantize_pack_sparse_mla_cache` and checks the
BF16 output and the base-2 LSE against an FP32 oracle that reads the
dequantized NVFP4 pools and the dequantized packed query (the SM120 NVFP4
kernel-error gate: O ``atol=rtol=5e-2``, LSE ``atol=rtol=2e-2``). The grid
covers single and dual caches, independent main/extra lengths, ``-1``
padding, rows without a valid entry, sinks, partial KV tiles, odd token
counts, HND/NHD, padded page pitches, column-sliced tables, CUDA-graph replay,
the single-CTA variant (up to 64 heads) and the tracking rows (K=256,
8 heads, non-default page sizes). The trtllm-gen DSv4 entry point keeps
refusing the NVFP4 cache on SM100/SM103.
"""

from __future__ import annotations

import math

import pytest
import torch

import flashinfer
from flashinfer.mla import (
    cake_sparse_mla_sm100_dsv4_nvfp4_prefill,
    nvfp4_quantize_append_sparse_mla_cache,
    nvfp4_quantize_pack_sparse_mla_cache,
    trtllm_batch_decode_sparse_mla_dsv4,
)
from flashinfer.mla.cake_dsv4 import _nvfp4_route
from flashinfer.utils import get_compute_capability
from tests.attention.sparse_mla_test_utils import (
    _BYTES_PER_TOKEN,
    _dequantize_nvfp4_cache,
)

HEAD_DIM = 512
SCALE = HEAD_DIM**-0.55
O_TOL = dict(atol=5e-2, rtol=5e-2)
LSE_TOL = dict(atol=2e-2, rtol=2e-2)
LOG2E = math.log2(math.e)


def _require_sm100_family() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, _ = get_compute_capability(torch.device("cuda"))
    if major != 10:
        pytest.skip("the CAKE DSv4 NVFP4 prefill requires SM100/SM103")


# Latent pools follow the kernel contract's input domain: N(offset, 0.25) clamped to [-1, 1]. The O gate
# (atol = rtol = 5e-2) is defined on that domain; the route's PV path dequantizes V to E4M3 with the
# per-16 scale folded in, and E4M3 spacing is 2^-4 of the value's binade (0.0625 for |v| in [1, 2),
# 0.25 for |v| in [2, 4)). With an unclamped N(0, 0.5) pool (|v| up to ~3) a two-key row with p ~ 0.5 can
# therefore legitimately differ from the FP32 oracle by ~0.06 on a single element (measured: 1 of
# 5,046,272 at 0.056, deterministic across runs, removed entirely when the oracle models the E4M3 V
# fold), which is a property of the quantized PV path shared with the FP8 route, not of this kernel.
_POOL_STD = 0.25
_POOL_CLAMP = 1.0


def _pool(gen, pages: int, page_size: int, layout: str, offset: float):
    latent = (
        (
            torch.randn((pages, page_size, HEAD_DIM), generator=gen, device="cuda")
            * _POOL_STD
            + offset
        )
        .clamp(-_POOL_CLAMP, _POOL_CLAMP)
        .to(torch.bfloat16)
    )
    return latent, nvfp4_quantize_pack_sparse_mla_cache(latent, kv_layout=layout)


def _selection(gen, rows: int, width: int, pool_tokens: int, lens_rule: str):
    """Random unique token ids inside the active prefix, -1 past it, independent lengths."""
    if lens_rule == "full":
        lens = torch.full((rows,), width, dtype=torch.int32)
    elif lens_rule == "random":
        lens = (
            torch.randint(0, width + 1, (rows,), generator=gen, device="cuda")
            .to(torch.int32)
            .cpu()
        )
        lens[0] = width  # at least one complete row
    elif lens_rule == "zero_rows":
        lens = (
            torch.randint(1, width + 1, (rows,), generator=gen, device="cuda")
            .to(torch.int32)
            .cpu()
        )
        lens[::3] = 0
    else:
        raise ValueError(lens_rule)
    table = torch.full((rows, width), -1, dtype=torch.int32)
    for r in range(rows):
        n = int(lens[r])
        if n:
            perm = torch.randperm(pool_tokens, generator=gen, device="cuda")[:n]
            table[r, :n] = perm.to(torch.int32).cpu()
            # Interior -1 padding inside the active prefix is masked too.
            if n >= 8 and r % 2 == 1:
                table[r, n // 2] = -1
    return table.cuda(), lens.cuda()


def _case(
    *,
    heads: int,
    main_topk: int,
    extra_topk: int = 0,
    rows: int = 77,
    main_page: int = 64,
    extra_page: int = 64,
    layout: str = "HND",
    sink: bool = True,
    lens_rule: str = "random",
    seed: int = 0,
    pool_pages: int = 32,
):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    main_latent, main_cache = _pool(gen, pool_pages, main_page, layout, -0.05)
    main_idx, main_lens = _selection(
        gen, rows, main_topk, pool_pages * main_page, lens_rule
    )
    case = dict(
        heads=heads,
        rows=rows,
        main_latent=main_latent,
        main_cache=main_cache,
        main_idx=main_idx,
        main_lens=main_lens,
        extra_latent=None,
        extra_cache=None,
        extra_idx=None,
        extra_lens=None,
        layout=layout,
    )
    if extra_topk:
        extra_pages = max(pool_pages * main_page // extra_page, 2)
        extra_latent, extra_cache = _pool(gen, extra_pages, extra_page, layout, 0.05)
        extra_idx, extra_lens = _selection(
            gen, rows, extra_topk, extra_pages * extra_page, lens_rule
        )
        case.update(
            extra_latent=extra_latent,
            extra_cache=extra_cache,
            extra_idx=extra_idx,
            extra_lens=extra_lens,
        )
    query = (
        torch.randn((rows, heads, HEAD_DIM), generator=gen, device="cuda") * 0.6
    ).to(torch.bfloat16)
    case["query"] = query
    case["q_packed"] = _pack_query(query)
    case["sinks"] = (
        (torch.randn((heads,), generator=gen, device="cuda") * 0.3).float()
        if sink
        else None
    )
    return case


def _pack_query(query: torch.Tensor) -> torch.Tensor:
    """[T, H, 512] BF16 -> [T, H, 384] packed NVFP4 rows (one-token pages)."""
    rows, heads = query.shape[:2]
    packed = nvfp4_quantize_pack_sparse_mla_cache(
        query.reshape(rows * heads, 1, HEAD_DIM)
    )
    return packed.view(rows, heads, _BYTES_PER_TOKEN)


def _flat_rows(cache: torch.Tensor) -> torch.Tensor:
    return _dequantize_nvfp4_cache(cache).reshape(-1, HEAD_DIM)


def _oracle(case, *, main_lens=None, chunk: int = 64):
    """FP32 oracle on the dequantized NVFP4 pools and the NVFP4-quantized query NoPE."""
    main_rows = _flat_rows(case["main_cache"])
    extra_rows = (
        _flat_rows(case["extra_cache"]) if case["extra_cache"] is not None else None
    )
    q_packed = case["q_packed"]
    q = _dequantize_nvfp4_cache(q_packed.reshape(-1, 1, 1, _BYTES_PER_TOKEN)).reshape(
        q_packed.shape[0], q_packed.shape[1], HEAD_DIM
    )
    main_idx = case["main_idx"]
    main_lens = case["main_lens"] if main_lens is None else main_lens
    rows, heads = q.shape[:2]
    out = torch.zeros((rows, heads, HEAD_DIM), dtype=torch.float32, device="cuda")
    lse = torch.full((rows, heads), float("-inf"), dtype=torch.float32, device="cuda")
    for start in range(0, rows, chunk):
        stop = min(rows, start + chunk)
        idx = main_idx[start:stop]
        pos = torch.arange(idx.shape[1], device="cuda").unsqueeze(0)
        valid = (idx >= 0) & (pos < main_lens[start:stop].unsqueeze(1))
        kv = main_rows[idx.clamp_min(0).long()]
        if case["extra_idx"] is not None:
            eidx = case["extra_idx"][start:stop]
            epos = torch.arange(eidx.shape[1], device="cuda").unsqueeze(0)
            evalid = (eidx >= 0) & (epos < case["extra_lens"][start:stop].unsqueeze(1))
            kv = torch.cat((kv, extra_rows[eidx.clamp_min(0).long()]), dim=1)
            valid = torch.cat((valid, evalid), dim=1)
        scores = torch.einsum("rhd,rkd->rhk", q[start:stop], kv) * SCALE
        scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
        row_lse = torch.logsumexp(scores, dim=-1)
        safe = torch.where(torch.isinf(row_lse), torch.zeros_like(row_lse), row_lse)
        probs = torch.exp(scores - safe.unsqueeze(-1)).masked_fill(
            ~valid.unsqueeze(1), 0.0
        )
        o = torch.einsum("rhk,rkd->rhd", probs, kv)
        if case["sinks"] is not None:
            s = case["sinks"].reshape(1, -1)
            o = o * torch.sigmoid(row_lse - s).unsqueeze(-1)
            row_lse = torch.logaddexp(row_lse, s.expand_as(row_lse))
        out[start:stop] = o
        lse[start:stop] = row_lse * LOG2E
    return out, lse


def _run(case, *, out=None, lse=None, offset: int = 0, extra_lens="given"):
    rows, heads = case["rows"], case["heads"]
    if out is None:
        out = torch.empty((rows, heads, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    if lse is None:
        lse = torch.empty((rows, heads), dtype=torch.float32, device="cuda")
    cake_sparse_mla_sm100_dsv4_nvfp4_prefill(
        case["q_packed"],
        case["main_cache"],
        case["main_idx"],
        out,
        lse,
        SCALE,
        topk_length=case["main_lens"],
        topk_length_offset=offset,
        attn_sink=case["sinks"],
        extra_kv_cache=case["extra_cache"],
        extra_indices=case["extra_idx"],
        extra_topk_length=case["extra_lens"] if extra_lens == "given" else None,
    )
    return out, lse


def _check(case, out, lse, **oracle_kwargs):
    ref_out, ref_lse = _oracle(case, **oracle_kwargs)
    torch.testing.assert_close(out.float(), ref_out, **O_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


# --------------------------------------------------------------------------- #
# Correctness grid                                                            #
# --------------------------------------------------------------------------- #

_GRID = [
    pytest.param(dict(heads=128, main_topk=512), id="h128-k512-single"),
    pytest.param(dict(heads=128, main_topk=128, extra_topk=128), id="h128-k128-dual"),
    pytest.param(
        dict(heads=64, main_topk=512, extra_topk=512, layout="NHD"),
        id="h64-k512-dual-nhd",
    ),
    pytest.param(
        dict(heads=64, main_topk=128, sink=False), id="h64-k128-single-nosink"
    ),
    pytest.param(
        dict(heads=32, main_topk=128, extra_topk=128, extra_page=2),
        id="h32-k128-dual-extrapage2",
    ),
    pytest.param(dict(heads=16, main_topk=512), id="h16-k512-single"),
    pytest.param(
        dict(heads=16, main_topk=128, extra_topk=128, layout="NHD"),
        id="h16-k128-dual-nhd",
    ),
    pytest.param(dict(heads=8, main_topk=128), id="h8-k128-single-tracking"),
    pytest.param(dict(heads=128, main_topk=256), id="h128-k256-single-tracking"),
    pytest.param(
        dict(heads=128, main_topk=128, main_page=32, extra_topk=128, extra_page=128),
        id="h128-k128-dual-page32-128",
    ),
]


@pytest.mark.parametrize("spec", _GRID)
def test_nvfp4_prefill_matches_dequantized_oracle(spec) -> None:
    """Ragged two-request batch (37 + 40 tokens: partial KV tiles, odd token count),
    independent random lengths, interior and trailing -1 padding."""
    _require_sm100_family()
    case = _case(seed=1, **spec)
    out, lse = _run(case)
    torch.cuda.synchronize()
    _check(case, out, lse)


@pytest.mark.parametrize("sink", [False, True])
def test_nvfp4_prefill_rows_without_valid_kv(sink: bool) -> None:
    """Rows with length 0 yield zeros and an LSE of -inf (or the sink alone)."""
    _require_sm100_family()
    case = _case(
        heads=128,
        main_topk=128,
        extra_topk=128,
        lens_rule="zero_rows",
        sink=sink,
        seed=2,
    )
    out, lse = _run(case)
    torch.cuda.synchronize()
    empty = (case["main_lens"] == 0) & (case["extra_lens"] == 0)
    assert bool(empty.any())
    assert torch.all(out[empty] == 0)
    if sink:
        torch.testing.assert_close(
            lse[empty], (case["sinks"] * LOG2E).expand(int(empty.sum()), -1), **LSE_TOL
        )
    else:
        assert torch.all(torch.isneginf(lse[empty]))
    _check(case, out, lse)


def test_nvfp4_prefill_omitted_extra_lengths_activate_every_column() -> None:
    _require_sm100_family()
    case = _case(heads=128, main_topk=128, extra_topk=128, lens_rule="full", seed=4)
    out, lse = _run(case, extra_lens=None)
    torch.cuda.synchronize()
    _check(case, out, lse)


def test_nvfp4_prefill_length_offset_applies_to_main_segment() -> None:
    _require_sm100_family()
    case = _case(heads=64, main_topk=512, lens_rule="full", seed=5)
    out, lse = _run(case, offset=-96)
    torch.cuda.synchronize()
    _check(case, out, lse, main_lens=(case["main_lens"] - 96).clamp_min(0))


def test_nvfp4_prefill_padded_page_pitch_and_append() -> None:
    """A vLLM-style pool with padding between pages, filled by the append helper."""
    _require_sm100_family()
    case = _case(heads=128, main_topk=512, seed=6, pool_pages=16)
    pages, page_size = 16, 64
    pitch = (page_size + 3) * _BYTES_PER_TOKEN
    backing = torch.full((pages * pitch,), 0xA5, dtype=torch.uint8, device="cuda")
    strided = torch.as_strided(
        backing, (pages, page_size, _BYTES_PER_TOKEN), (pitch, _BYTES_PER_TOKEN, 1)
    )
    slots = torch.arange(pages * page_size, dtype=torch.int64, device="cuda")
    nvfp4_quantize_append_sparse_mla_cache(
        case["main_latent"].reshape(-1, HEAD_DIM), slots, strided
    )
    padded_cache = torch.as_strided(
        backing,
        (pages, 1, page_size, _BYTES_PER_TOKEN),
        (pitch, pitch, _BYTES_PER_TOKEN, 1),
    )
    reference_out, reference_lse = _run(case)
    torch.cuda.synchronize()
    case["main_cache"] = padded_cache
    out, lse = _run(case)
    torch.cuda.synchronize()
    # Same bytes per page as the full-page pack: identical attention output and LSE.
    torch.testing.assert_close(out, reference_out, atol=0, rtol=0)
    torch.testing.assert_close(lse, reference_lse, atol=0, rtol=0)
    assert torch.all(
        backing.view(pages, pitch)[:, page_size * _BYTES_PER_TOKEN :] == 0xA5
    )


def test_nvfp4_prefill_column_sliced_tables_match_contiguous() -> None:
    """Row-strided views of one wide table are read without a copy."""
    _require_sm100_family()
    case = _case(heads=64, main_topk=128, extra_topk=512, seed=7)
    reference_out, reference_lse = _run(case)
    torch.cuda.synchronize()
    wide = torch.full(
        (case["rows"], 128 + 512 + 64), -7, dtype=torch.int32, device="cuda"
    )
    wide[:, :128] = case["main_idx"]
    wide[:, 128:640] = case["extra_idx"]
    case["main_idx"], case["extra_idx"] = wide[:, :128], wide[:, 128:640]
    out, lse = _run(case)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, reference_out, atol=0, rtol=0)
    torch.testing.assert_close(lse, reference_lse, atol=0, rtol=0)


def test_nvfp4_prefill_validates_shapes() -> None:
    _require_sm100_family()
    case = _case(heads=64, main_topk=128, seed=8)
    rows = case["rows"]
    with pytest.raises(ValueError, match="packed NVFP4 query"):
        cake_sparse_mla_sm100_dsv4_nvfp4_prefill(
            case["query"],
            case["main_cache"],
            case["main_idx"],
            torch.empty((rows, 64, HEAD_DIM), dtype=torch.bfloat16, device="cuda"),
            torch.empty((rows, 64), dtype=torch.float32, device="cuda"),
            SCALE,
        )
    with pytest.raises(ValueError, match="one row per query token"):
        cake_sparse_mla_sm100_dsv4_nvfp4_prefill(
            case["q_packed"],
            case["main_cache"],
            case["main_idx"][:-1],
            torch.empty((rows, 64, HEAD_DIM), dtype=torch.bfloat16, device="cuda"),
            torch.empty((rows, 64), dtype=torch.float32, device="cuda"),
            SCALE,
        )
    with pytest.raises(ValueError, match="out_lse must be float32"):
        cake_sparse_mla_sm100_dsv4_nvfp4_prefill(
            case["q_packed"],
            case["main_cache"],
            case["main_idx"],
            torch.empty((rows, 64, HEAD_DIM), dtype=torch.bfloat16, device="cuda"),
            torch.empty((rows, 64, 1), dtype=torch.float32, device="cuda"),
            SCALE,
        )
    with pytest.raises(ValueError, match="extra_indices requires extra_kv_cache"):
        cake_sparse_mla_sm100_dsv4_nvfp4_prefill(
            case["q_packed"],
            case["main_cache"],
            case["main_idx"],
            torch.empty((rows, 64, HEAD_DIM), dtype=torch.bfloat16, device="cuda"),
            torch.empty((rows, 64), dtype=torch.float32, device="cuda"),
            SCALE,
            extra_indices=case["main_idx"],
        )


def test_nvfp4_prefill_cuda_graph_replay() -> None:
    """Capture once, replay with new query bytes: no allocation, results track the eager call."""
    _require_sm100_family()
    case = _case(heads=128, main_topk=128, extra_topk=128, seed=10)
    out = torch.empty(
        (case["rows"], 128, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    lse = torch.empty((case["rows"], 128), dtype=torch.float32, device="cuda")
    _run(case, out=out, lse=lse)  # eager warm-up on the same tensors (JIT, caches)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        _run(case, out=out, lse=lse)
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            _run(case, out=out, lse=lse)
    torch.cuda.synchronize()
    for seed in (11, 12):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        case["query"].copy_(
            (torch.randn(case["query"].shape, generator=gen, device="cuda") * 0.6).to(
                torch.bfloat16
            )
        )
        case["q_packed"].copy_(_pack_query(case["query"]))
        out.fill_(0)
        graph.replay()
        torch.cuda.synchronize()
        _check(case, out, lse)


@pytest.mark.parametrize("heads", [8, 16, 32, 64, 96, 128])
def test_nvfp4_route_selection_matches_head_count(heads: int) -> None:
    if heads <= 64:
        expected = "nvfp4_h64_prefill_persistent"
    else:
        expected = "nvfp4_h128_prefill_persistent" + (
            "" if heads % 64 == 0 else "_thin_heads"
        )
    assert _nvfp4_route(heads) == expected


def test_trtllm_dsv4_entry_point_refuses_nvfp4_cache_on_sm100() -> None:
    """On SM100/SM103 the NVFP4 cache has its own entry point; the DSv4 trtllm-gen API refuses it."""
    _require_sm100_family()
    case = _case(heads=64, main_topk=128, seed=14)
    rows = case["rows"]
    workspace = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    for backend in ("cake", "sparse"):
        with pytest.raises(ValueError, match="SM120"):
            trtllm_batch_decode_sparse_mla_dsv4(
                case["query"],
                case["main_cache"],
                workspace,
                case["main_idx"],
                swa_topk_lens=case["main_lens"],
                seq_lens=torch.full((1,), rows, dtype=torch.int32, device="cuda"),
                cum_seq_lens_q=torch.tensor(
                    [0, rows], dtype=torch.int32, device="cuda"
                ),
                max_q_len=rows,
                bmm1_scale=SCALE,
                backend=backend,
                kv_cache_format="nvfp4",
            )
    assert (
        flashinfer.mla.cake_sparse_mla_sm100_dsv4_nvfp4_prefill
        is cake_sparse_mla_sm100_dsv4_nvfp4_prefill
    )
