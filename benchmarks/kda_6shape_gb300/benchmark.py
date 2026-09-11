"""PR #4845 BF16-state exported kernels versus pinned Moonshot FlashKDA."""

import argparse
import datetime
import importlib.metadata
import itertools
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
from pathlib import Path

import pynvml
import torch

from summarize import summarize

HERE = Path(__file__).resolve().parent
CASES = [
    (96, "fixed", [8192]),
    (96, "mixed", [1300, 547, 2048, 963, 271, 3063]),
    (96, "uniform", [1024] * 8),
    (64, "fixed", [8192]),
    (64, "mixed", [1300, 547, 2048, 963, 271, 3063]),
    (64, "uniform", [1024] * 8),
]


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def nvml_snapshot(handle):
    def query(fn, *args):
        try:
            value = fn(*args)
            return value.decode() if isinstance(value, bytes) else value
        except pynvml.NVMLError as error:
            return f"unavailable: {error}"

    data = {
        "gpu_name": query(pynvml.nvmlDeviceGetName, handle),
        "uuid": query(pynvml.nvmlDeviceGetUUID, handle),
        "driver": query(pynvml.nvmlSystemGetDriverVersion),
        "temperature_c": query(
            pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU
        ),
        "power_mw": query(pynvml.nvmlDeviceGetPowerUsage, handle),
        "power_limit_mw": query(pynvml.nvmlDeviceGetPowerManagementLimit, handle),
        "pstate": query(pynvml.nvmlDeviceGetPerformanceState, handle),
        "clock_event_reasons": query(
            pynvml.nvmlDeviceGetCurrentClocksThrottleReasons, handle
        ),
    }
    for label, clock in [
        ("sm", pynvml.NVML_CLOCK_SM),
        ("graphics", pynvml.NVML_CLOCK_GRAPHICS),
        ("memory", pynvml.NVML_CLOCK_MEM),
    ]:
        data[f"{label}_clock_mhz"] = query(pynvml.nvmlDeviceGetClockInfo, handle, clock)
        data[f"{label}_max_clock_mhz"] = query(
            pynvml.nvmlDeviceGetMaxClockInfo, handle, clock
        )
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    data.update(
        gpu_utilization_pct=util.gpu,
        memory_utilization_pct=util.memory,
        memory_total_bytes=mem.total,
        memory_used_bytes=mem.used,
    )
    return data


def source_info(name, revision, module):
    root = HERE / ".deps" / name
    actual = command("git", "-C", str(root), "rev-parse", "HEAD")
    if actual != revision or not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError(f"Unexpected {name} source: {actual}, {module.__file__}")
    if command("git", "-C", str(root), "diff", "HEAD", "--stat"):
        raise RuntimeError(f"Tracked changes in pinned {name} source")
    return {
        "commit": actual,
        "module": module.__file__,
        "submodules": command("git", "-C", str(root), "submodule", "status"),
    }


@torch.inference_mode()
def run_case(index, heads, layout, lengths, args, handle):
    import flash_kda
    from flashinfer.kda_evolution import prepare_flash_kda_evolution
    from flashinfer.testing import bench_gpu_time

    torch.manual_seed(10000 + index)
    shape = (1, sum(lengths), heads, 128)
    q, k, v, g = [torch.randn(shape, device="cuda").bfloat16() for _ in range(4)]
    beta = torch.randn(shape[:-1], device="cuda").bfloat16()
    a_log = torch.rand(heads, device="cuda")
    dt_bias = torch.rand(heads, 128, device="cuda")
    initial = (
        torch.randn(len(lengths), heads, 128, 128, device="cuda") * 0.25
    ).bfloat16()
    saved_initial = initial.clone()
    ours_out, peer_out = torch.empty_like(q), torch.empty_like(q)
    ours_state, peer_state = torch.empty_like(initial), torch.empty_like(initial)
    cu = (
        None
        if layout == "fixed"
        else torch.tensor(
            [0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int64
        )
    )
    scale = 1 / math.sqrt(128)
    print(f"Preparing H={heads} {layout}: {lengths}", flush=True)
    prepared = prepare_flash_kda_evolution(
        q,
        k,
        v,
        g,
        beta,
        a_log,
        dt_bias,
        initial,
        ours_out,
        ours_state,
        scale=scale,
        lower_bound=-5.0,
        cu_seqlens=cu,
    )
    workspace = torch.empty(
        flash_kda.get_workspace_size(sum(lengths), heads, len(lengths)),
        device="cuda",
        dtype=torch.uint8,
    )

    def peer():
        flash_kda._fwd_raw(
            q,
            k,
            v,
            g,
            beta,
            scale,
            peer_out,
            workspace,
            a_log,
            dt_bias,
            -5.0,
            initial,
            peer_state,
            cu,
        )

    prepared.launch()
    peer()
    torch.cuda.synchronize()
    errors = {}
    for label, actual, expected in [
        ("output", ours_out, peer_out),
        ("final_state", ours_state, peer_state),
    ]:
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        errors[label + "_max_abs"] = (
            (actual.float() - expected.float()).abs().max().item()
        )
    assert torch.equal(initial, saved_initial), "Initial state was mutated"
    # Calibrate using GPU samples after CUPTI/L2-flush initialization. The
    # timer's default CUDA-event estimate includes L2 and first-use overhead.
    budgets = {}
    for name, fn in [("ours", prepared.launch), ("flashkda", peer)]:
        pilot = bench_gpu_time(
            fn,
            enable_cupti=True,
            cold_l2_cache=True,
            use_cuda_graph=False,
            dry_run_iters=10,
            repeat_iters=20,
        )
        estimate = statistics.median(pilot)
        budgets[name] = {
            "dry_run_iters": math.ceil(1.1 * args.warmup_ms / estimate),
            "repeat_iters": math.ceil(1.1 * args.measure_ms / estimate),
        }
    samples = {"ours": [], "flashkda": []}
    blocks = []
    for name, fn in [
        ("ours", prepared.launch),
        ("flashkda", peer),
        ("flashkda", peer),
        ("ours", prepared.launch),
    ]:
        before = nvml_snapshot(handle)
        times = list(
            map(
                float,
                bench_gpu_time(
                    fn,
                    enable_cupti=True,
                    cold_l2_cache=True,
                    use_cuda_graph=False,
                    **budgets[name],
                ),
            )
        )
        samples[name].extend(times)
        blocks.append(
            {
                "backend": name,
                "median_ms": statistics.median(times),
                **budgets[name],
                "measured_gpu_ms": sum(times),
                "samples_ms": times,
                "hardware_before": before,
                "hardware_after": nvml_snapshot(handle),
            }
        )
    assert torch.equal(initial, saved_initial), "Timed calls mutated initial state"
    ours_ms = statistics.median(samples["ours"])
    peer_ms = statistics.median(samples["flashkda"])
    print(
        f"H={heads} {layout}: ours {ours_ms:.6f} ms, FlashKDA {peer_ms:.6f} ms, {peer_ms / ours_ms:.4f}x"
    )
    return {
        "heads": heads,
        "layout": layout,
        "sequence_lengths": lengths,
        "seed": 10000 + index,
        "correctness": "passed",
        **errors,
        "variant": prepared.variant,
        "target": prepared.target,
        "ours_ms": ours_ms,
        "flashkda_ms": peer_ms,
        "speedup": peer_ms / ours_ms,
        "blocks": blocks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--measure-ms", type=int, default=200)
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    args = parser.parse_args()
    if args.warmup_ms <= 0 or args.measure_ms <= 0:
        parser.error("Timing durations must be positive")
    import flashinfer
    import flash_kda
    import flash_kda_C
    from flashinfer.testing import utils as timing

    # Do not silently label a CUDA-event fallback as CUPTI.
    from cupti import cupti  # noqa: F401

    if int(importlib.metadata.version("cupti-python").split(".")[0]) < 13:
        raise RuntimeError("cupti-python >= 13 is required")
    pynvml.nvmlInit()
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (10, 3):
        raise RuntimeError("This reproduction targets GB300 / SM103a")
    # UUID remains correct when CUDA_VISIBLE_DEVICES remaps device indices.
    handle = pynvml.nvmlDeviceGetHandleByUUID("GPU-" + str(properties.uuid))
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc must be on PATH")
    hardware = nvml_snapshot(handle)
    hardware.update(
        sm_count=properties.multi_processor_count,
        compute_capability=[properties.major, properties.minor],
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        nvcc_path=nvcc,
        nvcc_version=command(nvcc, "--version"),
        python=platform.python_version(),
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        cupti_python=importlib.metadata.version("cupti-python"),
    )
    print(json.dumps(hardware, indent=2), flush=True)
    sources = {
        "flashinfer": source_info(
            "flashinfer", "9f1f3ea7807799a4b01face909b64ddd416ebe18", flashinfer
        ),
        "flashkda": source_info(
            "FlashKDA", "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b", flash_kda
        ),
        "flashkda_extension": flash_kda_C.__file__,
        "timer": timing.__file__,
    }
    assert (
        Path(flash_kda_C.__file__).resolve().is_relative_to(HERE / ".deps" / "FlashKDA")
    )
    payload = {
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "hardware": hardware,
        "sources": sources,
        "method": {
            "state_dtype": "bfloat16",
            "input_dtype": "bfloat16",
            "head_dim": 128,
            "lower_bound": -5.0,
            "timing": "CUPTI",
            "cold_l2": True,
            "order": "ours/flashkda/flashkda/ours",
            "warmup_ms_per_block": args.warmup_ms,
            "measure_ms_per_block": args.measure_ms,
            "scope": "prepared exported launch vs reusable-workspace _fwd_raw",
            "atol": 1e-2,
            "rtol": 1e-2,
        },
        "results": [],
    }
    for i, (heads, layout, lengths) in enumerate(CASES):
        payload["results"].append(run_case(i, heads, layout, lengths, args, handle))
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        torch.cuda.empty_cache()
    gm, table = summarize(payload)
    payload["geomean_speedup"] = gm
    payload["hardware_after"] = nvml_snapshot(handle)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(table)
    print(table)
    print(
        "Hardware after measurement:", json.dumps(payload["hardware_after"], indent=2)
    )
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
