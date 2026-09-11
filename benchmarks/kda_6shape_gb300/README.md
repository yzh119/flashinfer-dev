# GB300: PR #4845 versus FlashKDA, six shapes

复现 [FlashInfer PR #4845](https://github.com/flashinfer-ai/flashinfer/pull/4845)
的 BF16-state exported-kernel / FlashKDA 对比。
六个 shape 来自 [KDA-B200 的 CASES](https://github.com/Int21-AI/KDA-B200/blob/main/benchmarks/compare_cutlass_gb200.py#L21-L28)：

| H | Layout | Sequence lengths |
|--:|:--|:--|
| 96 / 64 | fixed | 8192 |
| 96 / 64 | mixed | 1300, 547, 2048, 963, 271, 3063 |
| 96 / 64 | uniform | 1024 × 8 |

所有场景 B=1（packed varlen）、总 token 数 8192、D=128。
Q/K/V/G/beta 和初始、最终 state 都是 BF16；A_log/dt_bias 为 FP32。
初始 state 为 `randn * 0.25`，scale 为 `1/sqrt(128)`，lower_bound 为 -5。
两边都在 kernel 内做 Q/K 归一化、gate 激活和 beta sigmoid。
**这里沿用 PR 性能表的 BF16 state；引用的 KDA-B200 脚本本身用 FP32 state，这里只采用它的六个 shape。**

本机实测六项 geomean 为 **2.859518×**（FlashKDA / ours），全部正确性检查通过。
详见 [results.md](results.md) 和 [原始 samples / 硬件记录](results.json)。
本次 GPU 为 NVIDIA GB300，152 SM，测量块结束时 SM 2070 MHz、显存 3996 MHz，
nvcc 13.3.73，驱动 580.167.08，PyTorch 2.14.0+cu130。

## 运行

需要 GB300、CUDA toolkit（nvcc 在 PATH 中）、已有 CUDA PyTorch 的 Python 环境、git、C++ 编译器。
`setup.sh` 使用 uv（若可用）或 pip 在所选 Python 环境中安装 benchmark 依赖。
PyTorch 由环境预先提供。源码和编译产物保存在本文件夹的 `.deps/` 下。

```bash
bash benchmarks/kda_6shape_gb300/setup.sh
CUDA_VISIBLE_DEVICES=0 bash benchmarks/kda_6shape_gb300/run.sh
python benchmarks/kda_6shape_gb300/summarize.py
```

可用 `PYTHON=/path/to/python` 同时指定 setup/run 的解释器。
运行脚本也接受 `--warmup-ms 50 --measure-ms 200 --output /path/to/results.json`。
输出路径的父文件夹须已存在。重复运行默认覆盖上次结果。

- `benchmark.py`：打印硬件信息、检查正确性、测量六个场景。
- `summarize.py [results.json]`：从绝对延迟重新计算六项加速比的 geomean。
- `results.json`：硬件、源码版本、实际路由、正确性误差、每块原始 samples。
- `results.md`：绝对延迟及 geomean 表格。
- `run.log`：完整运行输出（本地保留，git 忽略）。

## 固定版本和计时范围

- FlashInfer：PR 最终提交 `9f1f3ea7807799a4b01face909b64ddd416ebe18`。
- MoonshotAI/FlashKDA：`1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b`。
- FlashKDA CUTLASS：`5c149f52a436782210263fb2f19b354443a61c6a`，按其 gitlink 下载。

Ours 调用 PR 中 `prepare_flash_kda_evolution(...).launch()` 的导出 kernel；
baseline 调用 Moonshot `flash_kda._fwd_raw`，预分配 workspace。
两边都使用相同输入和独立的最终 state 输出；每次调用后初始 state 不变。
这是 prepared kernel 的 GPU 时间，不包含 public API 的 Python 开销、分配、JIT 和准备 metadata。
它也不是当前主分支 `recurrent_kda` 自动 dispatch 的性能。

每个 shape 先检查所有输出元素和最终 state（`atol=rtol=1e-2`），通过后才计时。
计时使用该固定 FlashInfer 版本的 `bench_gpu_time`，CUPTI、cold L2、无 CUDA graph，
按 ours / FlashKDA / FlashKDA / ours 顺序执行。
先用各实现 20 个 CUPTI samples 校准 iteration 数，另加 10% 余量；
每块目标 warmup 50 ms、measurement 200 ms，记录实际 iteration 数和 GPU 测量时长。
每个实现汇集两块 samples 后取 median。

`speedup = FlashKDA median / ours median`，大于 1 表示 ours 更快；
`geomean = exp(mean(log(speedup_i)))`，六个 shape 等权。

## 硬件信息

启动时打印 `nvcc --version`（工具链版本）和 PyTorch 编译所用 CUDA 版本，两者分别记录。
`pynvml`（pip 包名 `nvidia-ml-py`）按当前 CUDA 设备的 UUID 查询 GPU 型号、
驱动、当前/最大 SM、graphics 和 memory 频率、功耗/功耗上限、温度、P-state、
clock event reasons、显存容量/占用和利用率。
CUDA/PyTorch 提供 SM 数量和 compute capability。
每个计时块前后及整个运行结束时再次采样 NVML。
频率是采样时刻的动态读数；脚本不锁频、不修改功耗设置。
