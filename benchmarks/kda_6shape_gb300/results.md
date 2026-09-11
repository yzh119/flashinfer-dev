GPU: NVIDIA GB300; SMs: 152; driver: 580.167.08; state: BF16; timing: CUPTI cold-L2.

FlashInfer: `9f1f3ea7807799a4b01face909b64ddd416ebe18`; FlashKDA: `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b`.

| H | Layout | Ours (ms) | FlashKDA (ms) | Speedup |
|--:|:--|--:|--:|--:|
| 96 | fixed | 0.352288 | 1.036945 | 2.9435x |
| 96 | mixed | 0.268449 | 0.840321 | 3.1303x |
| 96 | uniform | 0.288865 | 0.687681 | 2.3806x |
| 64 | fixed | 0.325729 | 0.950402 | 2.9178x |
| 64 | mixed | 0.187361 | 0.661633 | 3.5313x |
| 64 | uniform | 0.197921 | 0.473649 | 2.3931x |

Geomean (FlashKDA / ours): **2.854398x**
