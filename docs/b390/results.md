# B390 v1 Results

This is the standalone performance and verification summary for the B390 tuned fork. The only formal versions are `baseline` and `v1`.

## Formal versions

| Name | Location | Definition |
| --- | --- | --- |
| `baseline` | `source-baseline`, tag `baseline` | Unmodified upstream snapshot at commit `46bdeb47d`. |
| `v1` | `source`, tag `v1`, HEAD `9c812b1d9` | B390 tuned tree merged with upstream `master` through `c7bda030e`. |

## Test model

| Item | Value |
| --- | --- |
| Model family | Qwen3.6 35B-A3B, IQ3_XXS, 3.0625 bpw |
| GGUF file | `Qwen3.6-35B-A3B-DSV4Pro-SFT-GPT56Sol-RL-Agent-LynnStyle-Q3_16G-LynnStyle-imatrix.gguf` |
| Model directory | `C:\models\Qwen3.6-35B-A3B-DSV4Pro-SFT-GPT56Sol-RL-Agent-GGUF` |

## Performance comparison

The test device was an Intel Arc B390 Xe2 UMA system using Vulkan. The four rows below use the same model and official default runtime parameters: `-p 512 -n 128 -pg "512,64"`, either `-d 0` or `-d 15808`, `-b 2048 -ub 512 -t 16 -fa auto -ctk f16 -ctv f16 -ngl -1 -dev Vulkan0 -r 5`.

| Build/profile | Context | Prefill `pp512` | Generation `tg128` | Combined `pp512+tg64` |
| --- | --- | ---: | ---: | ---: |
| Official baseline, default params | Empty `d=0` | `765.75 +/- 7.94 tok/s` | `34.21 +/- 0.16 tok/s` | `224.32 +/- 1.35 tok/s` |
| Official baseline, default params | Full `d=15808` | `218.60 +/- 0.82 tok/s` | `24.72 +/- 0.13 tok/s` | `111.32 +/- 2.54 tok/s` |
| Modified v1, default runtime params | Empty `d=0` | `871.55 +/- 19.47 tok/s` | `37.68 +/- 0.17 tok/s` | `240.19 +/- 5.07 tok/s` |
| Modified v1, default runtime params | Full `d=15808` | `608.72 +/- 8.51 tok/s` | `30.00 +/- 0.73 tok/s` | `190.50 +/- 2.59 tok/s` |

The three metric columns are prefill-only, generation-only, and combined prompt-plus-generation throughput. These are local measurements on the same machine and should be treated as directional rather than hardware-independent guarantees.

The empty-context PP512 rows are the source of the commonly remembered 700-800 tok/s result and are not directly comparable with the full-context `d=15808` rows.

## Optimization dimensions

The optimization is split into two related areas:

1. **Runtime/build parameters:** GPU offload, Flash Attention mode, batch and ubatch sizes, CPU threads, KV-cache types, context depth, and B390-specific CMake switches select the execution path and scheduling policy.
2. **Source-code and shader changes:** B390/Xe2 gating, cooperative-matrix and quantized GEMM/mat-vec tuning, Flash Attention phases, MoE routing/compaction, graph submission, and fused GDN/SSM state updates change the implementation itself.

The baseline/v1 benchmark holds the runtime parameters constant between builds. The reported difference therefore covers the tuned build configuration together with the source-code and Vulkan shader changes.

## Optimization summary

| Subsystem | Main changes | Effect |
| --- | --- | --- |
| Device and pipeline | B390/Xe2 detection, shape-aware selection, tuned tiles, and fallback paths | Enables specialized paths only for supported hardware and shapes. |
| Quantized GEMM and mat-vec | IQ3/IQ2/Q4_K/Q5_K cooperative-matrix, dequantization, and row-layout tuning | Reduces register and memory overhead in quantized matrix operations. |
| Flash Attention | Xe gating, head-size kernels, and split prefill/decode phases | Matches B390 attention shapes and controls phase/synchronization structure. |
| MoE routing | Skip unnecessary work, pack routed rows, compact experts, and route/cache support | Reduces route preparation and repeated dispatch for the target MoE workload. |
| Graph submission | Shape-aware submit and graph-FLOP-based submit decisions | Uses different submit boundaries for long and short graphs while retaining fallback behavior. |
| State updates | Fused GDN state writes and `SSM_CONV` state copies | Reduces intermediate writes and copies in recurrent state paths. |
| Final mat-vec tuning | IQ2_S `MUL_MAT_ID_VEC` uses `NUM_ROWS=16`; Q4_K/Q5_K block-row loops use `dont_unroll` | Lowers register pressure for frequent B390 decode shapes. |

## Verification

- Vulkan `MUL_MAT_ID`: `883/883`.
- Vulkan `MUL_MAT`: `1071/1071`.
- Release Vulkan DLL, `llama-cli`, and `llama-bench` builds completed.
- Fixed prompt/seed model smoke passed.

For the exact machine snapshot, CMake configuration, complete file inventory, and reproduction details, see [B390-VULKAN-V1.md](../../B390-VULKAN-V1.md).
