# llama.cpp B390 Vulkan tuned fork

This is the canonical code-side report for the B390 tuned fork. It records the version definition, optimization inventory, and implementation details. See [docs/b390/results.md](docs/b390/results.md) for the standalone performance summary and [docs/b390/operations.md](docs/b390/operations.md) for reproduction operations.

This repository contains two published reference points:

- `baseline`: an unmodified upstream llama.cpp tree at upstream commit `46bdeb47d`.
- `v1`: the B390 tuned tree plus the complete upstream `master` sync described below.

The tuning target is an Intel Arc B390 UMA system running the Qwen3.6 35B-A3B GGUF model through the Vulkan backend. The changes are intentionally hardware and workload gated. They are not a replacement for the generic Vulkan defaults on other devices.

## Target setup

| Item | Value |
| --- | --- |
| GPU | Intel Arc B390, Xe2, UMA |
| Backend | Vulkan |
| Model | Qwen3.6 35B-A3B, IQ3_XXS, 3.0625 bpw |
| Test model file | `Qwen3.6-35B-A3B-DSV4Pro-SFT-GPT56Sol-RL-Agent-LynnStyle-Q3_16G-LynnStyle-imatrix.gguf` |
| Model source | `C:\models\Qwen3.6-35B-A3B-DSV4Pro-SFT-GPT56Sol-RL-Agent-GGUF` |
| Build | Release, native CPU features, Vulkan enabled |
| Comparison profiles | `-p 512 -n 128 -pg "512,64" -d 0` and `-d 15808`, `-b 2048 -ub 512 -t 16 -fa auto -ctk f16 -ctv f16 -ngl -1 -dev Vulkan0 -r 5` |
| Tuning A/B profile | `-ngl -1 -fa on -b 512 -ub 128` |
| Tuning decode test | `-p 1 -n 64` |
| Tuning prompt test | `-p 128 -n 0` |

The benchmark numbers below are local measurements on this setup. They should be treated as directional evidence, not as a hardware-independent guarantee.

## Machine and software environment

This is the exact environment captured for the v1 comparison. The WMI adapter-memory field is reported as-is; on this UMA system it is not the total usable system memory.

| Item | Value |
| --- | --- |
| OS | Windows 11 Professional Insider Preview, build `10.0.29648` |
| CPU | Intel Core Ultra X7 358H, 16 cores / 16 logical processors |
| RAM | 31.52 GB |
| GPU | Intel Arc B390 GPU, device `0xb080`, vendor `0x8086` |
| GPU memory field | 4,293,918,720 bytes reported by WMI; system is UMA |
| Intel driver | `32.0.101.8991`, date `2026-08-24` |
| Vulkan instance | `1.4.350` |
| Vulkan device API | `1.4.356` |
| Vulkan driver | Intel proprietary Windows, `101.8991` |
| Vulkan capabilities observed | warp 32, shared memory 49,152 bytes, integer dot product, `KHR_coopmat`, FP16, UMA |
| Vulkan SDK | `1.4.357.0` |
| Compiler | MSVC `19.51.36252.0` |
| Build configuration | Release, `GGML_VULKAN=ON`, `GGML_NATIVE=ON`, baseline `GGML_LTO=OFF`, v1 `GGML_LTO=ON`, tests off, ccache off |

## Baseline/v1 comparison

The four-row comparison uses the exact model file listed above, official default runtime parameters, five repetitions, and the same Intel Arc B390/Vulkan0 device. In PowerShell, the `-pg` value must be passed as one quoted argument: `-pg "512,64"`.

| Build/profile | Context | Prefill `pp512` | Generation `tg128` | Combined `pp512+tg64` |
| --- | --- | ---: | ---: | ---: |
| Official baseline, default params | Empty `d=0` | `765.75 +/- 7.94 tok/s` | `34.21 +/- 0.16 tok/s` | `224.32 +/- 1.35 tok/s` |
| Official baseline, default params | Full `d=15808` | `218.60 +/- 0.82 tok/s` | `24.72 +/- 0.13 tok/s` | `111.32 +/- 2.54 tok/s` |
| Modified v1, default runtime params | Empty `d=0` | `871.55 +/- 19.47 tok/s` | `37.68 +/- 0.17 tok/s` | `240.19 +/- 5.07 tok/s` |
| Modified v1, default runtime params | Full `d=15808` | `608.72 +/- 8.51 tok/s` | `30.00 +/- 0.73 tok/s` | `190.50 +/- 2.59 tok/s` |

The `pp`, `tg`, and `pg` columns are different llama-bench workloads: prefill-only, generation-only, and combined prompt-plus-generation throughput.

### Why PP512 can be 700-800 tok/s

The commonly remembered 700-800 tok/s PP512 result is real, but it usually refers to an empty-context (`d=0`) prefill. It must not be compared directly with the formal long-context profile above, which uses `-d 15808`.

`-d 15808` first places 15,808 tokens in the KV cache and then measures prefill of 512 additional tokens. The attention part therefore operates at a much longer sequence length, so its throughput is substantially lower than the `d=0` result. This is why an official baseline can be around 750 tok/s in the short/empty-context test but around 220 tok/s in the long-context test.

## Optimization dimensions

The B390 optimization has two complementary parts: parameter/configuration tuning and source-code/shader changes.

### 1. Runtime and build parameter tuning

- Runtime parameters select the workload shape and execution path: GPU offload, Flash Attention mode, batch/ubatch size, CPU thread count, KV-cache types, and context depth.
- The tuned Release build enables B390-specific CMake options such as `GGML_VULKAN_B390_TUNED`, shape-aware submit, and the SLM-A reshape path; rejected experimental switches remain off.
- These settings affect which kernels and scheduling policy are selected, but do not by themselves change the numerical implementation.

### 2. Source-code and Vulkan shader changes

- C++ backend/runtime changes add B390/Xe2 detection, shape-aware pipeline selection, graph submission policy, MoE routing support, and fused GDN/SSM state updates.
- Vulkan shader changes tune cooperative-matrix tiles, IQ3/IQ2/Q4_K/Q5_K dequantization and mat-vec loops, Flash Attention phases, and expert routing/compaction.
- The baseline/v1 comparison keeps the runtime benchmark parameters identical; the measured difference comes from the tuned build configuration together with these code and shader changes.

The public defaults are documented by the upstream [llama-bench README](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) and [CLI README](https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md).

## v1 changes

### 1. B390 device gating and tuned Vulkan configuration

- Added B390/Xe2 device checks around tuned paths.
- Kept alternate tiles and special shader choices behind B390-specific gates.
- Added shape-aware and feature-specific configuration paths so tuned behavior does not silently become a generic Vulkan default.
- Preserved explicit fallback switches for experimental B390 paths.

### 2. IQ3 and quantized GEMM tuning

- Tuned IQ3 F16-B matrix multiplication for Arc B390.
- Added B390-specific IQ3_XXS tile selection and shape-aware selection.
- Tuned cooperative-matrix and subgroup choices for the B390 execution model.
- Added constant-table and pair-decode variants where they matched the active selector and shape.
- Kept Q4/Q5 bitfield dequant support in the B390 tuned path.
- Kept Q4-only dequant as an opt-in probe rather than a default.

### 3. Flash Attention and prefill paths

- Added Intel Xe Flash Attention gating.
- Split the prefill path into phases to control loop and synchronization structure.
- Added specialized attention shaders for the relevant head dimensions.
- Kept a fallback path for unsupported or unsuitable shapes.
- The tested B390 working point is the split configuration selected by the tuned path; narrow barrier variants were not promoted because they did not show a stable net gain.

### 4. MoE routing and expert work

- Skip unnecessary work in the cooperative-matrix MoE path.
- Pack routed rows using expert counts.
- Add expert-count scratch and cache handling to reduce repeated route preparation.
- Add compact-expert and route-compaction infrastructure.
- Keep optional indirect expert dispatch and route-cache experiments gated and off unless explicitly enabled.
- Add device-resident MoE cache and graph-plan support used by the local server/runtime experiments.

The route changes are structural support for the B390 workload. They are not all independent performance wins, and route experiments that failed correctness were removed from the default path.

### 5. Scheduler, submit, and memory behavior

- Added B390 tuned scheduler controls and shape-aware submit selection.
- Made submit decisions sensitive to graph FLOPs rather than only node count.
- Preserved explicit limits and fallback behavior for graph submission.
- Added scheduler telemetry used to distinguish fewer submissions from actual end-to-end speedup.
- Kept descriptor, barrier, command-buffer, and scratch lifetimes conservative where a more invasive change did not show a stable benefit.

### 6. GDN and SSM state fusions

- Fuse gated delta-net state writes into the cache where the graph shape is safe.
- Fuse single-token GDN state writes.
- Fuse the convolution state copy into `SSM_CONV`.

These changes are part of the final workload path and must remain in the v1 tree. They are not reverted by the matvec experiments.

### 7. Final MUL_MAT_VEC and MUL_MAT_ID_VEC tuning

The final incremental optimization is deliberately small and B390-only:

- IQ2_S `MUL_MAT_ID_VEC`: use `NUM_ROWS=16` on the B390 path. This targets the frequent `n=8` decode shapes, including `k=2048` and `k=512` cases.
- Q4_K `MUL_MAT_VEC`: use `[[dont_unroll]]` for the block-row loop under `B390_TUNED`.
- Q5_K `MUL_MAT_VEC`: use `[[dont_unroll]]` for the block-row loop under `B390_TUNED`.

The Q4_K and Q5_K changes reduce register pressure without changing the dequantization math or data layout.

Correctness and build checks for v1:

- Vulkan `MUL_MAT_ID`: `883/883`.
- Vulkan `MUL_MAT`: `1071/1071`.
- Release Vulkan DLL build: successful.
- `git diff --check`: passed.

## Complete change inventory

The following is the file-level inventory of the `baseline -> v1` tree. The detailed behavior is grouped by subsystem above; this list makes the published scope auditable.

| Area | Files changed or added | Purpose |
| --- | --- | --- |
| User-facing arguments | `common/arg.cpp`, `common/common.cpp`, `common/common.h`, `common/chat.h`, `tools/completion/completion.cpp` | Expose and carry the tuned runtime options and chat/completion behavior. |
| Vulkan backend core | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`, `ggml/src/ggml-vulkan/CMakeLists.txt` | B390 detection, feature gates, pipeline selection, scheduler/submit policy, MoE and attention integration. |
| Vulkan shader build | `ggml/src/ggml-vulkan/vulkan-shaders/CMakeLists.txt`, `vulkan-shaders-gen.cpp`, `types.glsl` | Register new shader variants and compile-time capability/configuration plumbing. |
| Quantized matvec/dequant | `dequant_iq3_xxs.comp`, `mul_mat_vec_iq3_xxs.comp`, `mul_mat_vec_q4_k.comp`, `mul_mat_vec_q5_k.comp`, `mul_mm_funcs.glsl`, `mul_mm_id_funcs.glsl` | IQ3/IQ2/Q4/Q5 dequant, GEMM and final B390 matvec tuning. |
| GEMM and expert shaders | `mul_mm.comp`, `mul_mm_cm2.comp`, `mul_mmq.comp`, `count_experts.comp`, `compact_experts.comp` | Cooperative-matrix, quantized matrix multiplication, route counting and expert compaction. |
| Attention shaders | `flash_attn_hdim64.comp`, `flash_attn_hdim96.comp`, `flash_attn_hdim128.comp`, `flash_attn_prefill_phase_1.comp`, `flash_attn_prefill_phase_2.comp`, `flash_attn_decode_phase_1.comp`, `flash_attn_decode_phase_2.comp` | Xe-gated head-size variants and split prefill/decode phases. |
| State fusions | `gated_delta_net.comp`, `ssm_conv.comp` | GDN state updates and SSM_CONV state-copy fusion. |
| Graph/runtime | `ggml/CMakeLists.txt`, `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml/src/ggml-cpu/ggml-cpu.c`, `include/llama.h`, `src/CMakeLists.txt`, `src/llama-context.cpp`, `src/llama-graph.cpp`, `src/llama-moecache.cpp`, `src/llama-moecache.h` | Public declarations, graph planning, runtime state, CPU capability/build integration and device-resident MoE cache support. |
| Server | `tools/server/README.md`, `tools/server/server-common.cpp`, `tools/server/server-common.h`, `tools/server/server-context.cpp` | Carry the tuned context/graph/cache behavior into server execution. |
| Documentation | `README.md`, `B390-VULKAN-V1.md` | Describe baseline/v1, reproduction conditions, the four-row comparison, all changes, and references. |

## Build and reproduce

The two comparison builds used the same source/configuration shape. The only intentional code difference is the published tree: `source-baseline` at `baseline` versus `source` at `v1`.

v1 was configured with these CMake options:

```powershell
cmake -S C:\llama.cpp\source -B C:\llama.cpp\build-b390-v1-lto -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DGGML_VULKAN=ON `
  -DGGML_NATIVE=ON `
  -DGGML_LTO=ON `
  -DGGML_BUILD_TESTS=OFF `
  -DGGML_CCACHE=OFF `
  -DGGML_VULKAN_B390_TUNED=ON `
  -DGGML_VULKAN_B390_Q4_ONLY_BITFIELD_DEQUANT=ON `
  -DGGML_VULKAN_B390_SHAPE_AWARE_SUBMIT=ON `
  -DGGML_VULKAN_B390_SLM_A_RESHAPE=ON `
  -DGGML_VULKAN_B390_GEMM_CACHE_SCOPE=OFF `
  -DGGML_VULKAN_B390_GRAPH_REPLAY=OFF `
  -DGGML_VULKAN_B390_Q4Q5_BITFIELD_DEQUANT=OFF `
  -DGGML_VULKAN_B390_Q5_ONLY_BITFIELD_DEQUANT=OFF `
  -DGGML_VULKAN_B390_ROUTE_CACHE_DATA_KEY=OFF `
  -DGGML_VULKAN_B390_ROUTE_COUNT_WG128=OFF `
  -DGGML_VULKAN_B390_ROUTE_HISTOGRAM_SCAN=OFF `
  -DGGML_VULKAN_B390_ROUTE_INDIRECT=OFF `
  -DGGML_VULKAN_B390_ROUTE_SINGLE_SCAN=OFF `
  -DVulkan_INCLUDE_DIR=C:\llama.cpp\vulkan-sdk\1.4.357.0\Include `
  -DVulkan_LIBRARY=C:\llama.cpp\vulkan-sdk\1.4.357.0\Lib\vulkan-1.lib
cmake --build C:\llama.cpp\build-b390-v1-lto --config Release -j 16
```

The baseline used the same command with `source-baseline` and `build-b390-baseline`, `GGML_LTO=OFF`, `GGML_CCACHE=OFF`, and all `GGML_VULKAN_B390_*` switches at their default/off state. Both builds completed successfully and generated `llama-cli.exe` and `llama-bench.exe`. The captured v1 cache also contains `CMAKE_C_FLAGS` and `CMAKE_CXX_FLAGS` set to `-DGIT_CONFIG_NOSYSTEM=1`; this is a repository-isolation setting, not a performance flag.

The actual MSVC compile flags in both Release build databases were `/O2 /Ob2 /DNDEBUG /MD -openmp` (with the C/C++ standard switches and warning suppressions shown in `compile_commands.json`). In other words, this build used MSVC's `/O2` speed optimization level; there is no MSVC `/O3` level. `GGML_NATIVE=ON` enables native CPU feature detection/configuration, but does not change `/O2` into an `/O3` equivalent. The v1 Vulkan shaders are additionally optimized by their own workgroup, subgroup, dequantization and loop-control choices described above.

The model run conditions used for the final matvec comparison were:

```text
-ngl -1 -fa on -b 512 -ub 128
decode: -p 1 -n 64
prompt: -p 128 -n 0
```

Alternate A and B in the same time window, report the number of repetitions, mean, standard deviation, and any observed GPU frequency or thermal drift. Do not infer an end-to-end gain from queue utilization, dispatch count, or logger time alone.

## Version history

The named release points are:

1. `baseline`: the unmodified upstream tree at `46bdeb47d`.
2. the pre-sync tuned v1 tree, preserved locally as `v1-before-upstream-merge` at `a57f8a594`.
3. `v1`: merge commit `aaffe04f5`, with the pre-sync tuned tree as its first parent and upstream `master` commit `c7bda030e` as its second parent.

The v1 merge therefore includes all upstream commits through `c7bda030e`, including the upstream merge of [PR #27483](https://github.com/ggml-org/llama.cpp/pull/27483). The B390 changes remain in the first-parent side of the merge. The old exploratory chain is not pushed as the public default history.

The original local development chain contained many exploratory commits. Their subjects are preserved here for traceability, but the exploratory chain is not part of the public history:

- B390 IQ3 tile and matmul tuning.
- Xe FA gate and prefill split.
- MoE skip, routed-row packing, expert-count cache, and route infrastructure.
- Scheduler and submit threshold experiments.
- GDN and SSM_CONV state fusion.
- Final IQ2_S, Q4_K, and Q5_K matvec tuning.

## Reference projects and specifications

- [llama.cpp](https://github.com/ggml-org/llama.cpp) - upstream inference runtime and Vulkan backend.
- [ggml](https://github.com/ggml-org/ggml) - tensor graph and backend infrastructure used by llama.cpp.
- [Vulkan Cooperative Matrix](https://registry.khronos.org/vulkan/specs/latest/html/chap47.html) - cooperative matrix execution model.
- [VK_EXT_subgroup_size_control](https://registry.khronos.org/vulkan/specs/latest/html/chap50.html) - subgroup size selection and control.
- [VK_KHR_shader_integer_dot_product](https://registry.khronos.org/vulkan/specs/latest/html/chap50.html) - integer dot-product capability used by quantized paths where supported.
- Intel Arc B390/Xe2 hardware - the target device for the gated paths in this fork.
- Qwen3.6 35B-A3B GGUF - the target MoE workload used for end-to-end measurements.

## Reference upstream pull requests

These upstream PRs were used as design references or comparison points. A reference does not mean that the full PR was copied or that its result is expected to reproduce on B390.

- [PR #24404](https://github.com/ggml-org/llama.cpp/pull/24404) - Xe-LPG cooperative-matrix work.
- [PR #24406](https://github.com/ggml-org/llama.cpp/pull/24406) - Intel Xe Flash Attention work.
- [PR #24407](https://github.com/ggml-org/llama.cpp/pull/24407) - GEMM, group GEMM, dequant, and expert pipeline directions.
- [PR #25483](https://github.com/ggml-org/llama.cpp/pull/25483) - skipping invalid or unnecessary MoE work.
- [PR #25952](https://github.com/ggml-org/llama.cpp/pull/25952) - fused MoE weighted expert reduction.
- [PR #26438](https://github.com/ggml-org/llama.cpp/pull/26438) - Intel Xe quantized-path tuning in OpenCL.
- [PR #26829](https://github.com/ggml-org/llama.cpp/pull/26829) - Arc B70 Pro Vulkan tuning package; only shape-compatible ideas were considered for B390.
- [PR #27449](https://github.com/ggml-org/llama.cpp/pull/27449) - larger-batch IQ3_S mat-vec handling.
- [PR #27621](https://github.com/ggml-org/llama.cpp/pull/27621) - extending MoE fusion to speculative decoding.
- [PR #27909](https://github.com/ggml-org/llama.cpp/pull/27909) - mat-vec row tuning for batched inference.
- [PR #27925](https://github.com/ggml-org/llama.cpp/pull/27925) - `mul_mat_id` K/N padding direction.
- [PR #27483](https://github.com/ggml-org/llama.cpp/pull/27483) - upstream model-loading peak-RAM reduction; included through the full upstream sync in v1.

## Scope and maintenance

This is a B390/Qwen3.6 tuned fork. Keep the B390 gates around hardware-specific behavior, retain the generic fallback paths, and re-run Vulkan backend correctness after changing shader selection, workgroup geometry, route handling, or state fusion. The cooperative-matrix ID prototype and the single-scan route are not part of v1 because they either lost end-to-end throughput or failed correctness.
