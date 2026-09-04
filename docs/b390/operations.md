# B390 v1 Operations

The formal versions are `baseline` and `v1`.

## Test model

Use this exact GGUF file for both builds:

```text
C:\models\Qwen3.6-35B-A3B-DSV4Pro-SFT-GPT56Sol-RL-Agent-GGUF\Qwen3.6-35B-A3B-DSV4Pro-SFT-GPT56Sol-RL-Agent-LynnStyle-Q3_16G-LynnStyle-imatrix.gguf
```

## Benchmark

Run the public comparison profile against each build:

```text
llama-bench.exe -m <model> -p 512 -n 128 -pg "512,64" -d <0|15808> -b 2048 -ub 512 -t 16 -fa auto -ctk f16 -ctv f16 -ngl -1 -dev Vulkan0 -r 5 -o csv
```

In PowerShell, keep `"512,64"` quoted so it is passed to `llama-bench` as one `-pg` value. Run it once with `-d 0` and once with `-d 15808`. This command produces prefill-only, generation-only, and combined prompt-plus-generation rows. The build-specific options are recorded in the compilation parameters in [B390-VULKAN-V1.md](../../B390-VULKAN-V1.md).

The optimization has two parts: identical runtime benchmark parameters are used for the A/B comparison, while the v1 build also enables B390-specific build switches and contains the source-code/Vulkan shader changes documented in [B390-VULKAN-V1.md](../../B390-VULKAN-V1.md).

Record the version or commit, CMake options, model path, driver, command, repetition count, mean, standard deviation, and GPU frequency or temperature changes. Full build instructions and the machine snapshot are in [B390-VULKAN-V1.md](../../B390-VULKAN-V1.md).

## Correctness and smoke

Run the Vulkan backend checks:

```text
test-backend-ops.exe test -b Vulkan0 -o MUL_MAT,MUL_MAT_ID -j 1
```

Run a fixed-prompt, fixed-seed model smoke for both formal builds and compare exit code, model loading, and deterministic output prefix.
