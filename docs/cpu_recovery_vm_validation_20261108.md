# CPU recovery validation on INUOSVM-1, 2026-11-08 KST

This report supplements the supplied handoff; its historical results are not treated as VM validation.

## Preserved baseline and source

- Original checkout: `/home/inuos/CloudGPUFit`, branch `codex/vram-native-inference-20261103`, HEAD `7f822d76af13bea39191beb1852ec67513b18f15`. Only pre-existing untracked `.cpu-prediction.lock` was present.
- Independent worktree: `/home/inuos/CloudGPUFit-cpu-recovery-20261108` at the same base, plus the supplied patch and the corrections below. No fetch, push, install, reset, clean or prune was used.
- Original predictions: `/home/inuos/cloudgpufit-predictions/native-v3-t4-j6`: 244 successful, 17 error, 81 absent. Historical manifest remains `running`, completed 260; no prediction process was active.
- Original `xmem-cpu` was running only `sleep infinity`. Source `/home/inuos/xmem-work` and HF cache mounts were confirmed before work.
- Host has 24 logical CPUs, approximately 47 GiB RAM, no swap. About 45 GiB was available at admission.
- ZIP entries and extracted package files match every provided SHA-256. Original results/logs (787 files) and original mounted predictor hashes were inventoried in `recovery-audit-20261108/before.json`.

## Independent review and corrections

The patch preserves native dtype, one warmup, three measured invocations, full generate, lifetime matching, resident storage order, and upstream allocator transitions/counters. Stable sorting follows event filtering; frees outside measured regions are retained. `SummaryTrace` suppresses only `_record`, whose upstream implementation does not alter byte counters. The parser still retains O(E) selected events and the allocator retains O(E) maps/counters.

VM findings required these additional corrections:

1. Docker 29.7.2 emits a `commit --pause=false` deprecation warning on stdout. The supplied runner treated that warning plus the SHA as an image ID. The first attempt (`recovery-check-01`) therefore failed at container creation, before model inference. The runner now resolves the image ID with `docker image inspect`.
2. New predictor files live at a separate read-only `/recovery` mount, with `/work/xMem` on PYTHONPATH. This avoids creating a new helper mountpoint inside the read-only original source. No file overlay requires editing original xMem.
3. An explicitly supplied local snapshot can be reused after checking its Python/torch/Transformers/CUDA/GPU environment against the prepared container; its ID and reuse source are recorded. This avoids repeatedly snapshotting the approximately 5.2 GB writable layer during validation.
4. The new runtime helper is included in the pipeline code fingerprint. Reference comparisons now require matching revision in both input and prediction, in addition to exact request/thread equality.
5. Measured inference and profiler stop have separate phase timings. Profiler start/stop still surrounds exactly the same three measured invocations.
6. Results record actual container ID, host PID/cgroup when observable, hard RAM/swap/CPU limits, mounts and final Docker state.

## Actual VM tests

Environment: Python 3.11.11, torch 2.6.0+cu124, Transformers 4.52.4, CUDA runtime 12.4, GPU unavailable. Allocator lock file hashes match the pinned xMem source. Validation image `cloudgpufit-cpu-validation:20261108` is local only; image ID is recorded in manifests.

- Host regressions: 40 tests, 36 passed / 4 allocator tests skipped because the host environment is not the prepared runtime.
- Isolated VM regressions: all 40 passed, including identical full allocated/reserved counter sequences for overlapping events and identical peaks for the same real tiny GPT-2 trace.
- Isolated VM integration: all 15 cases passed, covering tiny GPT-2/T5/ViT/ResNet/MobileNetV2 in FP32/FP16/BF16 and T5 self/cross KV.
- Real Docker lifecycle checks: exit 7 and exit 137 without OOM evidence remain process errors; a 64 MiB job allocating 256 MiB reports `OOMKilled=true`/`cgroup_oom`; timeout and SIGTERM cancellation have distinct classifications. Every test job container was removed and the original container stayed running. `lifecycle-results.json` records actual states.
- Observed representative cgroup: memory.max=34359738368 (32 GiB), memory.swap.max=0, cpu.max=400000/100000 (4 CPUs). `/work`, HF cache and `/recovery` were all read-only. Evidence: `active-container-inspect.json`, `active-cgroup-mountinfo.txt`.

## Limits

Graceful Python-handled cancellation and ordinary errors are tested. SIGKILL/host failure cannot execute Python finally; Docker containers/images may need cleanup by their recorded unique IDs after such a failure. CPU/GPU random weight and allocator initial-state differences remain a separate accuracy question. Historical elapsed times were collected under different concurrent loads and are not controlled speedup measurements.

## Original-size results

All cases ran sequentially, with threads=4, a 4-CPU Docker quota, 32 GiB hard RAM, no swap and a 6 GiB admission reserve. No other prediction or VM test was active during these cases. Python/torch/Transformers versions, allocator hashes, plan/revisions hashes, source hashes, snapshot ID and runner hash are recorded in each manifest. This checkout remains based on `7f822d7`; the validated patch and its SHA-256 identify the uncommitted modifications.

| Case ID | allocated bytes (old → new) | reserved bytes (old → new) | allocated / reserved delta | elapsed s | peak RSS GiB | cgroup peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| core__decoder_llm__small__fp16__01 | 310894080 → 310894080 | 346030080 → 346030080 | 0 / 0 bytes | 335.53 | 2.682 | 2.576 |
| core__decoder_llm__small__fp32__01 | 593162240 → 593162240 | 694157312 → 694157312 | 0 / 0 bytes | 35.37 | 2.850 | 2.643 |
| core__text_encdec__small__fp32__01 | 252521472 → 252521472 | 278921216 → 278921216 | 0 / 0 bytes | 24.87 | 1.876 | 1.662 |
| core__vision_cls__large__fp32__01 | 1229447168 → 1229447168 | 1237319680 → 1237319680 | 0 / 0 bytes | 11.53 | 2.143 | 1.909 |
| core__vision_cls__small__fp32__01 | 31478784 → 31478784 | 52428800 → 52428800 | 0 / 0 bytes | 6.03 | 0.659 | 0.421 |
| core__decoder_llm__small__fp32__12 | no successful reference → 1479036928 | no successful reference → 1965031424 | not comparable | 111.38 | 30.744 | 30.619 |

Elapsed time includes process startup and runner overhead; phase times below do not sum to it. RSS and cgroup accounting have different meanings (shared pages/file cache); they are not interchangeable or additive.

| Case ID | initialization | warmup | measured inference (3) | profiler stop | export | parse | lifetime matching | allocator |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| core__decoder_llm__small__fp16__01 | 13.788 | 71.654 | 223.204 | 6.400 | 2.282 | 6.468 | 0.293 | 5.697 |
| core__decoder_llm__small__fp32__01 | 4.569 | 1.838 | 6.833 | 3.200 | 1.276 | 4.298 | 0.333 | 6.515 |
| core__text_encdec__small__fp32__01 | 3.049 | 0.803 | 2.760 | 2.862 | 1.046 | 3.083 | 0.509 | 5.618 |
| core__vision_cls__large__fp32__01 | 6.218 | 0.425 | 1.224 | 0.051 | 0.028 | 0.072 | 0.004 | 0.126 |
| core__vision_cls__small__fp32__01 | 2.248 | 0.054 | 0.086 | 0.027 | 0.015 | 0.041 | 0.003 | 0.070 |
| core__decoder_llm__small__fp32__12 | 4.503 | 12.017 | 51.709 | 6.597 | 2.626 | 9.113 | 1.325 | 13.769 |

The T5 reference retained self KV storage 1,572,864 bytes and cross KV storage 6,291,456 bytes, with no analytic additions. Both original and new measured invocation counts are three.

## Findings and remaining recovery

The old `prepare_recovery.py --repo .` failure is consistent with a non-repository working directory. Providing the verified absolute repository path successfully prepared the independent worktree. The supplied patch had not been applied before this task.

The previous OOM-associated GPT-2 small FP32 case 12 (B4, input768, generate128) now completed successfully under the hard cap. Its measured inference took 51.709 s; warmup peak RSS was 2.116 GiB, and peak RSS had already reached 30.400 GiB by measured inference completion, before JSON export/parsing/allocator replay. Its final peak RSS was 30.744 GiB. The dominant RAM increase therefore persists inside profiled CPU inference in this case. Phase evidence alone does not separate native profiler retention, live tensor workspaces and CPU allocator caching; that attribution remains open.

GPT-2 small FP16 case 01 spent 71.654 s warming up and 223.204 s in three measured invocations, versus 6.400 s stopping the profiler, 6.468 s parsing and 5.697 s simulating. Actual CPU inference remains the dominant FP16 time cost on this host. Its total was 335.53 s; the historical 331.7 s was under a different load/quota regime, so no speedup is claimed. FP32 took 35.37 s total in this validation. Neither elapsed CPU time nor CPU RSS is used as a GPU prediction.

The patch demonstrably preserves the checked byte results and bounds failure/cleanup behavior. It does not establish a RAM-reduction factor: old phase/peak measurements are absent and the historical victim RSS is not an equivalent single-case peak measurement. The native profiler/CPU path still requires substantial memory. The recovered case nearly consumed the 32 GiB cap, so starting all remaining heavy GPT-2 XL/T5-3B conditions would be premature; their safe completion time and memory are unverified.

Original historical successes remain 244. Five successful references were rerun and matched exactly; this is evidence of equivalence, not a claim that all 244 have been individually rerun. One of the 98 previously failed/absent conditions was recovered, leaving **97** (16 original errors plus 81 absent). `remaining-after-validation.json` lists those exact conditions and the recovered new-version case. No historical success was copied into a new-version dataset. `--remaining-from` against only the original directory would still select 98; use the audited remaining list to avoid rerunning the recovered case.

Next work should first identify memory retention within the profiled inference stage under the same dtype/invocation/allocator semantics, using bounded representative jobs. Any design that separates profiler sessions, changes measured invocation count, or replaces CPU arithmetic must have a separate version and fixed-trace/lifetime equivalence validation. FP16 kernel cost also remains a separate performance issue. A defensible completion-time estimate for the remaining 97 is not available from these six cases.

## Final preservation and artifacts

`preservation-check.json` confirms all 787 original result/log files have the same SHA-256, original mounted predictor sources have identical SHA-256, original checkout status is unchanged, and the new helper was not created in original xMem. Only the independent worktree, new result/audit directories and a local validation image were created. At completion Docker showed only the original `xmem-cpu`, still running the original sleep process. The validation image is deliberately retained for reproducibility; no remote Git operation was performed.

- `/home/inuos/cloudgpufit-predictions/recovery-check-01`: preserved Docker 29 compatibility failures, no inference.
- `recovery-check-02`: two GPT-2 references.
- `recovery-check-03`: T5 and two Vision references.
- `recovery-check-04`: recovered former OOM-associated GPT-2 condition.
- `recovery-audit-20261108`: regression/integration logs, lifecycle tests, source/result inventories, actual cgroup/mount evidence, per-case summaries, and the remaining-condition list.
