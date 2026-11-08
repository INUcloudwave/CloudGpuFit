# Single-request CPU VRAM prediction — VM validation, 2026-11-08

The default is now **zero warmup calls and one measured request**. Repeating a request is not required to predict one request's peak VRAM. The earlier 1+3 implementation replayed three measured lifetimes into one allocator; it was not averaging estimates. CPU warmup was never replayed into that allocator. This change deliberately defines a single-request policy, with a separate prediction/cache version.

## Code and policy

- Independent worktree: `/home/inuos/CloudGPUFit-cpu-singlepass-20261108`; original checkout and earlier validated recovery worktree are preserved.
- Base commit: `7f822d76af13bea39191beb1852ec67513b18f15`, plus recovery fixes and this change. Prediction version: `4-native-inference-single-request`.
- `cpu_tensor_trace.py`: default warmup=0, measured=1; lifetime matching requires the configured number of regions. Model/input residency and upstream allocator transitions are unchanged.
- `predict_vram.py`: CLI/JSON accepts the counts, with defaults 0/1; `trace_summary` records actual counts, `cold_single_request` policy, and allocator initial state.
- `vram_policy.py`, `orchestrator/cache_key.py`: version bumped to avoid silently mixing old repeated-request results with the new policy.
- `tools/smoke_native_inference.py` and trace regressions assert zero warmup and one measured call. An actual tiny GPT-2 test verifies `generate()` is called exactly once. Explicit legacy settings 1/3 are also tested (four calls) for diagnostic comparisons; ordinary jobs do not use them.
- Full generation length, native dtype, KV tracking, input shapes, random seed and allocator source lock are unchanged. One `generate()` still contains prefill and the token-by-token decoding needed to produce the requested tokens.

## Actual VM validation

Python 3.11.11, torch 2.6.0+cu124, Transformers 4.52.4, CPU only; same local prepared snapshot, four threads, 4-CPU quota, 32 GiB hard limit, swap disabled. Cases ran sequentially with no other prediction active. Baselines are the earlier 1+3 VM runs in recovery-check-02/03/04. Each speed ratio is one observed pair, not a repeated statistical benchmark.

All 43 VM regression tests and all 15 tiny-model integrations passed. All six original-size cases below completed with warmup=0 and measured=1. **Allocated and reserved both matched the earlier 1+3 result byte for byte in every case.** This validates these cases; it is not a claim that all 342 conditions were rerun.

| Case | old seconds | new seconds | observed speed ratio | old peak RSS GiB | new peak RSS GiB | allocated bytes | reserved bytes | allocated/reserved delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| core__decoder_llm__small__fp32__01 | 35.37 | 15.89 | 2.23x | 2.850 | 2.018 | 593162240 | 694157312 | 0 / 0 bytes |
| core__vision_cls__small__fp32__01 | 6.03 | 5.89 | 1.02x | 0.659 | 0.624 | 31478784 | 52428800 | 0 / 0 bytes |
| core__text_encdec__small__fp32__01 | 24.87 | 12.06 | 2.06x | 1.876 | 1.167 | 252521472 | 278921216 | 0 / 0 bytes |
| core__vision_cls__large__fp32__01 | 11.53 | 10.13 | 1.14x | 2.143 | 1.869 | 1229447168 | 1237319680 | 0 / 0 bytes |
| core__decoder_llm__small__fp32__12 | 111.38 | 43.24 | 2.58x | 30.744 | 31.426 | 1479036928 | 1965031424 | 0 / 0 bytes |
| core__decoder_llm__small__fp16__01 | 335.53 | 104.55 | 3.21x | 2.682 | 1.564 | 310894080 | 346030080 | 0 / 0 bytes |

The largest tested CPU RAM case remained high: GPT-2 small FP32 case12 rose from peak RSS 30.744 to 31.426 GiB despite the shorter run. Repetition removal is a demonstrated time improvement, not a general RAM fix. Further work should identify memory retention during profiled inference. FP16 also still needs about 79.9 seconds for one measured generate call on this VM; native CPU execution remains its main cost.

No remaining 97-case sweep was started and no additional missing/error condition was recovered in this step; these six were comparisons against the earlier successful VM representatives. Old results retain their original versions and are not copied into new-version outputs. The prior 244 historical successes and recovered case remain preserved.

## Evidence

`/home/inuos/cloudgpufit-predictions/singlepass-audit-20261108` contains test logs, the representative runner, comparisons, source checksums and preservation results. Individual inputs, source/environment manifests, byte values, phase logs and Docker evidence are in `singlepass-check-01` through `singlepass-check-06`. Earlier recovery artifacts and original source/result hashes were rechecked after this run. No dependency install, remote Git operation or modification to original xMem was performed.
