# CPU prediction OOM diagnosis and bounded recovery

## Evidence and limits

The diagnostics archive was collected at 2026-11-05 13:51:06 KST. This is a stopped
run, not a live VM inspection on November 8. Its manifest records CloudGPUFit
`7f822d76af13bea39191beb1852ec67513b18f15`, torch 2.6.0+cu124,
Transformers 4.52.4, Python 3.11.11, threads=4 and jobs=6.
The copied predictor and trace helper hashes match the manifest, and the copied
locked allocator files match the upstream lock. xMem is pinned to
`1ee4bd08dda277a6248abee5b6d40e0d6067efaa`.

- Manifest: 260 completed, 244 successful, elapsed 154837.26 seconds (43.01 hours).
- Files: 244 successful predictions and 17 errors. The extra error is the last,
  manually interrupted `core__decoder_llm__large__fp16__06`, after the manifest's
  last update. 81 conditions have no prediction file.
- Kernel: 16 Python processes in the same Docker cgroup were killed by global
  host OOM. Every one has exactly one corresponding error JSON saved 0–2 seconds
  later. This is a strong one-to-one temporal association, not a recorded
  condition-to-host-PID mapping from the old runner.
- Docker inspect: Memory=0, MemorySwap=0. `/home/inuos/xmem-work` is bind-mounted
  read/write at `/work`. The old 36GB reservation was not a hard limit.
- `gpuGetDeviceCount failed with code 35` appears in all 244 successful logs as
  well as 5 failed logs. It does not distinguish failures and is not established
  as their cause.

## OOM association table

The suffix below follows `core__decoder_llm__`. RSS is the sum of anon and file RSS
from the kernel victim line, converted from KiB to GiB. Delta is error JSON mtime
minus the kernel kill timestamp, using KST.

| Condition suffix | Host victim PID | RSS GiB | Delta seconds |
| --- | ---: | ---: | ---: |
| small__fp32__11 | 475381 | 18.51 | 2 |
| small__fp32__12 | 475423 | 17.51 | 1 |
| small__fp32__14 | 475503 | 16.69 | 0 |
| small__fp16__14 | 474934 | 14.57 | 1 |
| small__fp16__12 | 474853 | 14.88 | 1 |
| medium__fp16__11 | 486170 | 28.73 | 2 |
| medium__fp16__12 | 502921 | 29.45 | 1 |
| medium__fp32__10 | 509327 | 18.31 | 1 |
| medium__fp16__14 | 508918 | 20.98 | 1 |
| medium__fp32__11 | 509367 | 43.78 | 2 |
| medium__fp32__14 | 509481 | 27.51 | 2 |
| medium__fp32__12 | 509406 | 44.41 | 2 |
| medium__bf16__11 | 509923 | 22.25 | 1 |
| medium__bf16__12 | 509964 | 25.71 | 1 |
| large__fp16__03 | 514053 | 46.25 | 2 |
| large__fp16__04 | 518456 | 46.25 | 2 |

GPT-2 XL FP16 case 03 (B4, input256, output64) ran about 11.44 hours before OOM;
case 04 (B8, input256, output64) ran about 12.87 hours. Their console intervals
show a single prediction at that stage, so reducing concurrency alone does not
solve these cases.

The successful XL B2 case took 20957.7 seconds and retained 371334 tensor
lifetimes. The old code additionally retained profiler results, all parsed trace
events and allocator history. `Trace._record` copies `list(self._segments)` at
every event: shallow reference lists, not deep snapshots. Its total reference
storage grows with the sum of segment counts across events. No phase-level RSS
was recorded, so the exact point of each old peak cannot be recovered.

CPU FP16 is another strong time signal: medium case 04 took 12882.4 seconds in
FP16 versus 130.8 seconds in FP32 and 196.2 seconds in BF16. These are whole-case
times under different concurrent loads, not a controlled kernel speed ratio.
Do not silently replace FP16 CPU execution with FP32 or rescale its trace.

## Changes

1. Keep warmup=1, measured invocations=3, native dtype, full generate, lifetime
   matching, timestamp tie handling and the pinned allocator's transitions.
2. Incrementally decode Chrome trace events using the Python standard library;
   retain only CPU memory events and inference annotations before stable sorting.
   The exporter is not assumed to emit chronological order. Matching still keeps
   frees outside inference regions, repeated addresses and unfreed lifetimes.
3. End the profiler helper's scope before parsing. This drops its references;
   it does not guarantee immediate RSS return to the OS. Model/input memory and
   O(number of memory events) analysis storage still remain.
4. Use the real upstream Trace counter methods with only `_record` disabled.
   Upstream allocator files and lock hashes stay unchanged. Full history can be
   requested with `--allocator_history` for diagnostics.
5. Emit low-frequency phase start/end records with PID, current/max RSS,
   available cgroup memory counters and elapsed time. Include Python traceback
   and stage in predictor errors. Distinguish simulation failures from a known
   allocator capacity exception.
6. Add a separate sequential recovery runner. It snapshots the prepared local
   CPU container's writable layer, mounts original xMem/cache read-only and
   overlays the new predictor files read-only. No package install or source
   synchronization into the old `/work/xMem` is required.
7. Each new case runs with its own Docker hard memory limit, swap disabled,
   timeout and cleanup on cancellation. Docker's OOMKilled state, exit code and
   last phase are saved before removal. Exit 137 alone is not labeled OOM.
8. Require fresh output directories. Compare both allocated and reserved exactly
   against an optional old successful reference, and stop on mismatch or
   incompatible inputs. Never rerun old successes through `--remaining-from`.
9. Block accidental legacy shared-container runs unless explicitly opted into
   with `--allow-unbounded-legacy-run`; preflight-only remains available.

The default recovery cap of 32GiB with a 6GiB host admission reserve is a starting
configuration for the approximately 47GiB VM, not a prediction of required RAM.
The default one-hour timeout is an operational bound, not a claim that every
original-size case will complete in one hour. A cap/timeout failure is recorded
and is not automatically retried. Other host workloads can still create memory
pressure after admission; a job cap does not reserve host memory.

## Validation completed here

- Python 3.12, torch 2.6.0+cpu, Transformers 4.52.4 in an isolated review runtime.
  This is not the VM's Python 3.11/cu124 image.
- 39 regression tests passed, including exact equality of every allocated and
  reserved counter in an overlapping synthetic event sequence, and exact peak
  equality when replaying the same real tiny GPT-2 Chrome trace through upstream
  and summary modes. Tests also cover chunk boundaries, malformed JSON, address
  reuse, unfreed memory, error classification and timeout cleanup logic.
- Tests load the actual pinned allocator/profiler modules while bypassing the
  unrelated package-level training estimator import. No allocator policy or
  byte counters are mocked in equivalence tests.
- The existing offline smoke suite also passed all 15 cases: tiny GPT-2, T5,
  ViT, ResNet and MobileNetV2, each in FP32/FP16/BF16. This exercises actual CPU
  forward/generate, including T5 self/cross-KV observations, on reduced models.
- No Docker daemon is available in this review runtime. Docker lifecycle tests
  validate construction and cleanup logic with mocked CLI calls; actual memory
  enforcement, snapshot mounts and cancellation need the VM check below.
- No original-size VM workload or new GPU instance has been started. No measured
  speedup or RAM reduction is claimed for the failing workloads yet.

## First VM check: two successful small references

Use a separate checkout of this branch. Do not run `--prepare` in the original
checkout or reuse the original result directory. Keep the existing `xmem-cpu`
container running idle because its prepared dependencies are used for the local
snapshot. The recovery runner never uploads that snapshot.

```bash
python3 -u tools/run_prediction_recovery.py \
  --output-root ~/cloudgpufit-predictions/recovery-check-01 \
  --case-id core__decoder_llm__small__fp32__01 \
  --case-id core__decoder_llm__small__fp16__01 \
  --compare-with ~/cloudgpufit-predictions/native-v3-t4-j6 \
  --threads 4 --memory-gib 32 --host-reserve-gib 6 --timeout-seconds 3600
```

Inspect each prediction's `reference_comparison`, phase logs and manifest.
Both allocated and reserved deltas must be zero before treating the new path
as equivalent on these cases. Do not widen tolerance if they differ.

Then validate one previous failure, for example
`core__decoder_llm__small__fp32__12` (B4, input768, output128), using a different
output root. Also check T5 self/cross-KV and one Vision reference before expansion.
Test cancellation and a deliberately small job cap on a tiny workload, checking
that only the uniquely named job container disappears and the old container and
results remain intact. An original-size failure has no successful old prediction
to use as an equality reference.

Only after these checks, `--remaining-from ~/cloudgpufit-predictions/native-v3-t4-j6`
selects the 17 errors and 81 missing conditions (98 total in this snapshot).
It does not copy old successes into a new-version accuracy dataset. Keep those
as explicitly versioned historical references. Three-to-one invocation reduction,
GPU reserved initial-state alignment, input-value alignment and FakeTensor/meta
execution are outside this change.
