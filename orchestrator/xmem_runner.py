"""Invokes predict_vram.py inside the xmem-cpu container and classifies the outcome.

Never builds a shell string from user input — the model name is user-supplied
and subprocess here always takes an argv list (no shell=True), so it can't be
used for command injection.
"""

import json
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

CONTAINER_NAME = "xmem-cpu"
SCRIPT_PATH = "/work/xMem/predict_vram.py"
WORKDIR = "/work/xMem"

# CPU-only profiling time scales with model size x batch_size x seq_len, not
# with network speed — there's no weight download in this path (load_model()
# uses AutoConfig + from_config, never from_pretrained, so only a small
# config.json is fetched). A 0.6B model has been observed to take ~10-15 min
# on one dev machine, so 15 min was too tight; 30 min gives real headroom.
TIMEOUT_SECONDS = 1800

# Keys predict_vram.py's argparse/--json actually accepts. Everything else on the
# incoming payload (type, deployment, warnings, ...) is dropped before it ever
# reaches the subprocess.
XMEM_KEYS = {
    "model",
    "batch_size",
    "seq_len",
    "resolution",
    "dtype",
    "mode",
    "max_new_tokens",
    "gpu_memory_gb",
    "model_config",
}

# Not a user-configurable value: this is xMem's internal "candidate GPU memory"
# ceiling used while estimating placement, unrelated to the deployment budget.
FIXED_GPU_MEMORY_GB = 80

PHASE_PREPARING = "모델 로딩 및 준비 중…"
PHASE_PROFILING = "CPU 프로파일링 실행 중…"
PHASE_FINALIZING = "결과 계산 중…"


def build_xmem_args(payload: dict[str, Any]) -> dict[str, Any]:
    params = payload.get("params") or {}
    # dtype lives on the top-level payload, not in params — it must be applied
    # after the **params spread so it can't be shadowed by a stray params
    # entry. Without it, predict_vram.py never receives --dtype and falls
    # back to its own argparse default ("fp16"), silently re-scaling the
    # FP32-profiled result with the wrong DTYPE_SCALE factor for any request
    # that isn't fp16 (fp32 comes back at half its real size).
    merged = {
        "model": payload["model"],
        "batch_size": payload["batch_size"],
        **params,
        "dtype": payload["dtype"],
        # Same reasoning as dtype above: applied after **params so it can't be
        # shadowed, and its presence lets predict_vram.py skip its own
        # AutoConfig.from_pretrained() fetch and reuse what the web app
        # already fetched when the user picked this model.
        "model_config": payload.get("model_config_snapshot"),
    }
    args = {k: v for k, v in merged.items() if k in XMEM_KEYS and v is not None}
    args["gpu_memory_gb"] = FIXED_GPU_MEMORY_GB
    return args


def cache_key_for(xmem_args: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(xmem_args, sort_keys=True).encode("utf-8")).hexdigest()


def parse_result(stdout: str) -> Optional[dict]:
    # predict_vram.py prints json.dumps(result, indent=2) — a multi-line block.
    # Find the last line that starts a top-level object (a "\n{" at column 0)
    # and parse from there instead of assuming the JSON is on the last line.
    idx = stdout.rfind("\n{")
    start = 0 if stdout.lstrip().startswith("{") and idx == -1 else idx + 1
    try:
        return json.loads(stdout[start:])
    except json.JSONDecodeError:
        return None


@dataclass
class RunOutcome:
    status: str  # "succeeded" | "failed"
    result: Optional[dict] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


def _tail(stdout: str, stderr: str) -> str:
    return f"{stdout[-2000:]}\n{stderr[-2000:]}".strip()


def _phase_for_line(line: str) -> Optional[str]:
    if "INFERENCE loop" in line:
        return PHASE_PROFILING
    if "Estimated Result" in line:
        return PHASE_FINALIZING
    return None


def run_xmem(xmem_args: dict[str, Any], on_progress: Optional[Callable[[str], None]] = None) -> RunOutcome:
    """Streams the subprocess's stdout so on_progress can be told when
    predict_vram.py moves from setup into the actual (slow) profiling loop —
    predict_vram.py prints "Using ... INFERENCE loop." right as that starts,
    and its final "Estimated Result" banner right before it emits the JSON."""
    report = on_progress or (lambda _msg: None)
    report(PHASE_PREPARING)

    try:
        proc = subprocess.Popen(
            [
                "docker", "exec", "-i", "-w", WORKDIR, CONTAINER_NAME,
                "python", SCRIPT_PATH,
                "--json", json.dumps(xmem_args),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except FileNotFoundError as exc:
        return RunOutcome(status="failed", error_code="internal", error_message=f"'docker'를 찾을 수 없습니다: {exc}")

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def _pump(stream, sink: list[str], watch_for_phase: bool) -> None:
        for line in stream:
            sink.append(line)
            if watch_for_phase:
                phase = _phase_for_line(line)
                if phase:
                    report(phase)
        stream.close()

    reader_out = threading.Thread(target=_pump, args=(proc.stdout, stdout_chunks, True), daemon=True)
    reader_err = threading.Thread(target=_pump, args=(proc.stderr, stderr_chunks, False), daemon=True)
    reader_out.start()
    reader_err.start()

    try:
        proc.wait(timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return RunOutcome(
            status="failed",
            error_code="timeout",
            error_message=(
                f"CPU 프로파일링이 {TIMEOUT_SECONDS // 60}분을 초과했습니다. "
                "네트워크가 아니라 모델 크기·배치 크기·시퀀스 길이에 비례해 CPU 연산 시간이 늘어나는 것이니, "
                "더 작은 설정으로 다시 시도해보세요."
            ),
        )
    finally:
        reader_out.join(timeout=5)
        reader_err.join(timeout=5)

    stdout = "".join(stdout_chunks)
    stderr = "".join(stderr_chunks)

    parsed = parse_result(stdout)

    # Success/failure is judged from the JSON body, never from the exit code:
    # predict_vram.py prints {"status": "error", ...} AND calls sys.exit(1) on
    # its own caught errors, so returncode alone can't distinguish that from a
    # crash with no JSON at all.
    if parsed and parsed.get("status") == "ok":
        return RunOutcome(status="succeeded", result=parsed)

    if parsed and parsed.get("status") == "error":
        return RunOutcome(
            status="failed",
            error_code="model_load_failed",
            error_message=str(parsed.get("error", "모델을 불러오지 못했습니다.")),
        )

    if proc.returncode == -9 or "Killed" in stderr:
        return RunOutcome(
            status="failed",
            error_code="oom",
            error_message="CPU 메모리 부족으로 프로파일링 프로세스가 종료되었습니다.",
        )

    return RunOutcome(status="failed", error_code="internal", error_message=_tail(stdout, stderr))
