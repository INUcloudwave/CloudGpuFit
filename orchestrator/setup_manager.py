"""Ensures the xmem-cpu container exists and is running — a Python port of
CloudGPUFit/setup.sh's steps, run directly by the orchestrator instead of by
shelling out to bash.

Why not just call `bash setup.sh`? On this Windows host, the plain `bash` on
PATH resolves to the WSL launcher stub (C:\\...\\WindowsApps\\bash.exe), not
Git Bash — and since WSL was installed with `--no-distribution`, that stub
has no Linux distro to run the script in, so it fails. Rather than hardcode a
path to Git's bash.exe and hope it's there, every step setup.sh performs is
just `docker`/`git` CLI calls, so this reimplements them with subprocess argv
lists (never shell=True) that work the same from Python on any OS.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"
CONTAINER_NAME = "xmem-cpu"
XMEM_REPO_URL = "https://github.com/Stone-ResearchLife/xMem"
XMEM_COMMIT = "1ee4bd08dda277a6248abee5b6d40e0d6067efaa"

# Where the CloudGPUFit repo (predict_vram.py + the patch file) lives.
# Defaults to this orchestrator/ directory's parent, i.e. the repo root a
# plain `git clone` produces (orchestrator/ sits directly under it) — not a
# "CloudGPUFit" folder nested one level further down.
CLOUDGPUFIT_DIR = Path(os.environ.get("CLOUDGPUFIT_DIR", Path(__file__).parent.parent)).resolve()

# Host directory bind-mounted into the container at /work — same default
# setup.sh uses (~/xmem-work), so a container created by either path is
# interchangeable.
WORK_DIR = Path(os.environ.get("XMEM_WORK_DIR", Path.home() / "xmem-work")).resolve()

# Mounted into the container so models already downloaded survive a container
# recreate (`docker rm xmem-cpu` + re-provision) instead of re-downloading.
HF_CACHE_DIR = Path(os.environ.get("HF_CACHE_DIR", Path.home() / ".cache" / "huggingface")).resolve()

# Written inside the CONTAINER's own filesystem (not under /work) right after
# pip install succeeds. Deliberately not on the host mount: a `docker rm` +
# recreate gets a fresh container filesystem with none of the pip packages,
# and this marker must disappear along with them so readiness is re-checked
# instead of trusting host-side state that survived the recreate.
PIP_READY_MARKER = "/root/.cloudgpufit_pip_ready"

PULL_TIMEOUT_SECONDS = 1800
PIP_INSTALL_TIMEOUT_SECONDS = 1800

PIP_PACKAGES = [
    "sortedcontainers==2.4.0",
    "pydantic~=2.10.6",
    "fire==0.7.1",
    "ures==1.9.1",
    "datasets~=3.0.2",
    "tqdm==4.67.1",
    "tensorboardX==2.6.2.2",
    "pynvml~=12.0.0",
    "GPUtil==1.4.0",
    "plotly==6.9.0",
    "matplotlib",
    "transformers==4.52.4",
    "accelerate==1.4.0",
]


class SetupError(Exception):
    pass


def _run(args, **kwargs) -> subprocess.CompletedProcess:
    try:
        # encoding="utf-8" (not just text=True) because git/docker/pip write
        # UTF-8 to stdout/stderr regardless of the OS locale — on Windows,
        # text=True alone decodes with the system codepage (cp949 on Korean
        # Windows), and a single non-cp949 byte crashes subprocess's internal
        # stdout-reader thread with UnicodeDecodeError.
        return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", **kwargs)
    except FileNotFoundError as exc:
        raise SetupError(f"'{args[0]}' 명령을 찾을 수 없습니다. 설치되어 있는지 확인하세요. ({exc})") from exc
    except subprocess.TimeoutExpired as exc:
        raise SetupError(f"'{args[0]}' 명령이 시간 초과되었습니다.") from exc


def container_state() -> str:
    """'running' | 'stopped' | 'missing'."""
    proc = _run(["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME])
    if proc.returncode != 0:
        return "missing"
    return "running" if proc.stdout.strip() == "true" else "stopped"


def _xmem_dir() -> Path:
    return WORK_DIR / "xMem"


def _xmem_ready_on_host() -> bool:
    """True once clone+patch+copy have all landed on the host side (the part
    of provisioning that a container restart can't undo)."""
    return (_xmem_dir() / "predict_vram.py").exists()


def _pip_ready_in_container() -> bool:
    """True once PIP_PACKAGES are actually installed in THIS container
    instance. A plain `docker exec ... test -f` is fast enough to run on
    every /setup call (no need to cache it), and unlike checking host-side
    files, it correctly comes back False after a `docker rm` + recreate even
    though /work (and predict_vram.py on it) still looks fully set up."""
    proc = _run(["docker", "exec", CONTAINER_NAME, "test", "-f", PIP_READY_MARKER])
    return proc.returncode == 0


def ensure_ready() -> dict:
    """Idempotent and safe to call on every page load / model pick. Neither
    a container existing nor /work looking populated is enough to call it
    "ready": `docker run` succeeding and a later step (clone/patch/copy/pip
    install) failing, or a container being recreated after `docker rm`, both
    leave stale-looking state around — every call must still notice that and
    finish the job, not just declare victory because `docker inspect` says
    Running=true or because a previous attempt's file copy already landed."""
    state = container_state()

    if state == "missing":
        _create_container()
        status = "provisioned"
    elif state == "stopped":
        proc = _run(["docker", "start", CONTAINER_NAME])
        if proc.returncode != 0:
            raise SetupError(f"컨테이너를 시작하지 못했습니다: {proc.stderr.strip()}")
        status = "started_existing"
    else:
        status = "already_running"

    if not _xmem_ready_on_host() or not _pip_ready_in_container():
        _finish_xmem_setup()
        status = "provisioned"
    else:
        # A ready container may still use a predictor copied before git pull.
        # Refresh only that file; keep the container, caches and dependencies.
        _sync_predictor()

    return {"status": status}


def _create_container() -> None:
    if not CLOUDGPUFIT_DIR.exists():
        raise SetupError(f"CloudGPUFit 저장소를 찾을 수 없습니다: {CLOUDGPUFIT_DIR}")

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    proc = _run(["docker", "pull", IMAGE], timeout=PULL_TIMEOUT_SECONDS)
    if proc.returncode != 0:
        raise SetupError(f"이미지 pull 실패: {proc.stderr.strip()}")

    # Mount the host's HF cache into the container so re-provisioning (e.g.
    # after `docker rm`) doesn't re-download every model from scratch.
    proc = _run(
        [
            "docker", "run", "-d", "--name", CONTAINER_NAME,
            "-v", f"{WORK_DIR}:/work",
            "-v", f"{HF_CACHE_DIR}:/root/.cache/huggingface",
            IMAGE, "sleep", "infinity",
        ]
    )
    if proc.returncode != 0:
        raise SetupError(f"컨테이너 생성 실패: {proc.stderr.strip()}")


def _sync_predictor() -> None:
    """Atomically refresh the predictor in the bind-mounted xMem checkout."""
    source = CLOUDGPUFIT_DIR / "predict_vram.py"
    target = _xmem_dir() / "predict_vram.py"
    if not source.is_file():
        raise SetupError(f"예측 코드를 찾을 수 없습니다: {source}")
    content = source.read_bytes()
    if target.is_file() and target.read_bytes() == content:
        return
    # Place the temporary file beside the target so replace stays atomic.
    fd, temporary = tempfile.mkstemp(prefix=".predict-vram-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(content)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _finish_xmem_setup() -> None:
    if not CLOUDGPUFIT_DIR.exists():
        raise SetupError(f"CloudGPUFit 저장소를 찾을 수 없습니다: {CLOUDGPUFIT_DIR}")

    xmem_dir = _xmem_dir()
    if not (xmem_dir / ".git").exists():
        # A previous attempt may have left an empty/partial directory behind
        # (e.g. a clone that timed out) — git clone refuses to write into a
        # non-empty target, so clear it first.
        if xmem_dir.exists():
            shutil.rmtree(xmem_dir)
        proc = _run(["git", "clone", XMEM_REPO_URL, str(xmem_dir)], timeout=600)
        if proc.returncode != 0:
            raise SetupError(f"xMem clone 실패: {proc.stderr.strip()}")
        proc = _run(["git", "checkout", XMEM_COMMIT], cwd=xmem_dir)
        if proc.returncode != 0:
            raise SetupError(f"xMem 커밋 체크아웃 실패: {proc.stderr.strip()}")

    _apply_patch_if_needed(xmem_dir)

    _sync_predictor()

    pip_cmd = "pip install --no-cache-dir " + " ".join(PIP_PACKAGES)
    proc = _run(["docker", "exec", "-t", CONTAINER_NAME, "bash", "-c", pip_cmd], timeout=PIP_INSTALL_TIMEOUT_SECONDS)
    if proc.returncode != 0:
        raise SetupError(f"패키지 설치 실패: {proc.stderr.strip()}")

    # Marker written LAST, only once every prior step in this function has
    # actually succeeded — _pip_ready_in_container() keys off it, so a
    # failure anywhere above (this function raises and stops) leaves it
    # absent and the next /setup call retries from here instead of assuming
    # the container is ready.
    _run(["docker", "exec", CONTAINER_NAME, "touch", PIP_READY_MARKER])


def _apply_patch_if_needed(xmem_dir: Path) -> None:
    """Applies core_inference.patch, distinguishing "clean apply" from
    "already applied" (harmless — happens on every retry after the first
    successful setup, since patch+copy always re-run alongside pip install)
    from a genuine conflict (the xMem checkout no longer matches either the
    pre- or post-patch state, e.g. XMEM_COMMIT changed upstream) — the old
    code treated the latter two identically as "not fatal", silently leaving
    a conflicted checkout in place instead of surfacing it."""
    patch_path = CLOUDGPUFIT_DIR / "xMem" / "patches" / "core_inference.patch"

    forward_check = _run(["git", "apply", "--check", str(patch_path)], cwd=xmem_dir)
    if forward_check.returncode == 0:
        applied = _run(["git", "apply", str(patch_path)], cwd=xmem_dir)
        if applied.returncode != 0:
            raise SetupError(f"패치 적용 실패: {applied.stderr.strip()}")
        return

    reverse_check = _run(["git", "apply", "--check", "--reverse", str(patch_path)], cwd=xmem_dir)
    if reverse_check.returncode == 0:
        return  # already applied — expected on every retry past the first

    raise SetupError(
        "xMem 패치가 충돌 상태입니다 (이미 적용된 상태도, 새로 적용 가능한 상태도 아님). "
        f"{xmem_dir} 의 xMem 체크아웃을 확인하세요.\n{forward_check.stderr.strip()}"
    )
