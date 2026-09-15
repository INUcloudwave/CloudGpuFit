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
from pathlib import Path

IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"
CONTAINER_NAME = "xmem-cpu"
XMEM_REPO_URL = "https://github.com/Stone-ResearchLife/xMem"
XMEM_COMMIT = "1ee4bd08dda277a6248abee5b6d40e0d6067efaa"

# Where the CloudGPUFit repo (predict_vram.py + the patch file) lives.
# Defaults to a sibling of this orchestrator/ directory.
CLOUDGPUFIT_DIR = Path(os.environ.get("CLOUDGPUFIT_DIR", Path(__file__).parent.parent / "CloudGPUFit")).resolve()

# Host directory bind-mounted into the container at /work — same default
# setup.sh uses (~/xmem-work), so a container created by either path is
# interchangeable.
WORK_DIR = Path(os.environ.get("XMEM_WORK_DIR", Path.home() / "xmem-work")).resolve()

# Mounted into the container so models already downloaded survive a container
# recreate (`docker rm xmem-cpu` + re-provision) instead of re-downloading.
HF_CACHE_DIR = Path(os.environ.get("HF_CACHE_DIR", Path.home() / ".cache" / "huggingface")).resolve()

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
    of provisioning that a container restart can't undo). Doesn't check that
    pip install succeeded inside the container — a partial pip install would
    surface later as a job failure (model_load_failed/internal), not here."""
    return (_xmem_dir() / "predict_vram.py").exists()


def ensure_ready() -> dict:
    """Idempotent and safe to call on every page load / model pick. A
    container existing is NOT enough to call it "ready": `docker run`
    succeeding and a later step (clone/patch/pip install) failing left a
    container that's running but has an empty /work — every later call must
    still notice that and finish the job, not just declare victory because
    `docker inspect` says Running=true."""
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

    if not _xmem_ready_on_host():
        _finish_xmem_setup()
        status = "provisioned"

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

    patch_path = CLOUDGPUFIT_DIR / "xMem" / "patches" / "core_inference.patch"
    check = _run(["git", "apply", "--check", str(patch_path)], cwd=xmem_dir)
    if check.returncode == 0:
        applied = _run(["git", "apply", str(patch_path)], cwd=xmem_dir)
        if applied.returncode != 0:
            raise SetupError(f"패치 적용 실패: {applied.stderr.strip()}")
    # else: already applied, or a conflict setup.sh itself would also just
    # warn about and leave for manual inspection — not fatal either way.

    shutil.copyfile(CLOUDGPUFIT_DIR / "predict_vram.py", xmem_dir / "predict_vram.py")

    pip_cmd = "pip install --no-cache-dir " + " ".join(PIP_PACKAGES)
    proc = _run(["docker", "exec", "-t", CONTAINER_NAME, "bash", "-c", pip_cmd], timeout=PIP_INSTALL_TIMEOUT_SECONDS)
    if proc.returncode != 0:
        raise SetupError(f"패키지 설치 실패: {proc.stderr.strip()}")
