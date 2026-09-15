# CloudGPUFit Orchestrator

`predict_vram.py` 실행을 큐잉·캐싱하는 FastAPI 서버. `web/`(Next.js)가 이 서버로
요청을 중계한다 — 브라우저는 이 서버에 직접 접근하지 않는다.

## 전제 조건

- Docker가 설치되어 있어야 한다. **Windows에 Docker가 없다면:**
  1. 관리자 권한 PowerShell에서 `scripts/install-docker-windows.ps1` 실행.
     - WSL2가 없으면 먼저 설치하고 재부팅을 요구한다 — 재부팅 후 스크립트를 다시
       실행하면 이어서 Docker Desktop을 설치한다.
     - 또는 수동 설치: [Docker Desktop 다운로드](https://www.docker.com/products/docker-desktop/)
       (WSL2 backend 선택).
  2. Docker Desktop을 한 번 실행해 최초 설정(라이선스 동의)을 마친다.
  3. 새 터미널에서 `docker run hello-world`로 정상 동작 확인.
- `xmem-cpu`라는 이름의 Docker 컨테이너가 떠 있어야 `predict_vram.py`를 실행할 수
  있다. **수동으로 준비할 필요는 없다** — UI에서 모델을 고르고 "다음"을 누르는
  순간 `POST /setup`이 자동 호출되어 없으면 만들고, 있으면 그냥 통과한다
  (`setup_manager.py`, `CloudGPUFit/setup.sh`와 같은 절차를 Python으로 재구현).
  최초 1회만 이미지 pull+clone+설치로 수 분 걸리고, 이후 호출은 즉시 반환된다.
  - `bash setup.sh`를 직접 돌려도 되지만 필수는 아니다. Windows에서 `bash`가
    WSL 스텁으로 잡혀 있으면(distro 없이 WSL만 설치된 경우) 그 경로 자체가
    깨질 수 있어 자동 경로를 기본으로 둔 것.
  - 저장소 위치는 `CLOUDGPUFIT_DIR` 환경변수로 지정 (기본값: 이 디렉터리의
    형제 폴더 `../CloudGPUFit`). 컨테이너에 마운트되는 작업 디렉터리는
    `XMEM_WORK_DIR` (기본값: `~/xmem-work`, `setup.sh`와 동일).
- 이 서버 자체는 `docker`·`git` CLI를 호출할 수 있는 호스트에서 실행해야 한다.

## 실행

```bash
cd orchestrator
python -m venv .venv
.venv/Scripts/activate      # Windows. macOS/Linux는 source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --port 8000
```

`web/.env.local`에 `ORCHESTRATOR_URL=http://localhost:8000` (기본값과 같으면 생략 가능).

## 설계 메모

- **동시 실행 = 1 고정.** `asyncio.Queue` + 워커 태스크 1개. xMem이 FP32 모델을
  CPU RAM에 올려 프로파일링하므로 병렬 실행 시 서로 OOM을 유발한다.
- **캐시 키는 xMem에 실제로 넘어가는 인자만으로 계산한다** (`xmem_runner.build_xmem_args`
  로 화이트리스트한 뒤 sha256). `deployment`(리전·예산)는 제외 — 바뀌어도 VRAM
  예측값은 동일하므로 재실행하지 않는다.
- **성공/실패는 항상 stdout의 JSON `status` 필드로 판정한다.** `predict_vram.py`는
  자체 에러도 JSON으로 찍고 `sys.exit(1)`하므로 종료 코드만 보면 오탐한다.
- 워커 루프(`worker_loop`)는 개별 job 처리 중 발생하는 어떤 예외든 잡아 해당
  job만 `failed/internal`로 마감한다 — 안 그러면 (예: `docker` 바이너리가 없어
  `FileNotFoundError`가 나는 경우) 예외 하나가 워커 태스크 자체를 죽여 이후 모든
  job이 영원히 `queued`로 멈춘다.
- `jobs.db`(SQLite, gitignore 대상)는 이 디렉터리에 생성된다. 지우면 캐시와
  이력이 모두 초기화된다.
- `POST /setup`은 `asyncio.Lock`으로 동시 호출을 막는다 — 잠겨 있는 동안 온
  요청은 `409 {"status": "in_progress"}`를 받는다. `docker`/`git` 자체가 없어
  터지는 경우까지 포함해 실패는 전부 `SetupError`로 모아 `{"status":"error"}`
  JSON으로 응답한다(예외가 그대로 새서 500 스택트레이스가 나가지 않도록).

## 알려진 후속 작업 (이번 범위 밖)

- 실제 프로비저닝 경로인 `setup_manager.py`는 호스트의 `~/.cache/huggingface`를
  컨테이너의 `/root/.cache/huggingface`에 마운트해서 컨테이너를 재생성해도
  모델을 다시 받지 않는다(`HF_CACHE_DIR` 환경변수로 위치 변경 가능). 다만
  CloudGPUFit 저장소의 `setup.sh`(bash 버전)에는 이 마운트가 아직 없다 — 그건
  별도 GitHub 저장소라 여기서 직접 수정·푸시하지 않았다. `bash setup.sh`를
  직접 쓰는 경우에만 해당 저장소 쪽에 같은 diff를 적용할 것.
