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
    부모 폴더, 즉 `git clone`했을 때의 저장소 루트). `orchestrator/`를 저장소
    밖으로 옮겨 별도 실행하는 경우에는 이 환경변수를 직접 지정해야 한다.
    컨테이너에 마운트되는 작업 디렉터리는
    `XMEM_WORK_DIR` (기본값: `~/xmem-work`, `setup.sh`와 동일).
- 이 서버 자체는 `docker`·`git` CLI를 호출할 수 있는 호스트에서 실행해야 한다.

## 실행

```bash
cd orchestrator
python -m venv .venv
.venv/Scripts/activate      # Windows. macOS/Linux는 source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-runtime.txt   # eager_runtime 의존성 (torch 등, 용량 큼)
uvicorn main:app --port 8000
```

`web/.env.local`에 `ORCHESTRATOR_URL=http://localhost:8000` (기본값과 같으면 생략 가능).

## 예측 파이프라인 (`POST /jobs` → `process_job()`)

한 job(`main.py:process_job`)이 순서대로 하는 일:

1. **VRAM 예측** — `xmem_runner.run_xmem()`이 `predict_vram.py`를 `xmem-cpu`
   컨테이너 안에서 실행해 `required_vram_gb`를 얻는다.
2. **인스턴스 후보 필터링** — `instance_catalog.load_instance_catalog()`가
   `aws_pricing.py`로 해당 리전의 AWS 가격을 실시간 조회(Bulk Price List API,
   자격증명 불필요, 디스크에 리전별 TTL 캐시)한 뒤, `instance_filter.py`가 VRAM·
   리전·예산 조건과 **eager_runtime에 자기 target이 있는 인스턴스** 조건으로 후보를
   추린다. 현재 7개: g4dn.xlarge, g5.xlarge, g6.xlarge, g6e.xlarge, g7.2xlarge,
   g7e.2xlarge, gr6.4xlarge. 측정하지 않은 크기(2xlarge~16xlarge 등)는 다른
   인스턴스 결과를 빌리거나 vCPU 배율로 추정해 보여주지 않는다 — 이들은
   `SUPPORTED_GPU_INSTANCES`(26개: 기존 목록에서 g5g 제외, g7.2xlarge 추가)에 남아 있고, 자기
   target(`eager_runtime/targets/<instance>/`, 실측 또는 `size_profiles/` +
   `build_size_target.py`로 생성)이 추가되는 순간 코드 수정 없이 후보가 된다.
   가격은 `aws_pricing.py`가 이 26개에 대해 조회한다. 목록에 인스턴스를 새로 넣었다면
   `.aws_pricing_cache/`를 지워야 TTL(24시간)을 기다리지 않고 바로 반영된다.
3. **Runtime(지연시간/처리량) 예측** — `runtime_adapter.py`가 후보들에 대해
   `eager_runtime.predict_runtime_targets()`를 job의 `engine_mode`(`"eager"` 기본,
   `"compiled"`)로 호출해 median/P90 지연시간과 (나이브 추정) 처리량을 예측한다.
   지원 워크로드: decoder_llm, text_encdec, vision_cls, audio_encoder(Wav2Vec2 계열).
   audio_encoder는 VRAM 예측(`predict_vram.py`)과 웹 폼 지원이 함께 필요하다.
   `eager_runtime/`의 구조·CLI는
   [`eager_runtime/README_TARGETS.md`](eager_runtime/README_TARGETS.md) 참고.
   `eager_runtime`은 transformers==4.45.1 환경에서 보정됐다(버전이 다르면 예측값이 달라짐).

이 결과(`vram`/`candidates`/`runtime`)를 그대로 `GET /jobs/{id}`가 반환한다. SLO
충족 판정과 중요도 기반 순위 매기기는 이 서버가 아니라 `web/lib/slo.ts`·
`web/lib/ranking.ts`(순수 함수, 브라우저에서 계산)에서 이루어진다 — `PATCH
/jobs/{id}/slo`는 사용자가 입력한 허용 기준을 저장만 할 뿐, 재계산이나 캐시 키에
영향을 주지 않는다.

## 개발용 테스트 (`dev/`, 서비스 코드는 import하지 않음)

`orchestrator/`에서 실행한다.

| 명령 | 내용 |
|---|---|
| `python dev/logic_test.py` | 후보 필터·어댑터·파라미터·캐시 키 로직 (가짜 eager_runtime, 1초) |
| `python dev/api_smoke.py --quick` | 실제 FastAPI 앱으로 `POST /jobs`→`GET /jobs/{id}` 스모크 (xMem·가격·DB만 테스트용으로 대체) |
| `python dev/api_smoke.py` | 위를 4개 워크로드 × Eager/Compiled로 |
| `python dev/predict_from_file.py dev/requests/opt_decoder.json` | 요청 JSON 하나로 예측해 인스턴스별 표 출력 |
| `python dev/send_jobs.py` | **실행 중인 서버**(`uvicorn main:app --port 8000`, xMem 컨테이너 필요)에 `dev/jobs/*.json`을 실제로 보내 VRAM·후보·예측 요약 출력. `E*` 파일은 400 거절이 정상 |
| `python eager_runtime/regression/run_regression.py` | eager_runtime 회귀 테스트 (`--full`은 공식 E2E 재현, 1시간 안팎) |

테스트의 VRAM 값과 가격 없는 카탈로그는 로직 확인용 입력이며 실제 데이터가 아니다.

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
