# CloudGPUFit — 웹 마법사

HF 모델을 고르면 GPU VRAM 요구량, AWS GPU 인스턴스 후보, 후보별 추론 지연시간/처리량
예측, SLO 충족 여부, 가격·지연시간·처리량 중요도에 따른 추천 순위까지 보여주는
Next.js(App Router) 프론트엔드. 실제 예측/캐싱/AWS 조회는 모두
[`orchestrator/`](../orchestrator/README.md)(FastAPI)가 수행하며, 브라우저는 이 앱의
Route Handler(`app/api/*`)를 거쳐서만 오케스트레이터를 호출한다 — 브라우저가
오케스트레이터에 직접 접근하는 경로는 없다.

전체 파이프라인(웹 ↔ 오케스트레이터 ↔ xMem/eager_runtime ↔ AWS)에 대한 설명은
[루트 README](../README.md)를 참고.

## 실행

```bash
npm install
cp .env.example .env.local   # 필요 시 값 채우기 (아래 "환경변수" 참고)
npm run dev
```

[http://localhost:3000](http://localhost:3000) 접속. 오케스트레이터가
`http://localhost:8000`에서 먼저 떠 있어야 한다(`orchestrator/README.md`의 실행법 참고) —
페이지 최초 진입 시 자동으로 `POST /setup`을 호출해 xMem CPU 프로파일링 컨테이너를
준비시킨다(최초 1회만 수 분 소요).

### 환경변수 (`.env.local`)

| 변수 | 필수 | 설명 |
|---|---|---|
| `ORCHESTRATOR_URL` | 아니오 (기본값 `http://localhost:8000`) | FastAPI 오케스트레이터 주소. Route Handler만 이 값을 읽는다 — 브라우저 번들에는 절대 노출되지 않는다. |
| `HF_TOKEN` | 아니오 | Hugging Face API 호출(검색/모델 유형 판별)에 서버 쪽에서 쓰는 토큰. 없어도 동작하지만, 없으면 비인증 rate limit을 받고 게이트(gated)된 모델은 볼 수 없다. |

게이트된 모델(예: Llama 계열)은 사용자가 화면에서 자신의 HF 토큰을 직접 입력해 접근 권한을
확인할 수 있다 — 이 토큰은 React 상태로만 존재하며 요청 헤더(`x-hf-token`)로만 전달되고,
job 페이로드·DB·로그 어디에도 저장되지 않는다.

## 마법사 흐름

1. **모델 검색** (`ModelSearchStep`) — HF 모델 검색 → 선택 시 `pipeline_tag`로 유형을
   자동 판별(`decoder_llm` / `vision_cls` / `text_encdec`). 그 외 유형은 아직 예측을
   지원하지 않는다.
2. **파라미터 입력** — 두 가지 방식 중 선택:
   - **직접 입력** (`ParamsStep`) — dtype·batch_size와 유형별 파라미터(`seq_len`,
     `resolution`, `encoder_seq_len`/`decoder_seq_len` 등)를 직접 지정.
   - **시나리오 선택** (`ScenarioStep`) — "챗봇", "문서 요약" 같은 미리 정의된 워크로드
     프리셋 중 하나를 골라 파라미터를 자동으로 채움.
   - 두 방식 모두 리전(`aws_region`)과 예산(`budget_usd_per_hour`)을 함께 입력.
3. **제출 → SLO 입력** (`SloGate` → `SloStep`) — job이 백그라운드에서 예측을 시작한
   *직후* 화면. `max_latency_ms`(허용 최대 P90 지연시간)와 `min_throughput_rps`(최소
   허용 처리량)를 입력한다. 이 값은 결과를 평가하는 기준일 뿐 워크로드 입력이
   아니라서, 예측 계산 자체나 캐시 키에는 영향을 주지 않는다.
4. **결과** (`JobFlow`):
   - VRAM 예측치와 세부 breakdown.
   - 조건(VRAM/리전/예산)을 만족하는 AWS 인스턴스 후보별 median/P90 지연시간과
     처리량(나이브 추정치 — 자세한 가정은 코드 주석 참고) 표.
   - **추천 순위** — SLO 충족 여부(충족 > 미판정 > 미충족)를 1차 기준으로, 같은
     그룹 안에서는 가격/지연시간/처리량 중요도 슬라이더로 가중합한 점수 순으로
     정렬한 순위표.

## 지원 워크로드 유형

| 유형 | HF `pipeline_tag` | 필수 파라미터 |
|---|---|---|
| `decoder_llm` | `text-generation` | `seq_len` |
| `vision_cls` | `image-classification` | `resolution` |
| `text_encdec` | `text2text-generation`, `translation`, `summarization` | `encoder_seq_len`, `decoder_seq_len` |

그 외 `pipeline_tag`(오디오, fill-mask 등)는 `unsupported`로 분류되어 예측을 진행할 수
없다.

## 주요 코드 위치

- `components/Wizard.tsx` — 전체 마법사 상태 머신(위 4단계 전환).
- `lib/workloadSchema.ts` — 유형별 입력 폼 스키마, 시나리오 프리셋, payload 빌더.
- `lib/slo.ts` — 순수 SLO 판정 함수(`evaluateSlo`).
- `lib/ranking.ts` — 순수 후보 순위 함수(`rankCandidates`) — SLO 판정과 사용자 중요도를
  합쳐 정렬.
- `app/api/*` — 오케스트레이터로의 유일한 통로(Route Handlers).
