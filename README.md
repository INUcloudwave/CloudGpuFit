# xMem 기반 GPU VRAM 사전 예측 도구

CPU만으로 추론 워크로드의 GPU VRAM 필요량을 예측합니다. 실제 GPU 없이도
`required_vram_gb` 값을 얻어 AWS GPU 인스턴스 후보를 1차 필터링하는 데 씁니다.

> **`localPrototype` 브랜치 안내**: 이 브랜치에는 이 도구를 실제로 감싸는
> 로컬 프로토타입(웹 UI + 오케스트레이터)이 추가되어 있습니다. 아직 `main`에는
> 없는, 검토용 코드입니다.
>
> **전체 흐름**: 사용자가 웹 마법사에서 HF 모델을 고르면(`web/`), 오케스트레이터
> (`orchestrator/`)가 (1) `predict_vram.py`로 CPU 프로파일링해 필요 VRAM을 구하고,
> (2) AWS 가격 카탈로그를 실시간 조회해 그 VRAM·리전·예산에 맞는 GPU 인스턴스
> 후보를 추리고, (3) 후보마다 `eager_runtime`으로 추론 지연시간(median/P90)과
> 처리량을 예측한다. 사용자가 입력한 SLO(허용 지연시간·최소 처리량)를 만족하는지
> 판정하고, 가격·지연시간·처리량에 대한 사용자 지정 중요도로 후보를 순위 매겨
> 보여주는 것까지가 한 번의 예측 요청(job)이다.
>
> - [`web/`](web/README.md) — 사용자 입력 마법사 (Next.js). 모델 검색 → 유형
>   자동 판별 → 파라미터 입력(직접 입력 또는 시나리오 선택) → 리전/예산 입력 →
>   SLO 입력 → 결과(VRAM, 인스턴스 후보별 지연시간/처리량, SLO 판정, 중요도
>   기반 추천 순위) 확인.
> - [`orchestrator/`](orchestrator/README.md) — `web/`과 이 저장소의
>   `predict_vram.py`/`eager_runtime` 사이를 잇는 FastAPI 서버. job 큐·캐싱,
>   컨테이너 자동 프로비저닝(`bash setup.sh`를 사용자가 직접 안 돌려도 되게),
>   AWS 가격 실시간 조회.
>
> 아래 1~5번 섹션은 이 도구(`predict_vram.py`)를 CLI로 직접 쓸 때의 설명이고,
> `web`/`orchestrator`를 쓸 때는 각 폴더의 README를 먼저 보세요.

## 1. 설치

```bash
git clone <이 저장소 URL>
cd <repo>
bash setup.sh
```

`setup.sh`가 하는 일:
- `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` 이미지 pull
- 컨테이너 실행 및 xMem 코어 패치 3개 파일 적용
- 필요 패키지 설치 확인

## 2. 실행

```bash
docker exec -t xmem-cpu bash -c "cd /work/xMem && python predict_vram.py \
  --model <모델명> --batch_size <N> --seq_len <N> --dtype <fp32|fp16|bf16>"
```

### 입력값

| 인자 | 설명 | 기본값 | 비고 |
|---|---|---|---|
| `--model` | HF 모델 ID 또는 torchvision 모델 키 | 필수 | 아래 표 참고 |
| `--batch_size` | 배치 크기 | 1 | |
| `--seq_len` | 시퀀스 길이 (LLM만) | 512 | |
| `--encoder_seq_len` | 인코더 시퀀스 길이 (인코더-디코더만) | 512 | T5/BART 등 |
| `--decoder_seq_len` | 디코더 시퀀스 길이 (인코더-디코더만) | 128 | T5/BART 등 |
| `--resolution` | 이미지 해상도 (Vision만) | 모델 config의 image_size | 없으면 224 |
| `--workload_type` | `decoder_llm`/`text_encdec`/`vision_cls` 강제 지정 | 자동 판별 | pipeline_tag로 판별 실패 시에만 필요 |
| `--dtype` | 목표 dtype | fp16 | 실제 추론 시 사용할 dtype 입력 |
| `--mode` | prefill / generate | prefill | **generate는 실험적, 아래 참고** |
| `--max_new_tokens` | generate 시 생성 토큰 수 | 256 | |

Vision 모델명은 대소문자까지 정확히 일치해야 합니다:
`VGG16`, `VGG19`, `ResNet101`, `ResNet152`, `MobileNetV2`, `MobeNetV3Small`,
`MobeNetV3Large`, `MnasNet`, `ConvNeXtTiny`, `ConvNeXtBase`, `RegNetX400MF`, `RegNetY400MF`

LLM은 Hugging Face 모델 ID를 그대로 쓰면 됩니다 (예: `facebook/opt-125m`, `Qwen/Qwen3-0.6B`).

### 출력

```json
{
  "status": "ok",
  "required_vram_gb": 2.86,
  "warnings": []
}
```

`required_vram_gb`를 인스턴스 선정 기준값으로 사용하세요.
`warnings`에는 결과 해석에 영향을 주는 사항이 담깁니다.

## 3. 알려진 제약

- CPU 프로파일링은 항상 FP32로 수행 후 dtype 계수(0.5)로 환산합니다.
- 결과에는 allocator gap + CUDA context를 흡수하는 고정 마진 2GB가 포함됩니다.
- 7B급 이상 대형 모델은 CPU 프로파일링 중 메모리 부족(OOM)으로 실패할 수 있습니다.
- `generate` 모드는 KV cache를 추적하지 않고 수식으로 추정하며, 과대추정 경향이 있습니다.
- 인코더-디코더(T5/BART 등)는 `--encoder_seq_len`/`--decoder_seq_len`으로 지원합니다. 오디오(Whisper 등) 모델은 아직 지원하지 않습니다.
- 이미지 분류는 torchvision 모델 키와 Hugging Face 이미지 분류 모델 ID를 모두 받습니다.

## 4. 문제 발생 시

- 결과가 안 나오고 `Killed`가 뜨면 모델이 너무 커서 CPU RAM 부족(OOM)입니다. 더 작은 모델로 시도하거나 담당자에게 문의하세요.
- 기타 에러는 `{"status": "error", "error": "..."}` 형태로 반환됩니다. 에러 메시지와 함께 담당자에게 전달해주세요.
