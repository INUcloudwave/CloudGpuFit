# xMem 기반 GPU VRAM 사전 예측 도구

CPU만으로 추론 워크로드의 GPU VRAM 필요량을 예측합니다. 실제 GPU 없이도
`required_vram_gb` 값을 얻어 AWS GPU 인스턴스 후보를 1차 필터링하는 데 씁니다.

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
| `--batch_size` | 배치 크기 | 1 | 검증 범위: LLM 1~30, Vision 32~64 |
| `--seq_len` | 시퀀스 길이 (LLM만) | 512 | 검증 범위: 128~2048 |
| `--resolution` | 이미지 해상도 (Vision만) | 224 | 검증됨: 86, 224 |
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
`warnings`가 비어있지 않으면 검증 범위 밖 입력이니 결과를 참고용으로만 쓰세요.

## 3. 검증 현황

| 워크로드 | 오차 범위 | 비고 |
|---|---|---|
| LLM prefill (tensor) | ±2.4% 이내 | seq_len 128~2048, batch 1~30 |
| CNN prefill | ±5% 이내 | 해상도 86, 224 모두 확인 |
| generate (KV cache 포함) | **미검증, 과대추정 확인됨** | 사용 시 여유값으로만 참고 |

## 4. 알려진 제약

- CPU 프로파일링은 항상 FP32로 수행 후 dtype 계수(0.5)로 환산합니다.
- 결과에는 allocator gap + CUDA context를 흡수하는 고정 마진 2GB가 포함됩니다.
- 7B급 이상 대형 모델은 CPU 프로파일링 중 메모리 부족(OOM)으로 실패할 수 있습니다.
- `generate` 모드는 KV cache를 수식으로 추정하며 실측 대비 과대추정 경향이 있습니다(약 +5% 수준, 1개 샘플 기준).
- 인코더-디코더(T5) / 오디오(Whisper) 모델은 아직 지원하지 않습니다.

## 5. 문제 발생 시

- 결과가 안 나오고 `Killed`가 뜨면 모델이 너무 커서 CPU RAM 부족(OOM)입니다. 더 작은 모델로 시도하거나 담당자에게 문의하세요.
- 기타 에러는 `{"status": "error", "error": "..."}` 형태로 반환됩니다. 에러 메시지와 함께 담당자에게 전달해주세요.
