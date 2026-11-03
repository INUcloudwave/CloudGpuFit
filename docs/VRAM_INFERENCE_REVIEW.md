# 추론 VRAM 예측 경로 검토 및 v3 실행 방법

검토 기준: `localPrototype`의 `2d6198bf53788af1a42c8b462e1169f7e3606120`.
응답시간·처리량 예측기는 변경하지 않았다. 현재 수정은 CPU 예측 및 메모리 후보 필터 경로다.

## 1. outside 가정에서 확인한 범위

`outside = 같은 시점의 NVML process bytes − PyTorch reserved bytes`다.
`NVML 피크 − reserved 피크`를 서로 다른 시점에서 빼지 않았다.
또한 `reserved − allocated`는 allocator 내부 캐시이므로 outside와 구분한다.

GPU마다 FP16·조건01의 9모델만 보정값 산정에 사용하였다. 조건별 샘플의 중앙값을 구하고
9개의 중앙값에서 다시 중앙값을 구했다. 다음 값은 하드웨어 명세에서 유도한 상수가 아니라
실측한 소프트웨어/드라이버 환경에 한정된 경험값이다.

| GPU | 기준9조건에서 정한 outside | 기준9조건의 최대 관측값 | 나머지 조건·실행에서 관측한 전체 범위 |
|---|---:|---:|---:|
| T4 | 132 MiB | 132 MiB | 128–132 MiB |
| A10G | 308 MiB | 312 MiB | 302–312 MiB |
| L4 | 232 MiB | 236 MiB | 226–236 MiB |
| RTX 3090 | 312 MiB | 316 MiB | 306–316 MiB |

기준값 산정에 쓰지 않은 1,902개 조건·실행 조합에서 조건별 중앙값의 최대 절대오차는
T4 4MiB, 나머지 6MiB였다. L4는 별도 4·16vCPU 실행까지 포함한다.
이는 반복 실행에 대한 통계적 신뢰구간은 아니다. 드라이버·PyTorch·CUDA·라이브러리 로딩 방식·
다중 프로세스·컴파일·다른 커널에서는 다시 확인해야 한다.

근거 파일은 `gpu_overheads.json`(기준조건·런타임 범위·원본SHA256)과
`docs/vram-overhead-validation.json`이다. 재현 도구는 `tools/calibrate_gpu_overheads.py`다.
L40S는 원본 결과를 아직 받지 않아 상수를 만들지 않았다. 미보정 GPU에 0이나 다른 GPU 값을
몰래 대입하지 않는다. 결과를 받으면 기준9조건으로 별도 보정한다.

## 2. 기존 추론 코드의 문제

1. 모델을 FP32로 CPU에서 forward 프로파일링하고 결과 전체를 FP16/BF16에서 절반으로
   환산했다. 가중치 외 정수 입력·마스크·bool 버퍼, FP32 중간 연산까지 모두 절반이라는
   가정이므로 메모리 성분별 정확성을 보장할 수 없었다.
2. 과소예측 보정으로 FP32 32MiB, FP16/BF16 16MiB를 더했다. 이 상수는 제거했다.
3. CPU 경로는 `use_cache=False` prefill이었고 generate는 KV 수식을 나중에 더했다.
   prefill과 decode의 피크를 단순 합산하는 문제, T5 Cross KV 누락 가능성,
   마지막 토큰을 반환하는 데 필요하지 않은 추가 forward의 cache 길이 가정 문제가 있었다.
4. xMem의 원래 학습 추정기는 gradient/optimizer 처리와 반복 구간 교체 같은 학습 가정을
   갖고 있다. 추론에 그대로 적용하는 대신 추론 영역의 텐서 생명주기만 allocator에 입력한다.
5. 원래 xMem이 `memory.segment`도 계산했는데 래퍼는 이를 최종 요구량에 사용하지 않았다.
   allocated에 outside만 더해서 전체 프로세스 사용량으로 제시하면 allocator 캐시가 빠진다.

## 3. 수정한 계산 경로

`pinned config → native dtype CPU model → eval + inference_mode → forward/full generate
→ 추론 영역의 tensor allocation/free → xMem allocator simulation → GPU별 outside`.

- `predict_vram.py`: 목표 FP32/FP16/BF16 그대로 모델·Vision 입력을 만들며 정밀도 fallback이 없다.
  `attention=eager`, seed=20261006, 기본 CPU4스레드다. CPU 연산이 해당 dtype을 지원하지 않으면
  실패를 기록한다. FP32 결과로 바꿔 실행하고 같은 결과인 것처럼 처리하지 않는다.
- `cpu_tensor_trace.py`: CPU RSS가 아닌 torch tensor allocation/free 사건을 읽는다.
  데이터로더·모델 생성은 측정 영역 밖이다. 모델/입력의 고유 storage를 별도 상주 메모리로
  재구성하고, 준비된 입력에 대해 warmup1회 후 본 측정3회다.
- 생성은 실측과 같은 greedy, beam1, use_cache=True, min_new_tokens=max_new_tokens다.
  Self/Cross KV를 포함한 전체 generate의 실제 할당을 추적하며 KV 수식을 추가하지 않는다.
  KV logical bytes와 underlying storage bytes를 별도 기록한다. 예를 들어 GPT-2의 초기
  cache가 공유 QKV storage의 view일 때 두 값은 다를 수 있다.
- xMem upstream commit은 `1ee4bd08...`를 유지하고 allocator를 직접 사용한다.
  기존 학습 코어 패치는 이 경로에 필요하지 않으며 기존 checkout의 패치를 되돌리지 않는다.
- `vram_policy.py`: `예상 NVML = 예상 reserved + outside 중앙값`.
  후보 메모리 충족 판정에는 `예상 reserved + 기준9조건의 outside 최대 관측값`을 사용한다.
  이 최대값은 임의 안전마진이나 향후 워크로드의 보장된 상한이 아니다.
- GPU를 지정하지 않은 최상위 `required_vram_*`는 보정된 GPU들 중 최대 요구량이다.
  `required_vram_scope`로 이 범위를 표시하고, 후보 필터는 GPU별 `gpu_requirements`를 사용한다.
  T4의 BF16 및 미보정 GPU는 해당 프로세스 메모리 기준 후보에서 제외한다.
- 캐시는 예측 코드·보정표 변경 때 갱신된다. 응답시간/처리량 모델 파일은 변경하지 않았다.

GPU별 커널 workspace, 실제 CUDA allocator 초기 상태, 모델 로딩/이전 모드에서 남은 캐시는
CPU trace와 시뮬레이션만으로 완전히 재현되지 않는다. 따라서 outside가 안정적이라는 사실이
**전체 VRAM 오차가 이미 작다**는 뜻은 아니다. 현재 v3의 정확도는 아래 342조건 CPU 실행 후
각 메모리 지표를 따로 비교해 확인해야 한다. 예측값을 보지 않고 32MiB 같은 보정을 다시
만들지 않았다. 추가 보정이 필요하면 train/validation 분리를 유지한다.

## 4. inuos에서 실행

기존 `xmem-cpu` 컨테이너와 설치 환경을 재사용한다. 먼저 저장소 폴더에서 다음을 실행한다.
현재 전달 방식은 검토용 패치이며 GitHub 브랜치/PR은 생성되지 않았다.
기존 코드가 검토 기준 커밋과 다르거나 수정되어 있으면 적용 검사가 중단되도록 한다.

```bash
git apply --check /path/to/cloudgpufit-vram-native-v3.patch
git apply /path/to/cloudgpufit-vram-native-v3.patch
python3 tools/run_prediction_suite.py \
  --output-root ~/cloudgpufit-predictions/native-v3 \
  --case-set baseline9 --threads 4 --prepare --preflight-only
```

추후 저장소 반영을 승인받아 별도 브랜치가 제공되면 위 패치 적용 대신 해당 브랜치를
체크아웃할 수 있다. 검토 기준은 localPrototype의 2d6198b이며 임의로 현재 작업을 덮어쓰지 않는다.

사전 확인은 CPU 컨테이너 실행 여부, CUDA 디바이스 미사용, PyTorch2.6.0+cu124·CUDA12.4·
Transformers4.52.4·Python3.11.11, xMem commit/allocator 해시, 컨테이너에 복사된 예측 소스를
확인한다. 패키지를 자동 업그레이드하지 않는다. 환경이 다르면 실행을 중단한다.

그다음 tmux 안에서 기준9조건을 먼저 실행한다.

```bash
python3 tools/run_prediction_suite.py \
  --output-root ~/cloudgpufit-predictions/native-v3 \
  --case-set baseline9 --threads 4
```

기준9조건의 성공·형상·메모리 결과 확인 후 같은 출력 폴더로 전체를 진행한다.
이미 성공한9조건은 코드·입력 identity가 같으면 재사용한다.

```bash
python3 tools/run_prediction_suite.py \
  --output-root ~/cloudgpufit-predictions/native-v3 \
  --case-set all --threads 4
```

CPU에서 실제 inference를 프로파일링하므로 큰 모델·긴 출력은 오래 걸릴 수 있다.
GPU 실측의 몇 분/몇 시간과 같은 시간을 보장하지 않는다. 입력은342조건이고 GPU 실측의
5개 모드를 CPU에서 각각 다시 실행하는 것은 아니다. 기존 CPU 예측 프로세스가 있으면
새 실행이 이를 죽이지 않고 거부한다. 작업 중에는 tmux detach를 사용한다.

결과 수신용 압축 예:

```bash
tar -czf ~/cloudgpufit-native-v3-predictions.tar.gz \
  -C ~/cloudgpufit-predictions native-v3
```

비교는 같은 case_id·revision·actual_input_shapes를 확인한 뒤
`tools/compare_vram_predictions.py --predictions ... --actual ... --output ...`로 한다.
allocated/reserved/NVML 각각 MAPE·절대오차 중앙값·최대오차·과소예측 비율을 계산하고,
누락·실패·T4 BF16 제외를 그대로 남긴다. outside 표만 바뀌면 이 도구가 기존 CPU 예측의
allocated/reserved에 새 outside를 적용하므로 CPU를 다시 돌릴 필요가 없다.

## 5. 검증 상태

단위 회귀 테스트19개와 작은 GPT-2/T5/ViT/ResNet/MobileNetV2의 native FP32·FP16·BF16
실제 CPU 통합 실행15개를 별도로 수행한다. 로컬 검증 런타임은 PyTorch2.6.0+cpu/Python3.12이므로,
inuos의 기존 CUDA 런타임 이미지와 완전히 동일한 환경이라고 주장하지 않는다.
이 차이는 inuos preflight 및 기준9조건으로 확인한다. 현재 GPU 실측은 재실행할 필요가 없다.
