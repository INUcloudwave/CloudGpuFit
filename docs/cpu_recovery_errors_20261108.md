# CPU xMem 예측 복구: 오류와 수정 내역

2026-11-08 INUOSVM-1의 실제 코드·결과·컨테이너를 확인하고 검증한 내용입니다. 현재 기본 정책은 **warmup 0회, 측정 추론 1회**이며 예측 버전은 `4-native-inference-single-request`입니다. 이전 진단·검증 문서의 warmup 1회/측정 3회 설명은 당시 버전의 기록입니다.

## 과거 실행에서 발생한 오류

### 프로세스가 결과 JSON 없이 종료됨

이전 실행의 오류 파일 17개는 대부분 `Prediction JSON missing; inspect the case log`로 저장돼 있었습니다. 이는 실제 원인을 설명하는 오류가 아니라, 예측 프로세스가 결과 JSON을 남기지 못했다는 후속 증상입니다.

진단 기록상 16개는 커널의 Python OOM 종료 시각과 오류 파일 시각이 0~2초 차이로 일대일 대응합니다. 기존 실행기가 조건별 host PID를 남기지 않았으므로 시각 대응에 근거한 판단이며, 각 조건과 피해 PID가 실행기에서 직접 연결된 증거는 없습니다. 나머지 한 개는 마지막에 사용자가 중단한 GPT-2 XL FP16 조건입니다. 종료 코드 137만으로 OOM이라고 판정하지 않습니다.

원인·기여 요인:

- `ram_reservation_gb`는 병렬 작업 선택에 쓰는 추정치였고 실제 메모리 제한이 아니었습니다. 기존 컨테이너의 Docker Memory/MemorySwap은 0이었습니다.
- 모델과 입력, native profiler 결과, 전체 trace JSON, 수명 목록, allocator 분석 자료가 같은 프로세스에서 겹쳐 살아 있었습니다.
- upstream `Trace._record`는 이벤트마다 segment 참조 목록과 상세 메시지를 저장했습니다. 이 비용은 이벤트 수와 segment 수에 따라 커지며 peak 계산 자체에 필요하지 않습니다.
- 큰 FP16 조건은 단일 예측만 실행 중인 구간에도 OOM과 대응해, 병렬 수 감소만으로 충분하지 않았습니다.

적용한 수정:

- stdlib JSONDecoder로 trace를 순차 읽고 CPU 메모리 이벤트와 추론 구간만 보관합니다. 필터링 뒤 stable timestamp sort를 유지하며 구간 밖 free도 수명 매칭에 포함합니다.
- profiler의 참조 생존 범위를 분리합니다. 이것만으로 RSS가 즉시 OS에 반환된다고 보장하지는 않습니다.
- upstream allocator 전이와 byte counter를 유지하고 시각화용 `_record`만 생략합니다. upstream 파일이나 allocator lock은 변경하지 않았습니다.
- 작업별 독립 컨테이너에 RAM hard limit, swap 비활성, timeout을 적용합니다. 기존 source/cache는 읽기 전용으로 연결합니다.
- 종료 코드, Docker OOMKilled, 오류 단계/traceback, host PID/cgroup, 실제 제한과 mount를 기록합니다. OOM·timeout·취소·일반 프로세스 종료를 구분하고 작업 컨테이너를 정리합니다.
- 무제한 shared-container 실행기는 명시적 legacy opt-in 없이 실행되지 않도록 합니다.

### simulator가 일반 예외도 OOM으로 표시함

upstream `AllocatorSim.simulate`는 메모리 부족 이외의 예외도 `oom=True`로 바꿉니다. 예외를 보고하는 allocator wrapper를 추가해 알려진 용량 부족과 일반 allocator 오류를 구분합니다. 원인이 보존되지 않은 내부 오류도 용량 부족이라고 단정하지 않습니다.

### CUDA code 35 경고

`gpuGetDeviceCount failed with code 35`는 기존 성공 244개 로그에도 등장했습니다. CPU 예측 실패의 원인으로 확인된 경고가 아니며, GPU 사용을 켜서 해결하려고 하지 않았습니다.

## 준비·복구 과정에서 발견한 오류

| 오류/문제 | 확인 및 조치 |
|---|---|
| `fatal: not a git repository` | `--repo .`가 저장소 밖을 가리킨 상황과 일치합니다. 실제 저장소 절대경로로 별도 worktree 준비를 실행해 성공했습니다. 원본 checkout을 reset/clean하지 않았습니다. |
| Docker `invalid reference format` | Docker 29.7.2가 commit 경고를 stdout에 출력했고, 제공 패치가 경고와 SHA 전체를 이미지 ID로 사용했습니다. commit 뒤 `docker image inspect`로 ID를 얻도록 수정했습니다. |
| read-only 원본 아래 새 helper mountpoint 문제 | 실제 발생한 오류로 단정하지 않습니다. 원본에 없는 파일을 그 아래 overlay하는 설계를 보완해 별도 `/recovery` 읽기 전용 mount에서 새 predictor를 실행합니다. |
| 새 helper가 cache fingerprint에서 누락 | `inference_trace_runtime.py`를 fingerprint에 포함하고 정책 버전을 갱신했습니다. |
| 기존값 비교에서 revision 미확인 | 요청·threads뿐 아니라 입력과 결과의 revision도 확인합니다. allocated/reserved는 각각 정확한 byte로 비교합니다. |

## 속도 개선

한 요청의 VRAM peak 예측에는 여러 요청 반복이 필수가 아니므로 기본값을 warmup 0회/측정 1회로 변경했습니다. 이전 3회는 평균을 내던 것이 아니라 같은 allocator에 연속 재생하던 방식이었습니다. CPU warmup 이력은 원래도 GPU allocator에 재생하지 않았습니다.

full generate와 native dtype, KV 추적, 모델/입력 상주 storage, allocator 전이를 유지합니다. 한 번의 generate 안에서 필요한 토큰별 decode는 생략하지 않습니다. 명시적 1/3 옵션은 진단 비교에만 사용할 수 있습니다.

| 조건 | 이전 1+3 총시간 | 변경 0+1 총시간 |
|---|---:|---:|
| GPT-2 small FP16 01 | 335.5초 | 104.5초 |
| GPT-2 small FP32 01 | 35.4초 | 15.9초 |
| GPT-2 small FP32 12 | 111.4초 | 43.2초 |
| T5-small FP32 01 | 24.9초 | 12.1초 |

동일 VM·snapshot·threads=4·4 CPU quota·32GiB 제한에서 순차 실행한 단일 비교쌍입니다. 반복 통계 benchmark나 순수 kernel 속도비가 아닙니다.

## 검증 결과와 남은 문제

- VM 회귀검사 43개와 작은 모델 5종 × 정밀도 3개 통합검사 15개가 통과했습니다.
- 원래 크기 대표 6개에서 allocated와 reserved가 이전 1+3 결과와 각각 **0 byte 차이**였습니다.
- 종료 7, OOM 증거 없는 종료 137, 실제 64MiB 제한 OOM, timeout, SIGTERM 취소를 검증했고 작업 컨테이너가 정리됐습니다. SIGKILL/호스트 장애에서 Python finally 실행까지 보장하지는 않습니다.
- 원본 결과·로그 787개와 원본 source/Git 상태, 이전 검증 worktree는 보존했습니다.
- 과거 실패 대표 GPT-2 FP32 12는 성공했지만, 단일 추론에서도 peak RSS가 **31.426GiB**였습니다. RAM 폭증이 모두 해결된 것은 아닙니다.
- FP16은 단일 추론에서도 CPU 계산 비용이 큽니다. 전체 342개 재실행은 하지 않았으며, 기존 실패/미실행 중 **97개가 남아 있습니다**.

단일 요청의 상세 byte 값·시간·RAM 표는 [단일 요청 VM 검증 보고서](cpu_single_request_validation_20261108.md)를, OOM 진단 근거는 [이전 진단](cpu_prediction_recovery_20261108.md)을 참고하세요.
