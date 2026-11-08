# CPU 예측: RAM 20GiB 기준 병렬 작업 시작 제어

기본 최대 동시 작업은 2개입니다. 신규 조건을 시작하기 전에 `/proc/meminfo`의 `MemTotal - MemAvailable`로 호스트 전체 사용 RAM을 계산합니다. 다른 프로세스의 메모리 압박도 포함하며, 회수 가능한 cache는 MemAvailable을 통해 가용량에 반영됩니다.

- 사용 RAM이 **20GiB 이하**이고 가용 RAM이 host reserve 이상이며 동시 실행 슬롯이 남으면 다음 조건을 시작합니다.
- 20GiB를 초과하면 새 조건만 대기합니다. 이미 실행 중인 조건은 계속 실행하며, 매초 확인해 사용량이 줄면 다음 조건을 시작합니다.
- 기본 최대 2개 조건 × 조건당 CPU 스레드 4개입니다. 동시 기동은 기본 5초 간격으로 배치해 초기 메모리 증가를 관찰합니다. 활동 중인 작업이 없으면 이 간격 때문에 기다리지 않습니다.
- 조건별 RAM hard limit(기본 32GiB), swap 비활성, timeout, 로그·종료 상태는 유지합니다. 메모리 때문에 시작을 기다리는 시간은 해당 조건의 실행 timeout에 포함하지 않습니다.
- 시작하지 않은 대기 조건을 error로 기록하지 않습니다. scheduler.jsonl과 manifest에 시작·대기·완료, 활동 조건 및 시작 판단 시 메모리를 기록합니다.
- 취소 신호를 모든 worker에 전달하고 각 작업 컨테이너를 정리한 뒤 취소 결과를 저장합니다. 한 조건의 일반 오류/OOM/timeout은 다른 조건을 취소하지 않습니다. 정확 비교가 불일치하면 신규 시작을 중단하고 활동 작업을 정리합니다.

**20GiB는 신규 작업을 시작하는 기준이며 시작 후 총 RAM 증가를 제한하지는 않습니다.** 조건별 32GiB 상한 두 개의 합계는 현재 약 47GiB인 호스트 RAM을 넘을 수 있습니다. 이미 단일 조건 RSS가 약 31.4GiB인 사례를 확인했으므로 이 정책만으로 host OOM 방지를 보장할 수는 없습니다. 기본 동시 개수를 2개로 두었으며, 공유 총 RAM hard cap은 구현하지 않았습니다.

## 전체 342개 실행

현재 INUOSVM-1의 준비된 환경과 tmux 세션에서 실행합니다.

```bash
cd /home/inuos/CloudGPUFit-cpu-singlepass-20261108 || exit 1
set -o pipefail
cgf_out="/home/inuos/cloudgpufit-predictions/parallel-all342-$(TZ=Asia/Seoul date +%Y%m%d-%H%M%S)"
python3 -u tools/run_prediction_recovery.py \
  --all \
  --output-root "$cgf_out" \
  --snapshot-image cloudgpufit-cpu-validation:20261108 \
  --jobs 2 --threads 4 \
  --start-below-used-gib 20 \
  --memory-gib 32 --host-reserve-gib 6 \
  --timeout-seconds 21600 \
  2>&1 | tee "${cgf_out}.console.log"
```

`--all`은 기존 성공을 포함한 342개 core 조건을 모두 선택합니다. 결과 폴더는 새 경로여야 합니다. 예측 기본값은 여전히 warmup 0회·측정 1회이며 native dtype과 전체 generate를 유지합니다. 시간제한 6시간은 조건당 제한입니다.

## 실제 검증

- 호스트 회귀검사 50개 중 5개는 준비된 torch/xMem 환경이 필요해 생략했고, 격리 VM 환경에서는 50개 모두 통과했습니다.
- 20GiB 경계의 등호 포함, 초과 시 대기와 감소 후 재개, 최대 동시 개수, RAM 증가 시 기존 작업 유지, 정확 비교 불일치 시 활동 작업 취소, 전체 취소 전파를 검사했습니다.
- 실제 Docker 병렬 예측: GPT-2 small FP32 01과 T5-small FP32 01이 겹쳐 실행됐고, 둘 다 기존 성공값 대비 allocated·reserved 차이가 각각 0 byte였습니다.
- 실제 저메모리 테스트 컨테이너 두 개에서 개별 timeout이 다른 작업의 정상 완료를 방해하지 않는지 확인했습니다. SIGTERM 취소 시에는 두 컨테이너 모두 cancelled로 기록하고 제거했습니다.
- 원본 xmem-cpu에는 sleep infinity만 남아 있습니다. 342개 전체 실험은 시작하지 않았습니다.

증거: `/home/inuos/cloudgpufit-predictions/parallel-audit-20261108`, `parallel-check-01`.
