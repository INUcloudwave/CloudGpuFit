#!/usr/bin/env bash
# setup.sh — xMem 추론 VRAM 예측 도구 환경 세팅
#
# 이 저장소 루트에서 실행한다.
#   bash setup.sh
#
# 하는 일:
#   1) 작업 디렉토리 준비 (~/xmem-work)
#   2) pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime 이미지 pull
#   3) 컨테이너 실행 (xmem-cpu, sleep infinity)
#   4) xMem 저장소를 특정 커밋(1ee4bd08)으로 clone
#   5) 코어 패치 3개 파일 적용 (patches/core_inference.patch)
#   6) predict_vram.py 복사
#   7) 추가 패키지 설치

set -euo pipefail

XMEM_COMMIT="1ee4bd08dda277a6248abee5b6d40e0d6067efaa"
WORK_DIR="${HOME}/xmem-work"
CONTAINER_NAME="xmem-cpu"
IMAGE="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/7] 작업 디렉토리 준비: ${WORK_DIR}"
mkdir -p "${WORK_DIR}"

echo "[2/7] 이미지 pull: ${IMAGE}"
docker pull "${IMAGE}"

echo "[3/7] 컨테이너 실행: ${CONTAINER_NAME}"
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "  이미 존재함, 재사용"
else
  docker run -d \
    --name "${CONTAINER_NAME}" \
    -v "${WORK_DIR}:/work" \
    "${IMAGE}" \
    sleep infinity
fi

echo "[4/7] xMem clone (commit ${XMEM_COMMIT:0:8})"
if [ ! -d "${WORK_DIR}/xMem" ]; then
  git clone https://github.com/Stone-ResearchLife/xMem "${WORK_DIR}/xMem"
  (cd "${WORK_DIR}/xMem" && git checkout "${XMEM_COMMIT}")
else
  echo "  이미 존재함, 건너뜀"
fi

echo "[5/7] 코어 패치와 native inference 패치 적용"
cd "${REPO_DIR}"
python3 - <<'PYSETUP'
from orchestrator.setup_manager import _apply_patch_if_needed, _sync_predictor, _xmem_dir
_apply_patch_if_needed(_xmem_dir())
_sync_predictor()
PYSETUP

echo "[7/7] 추가 패키지 설치"
docker exec -t "${CONTAINER_NAME}" bash -c "
  pip install --no-cache-dir \
    sortedcontainers==2.4.0 \
    'pydantic~=2.10.6' \
    fire==0.7.1 \
    ures==1.9.1 \
    'datasets~=3.0.2' \
    tqdm==4.67.1 \
    tensorboardX==2.6.2.2 \
    'pynvml~=12.0.0' \
    GPUtil==1.4.0 \
    plotly==6.9.0 \
    matplotlib \
    transformers==4.52.4 \
    accelerate==1.4.0
"

echo ""
echo "완료. 실행 예시:"
echo "  docker exec -t ${CONTAINER_NAME} bash -c \"cd /work/xMem && python predict_vram.py --model facebook/opt-125m --batch_size 4 --seq_len 512 --dtype fp16\""
