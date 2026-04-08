#!/usr/bin/env bash
# ============================================================================
# Full pipeline orchestrator — Ubuntu 24, H200 × 2, 32 코어, 384GB RAM 폐쇄망
# ============================================================================
# 실행:
#   bash run_pipeline.sh                    # data → images → 모든 실험
#   bash run_pipeline.sh skip-data          # data 생략 (이미 존재)
#   bash run_pipeline.sh skip-data skip-img # data + image 생략 (실험만)
#   bash run_pipeline.sh sweep              # sweep 그룹만
# ============================================================================

set -euo pipefail

cd "$(dirname "$0")"

# ----------------------------------------------------------------------------
# 0. 환경 검증
# ----------------------------------------------------------------------------
echo "============================================================"
echo " Step 0/4 — 환경 검증"
echo "============================================================"

python -c "import torch; print(f'  torch:    {torch.__version__}'); \
print(f'  cuda:     {torch.cuda.is_available()}'); \
print(f'  n_gpus:   {torch.cuda.device_count()}'); \
[print(f'  GPU {i}:    {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory/1024**3:.0f}GB)') for i in range(torch.cuda.device_count())]"

# Pretrained weights — 둘 중 하나 있으면 OK (train.py가 자동 fallback)
#   1. weights/convnextv2_tiny_pretrained.pth   (로컬 fp32, 110MB, gitignored)
#   2. weights/convnextv2_tiny.fp16.pth         (커밋된 fp16, 55MB, repo에 포함)
if [ -f "weights/convnextv2_tiny_pretrained.pth" ]; then
    echo "  weights:  weights/convnextv2_tiny_pretrained.pth (fp32, 로컬)"
elif [ -f "weights/convnextv2_tiny.fp16.pth" ]; then
    echo "  weights:  weights/convnextv2_tiny.fp16.pth (fp16, 커밋됨)"
else
    echo ""
    echo "[ERROR] pretrained weights 없음."
    echo "        repo에 weights/convnextv2_tiny.fp16.pth가 포함돼야 함 (git pull 확인)"
    echo "        또는 'python scripts/download_pretrained.py' 로 fp32 다운 (인터넷 필요)"
    exit 1
fi

# ----------------------------------------------------------------------------
# CLI 파싱
# ----------------------------------------------------------------------------
SKIP_DATA=0
SKIP_IMG=0
GROUPS=""
for arg in "$@"; do
    case $arg in
        skip-data) SKIP_DATA=1 ;;
        skip-img|skip-image|skip-images) SKIP_IMG=1 ;;
        sweep|reg|lr|mc) GROUPS="$GROUPS $arg" ;;
        *) echo "[WARN] unknown arg: $arg" ;;
    esac
done

# ----------------------------------------------------------------------------
# 1. 데이터 생성
# ----------------------------------------------------------------------------
if [ $SKIP_DATA -eq 1 ]; then
    echo ""
    echo "[SKIP] data 생성 (skip-data)"
elif [ -f "data/scenarios.csv" ] && [ -f "data/timeseries.csv" ]; then
    echo ""
    echo "[SKIP] data 생성 (이미 존재 — 강제 재생성하려면 data/ 비우고 재실행)"
else
    echo ""
    echo "============================================================"
    echo " Step 1/4 — 데이터 생성 (32 코어 중 24개 병렬)"
    echo "============================================================"
    python generate_data.py --workers 24
fi

# ----------------------------------------------------------------------------
# 2. 이미지 생성 (32 코어 multiprocessing)
# ----------------------------------------------------------------------------
if [ $SKIP_IMG -eq 1 ]; then
    echo ""
    echo "[SKIP] image 생성 (skip-img)"
elif [ -d "images/train" ] && [ -d "images/test" ] && [ -d "images/val" ]; then
    n_train=$(find images/train -name '*.png' 2>/dev/null | wc -l)
    if [ "$n_train" -gt 0 ]; then
        echo ""
        echo "[SKIP] image 생성 (이미 존재: $n_train train images — 강제 재생성하려면 images/ 비우고 재실행)"
    else
        echo ""
        echo "============================================================"
        echo " Step 2/4 — 이미지 생성 (workers=24)"
        echo "============================================================"
        python generate_images.py --workers 24
    fi
else
    echo ""
    echo "============================================================"
    echo " Step 2/4 — 이미지 생성 (workers=24)"
    echo "============================================================"
    python generate_images.py --workers 24
fi

# ----------------------------------------------------------------------------
# 3. 실험 실행 (H200 2장 병렬, bf16 + compile + bs 256)
# ----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Step 3/4 — 실험 실행 (H200 × 2 병렬, bf16 + compile)"
echo "============================================================"

# torch.compile 캐시 가속용 (한 번 컴파일하면 다음 실험은 즉시 시작)
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PWD/.torch_compile_cache}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

# CUDA mem 단편화 완화 (큰 batch + bf16에서 권장)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ -n "$GROUPS" ]; then
    python run_experiments.py --server h200 --gpus 2 --groups $GROUPS
else
    python run_experiments.py --server h200 --gpus 2
fi

# ----------------------------------------------------------------------------
# 4. 결과 요약
# ----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Step 4/4 — 결과 요약 재생성"
echo "============================================================"
python run_experiments.py --server h200 --only-summary

echo ""
echo "============================================================"
echo " ALL DONE"
echo "============================================================"
echo "  결과: logs/{experiment}/best_info.json"
echo "  요약: logs/experiments_summary.json"
