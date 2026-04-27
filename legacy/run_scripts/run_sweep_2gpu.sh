#!/bin/bash
# ============================================================================
# 2-GPU parallel sweep runner
# - Alternates between GPU 0 and GPU 1
# - Runs 2 experiments in parallel, waits for both, then next pair
# - Skips if best_info.json already exists (resume-safe)
# - Usage: bash run_sweep_2gpu.sh
# ============================================================================
set -e

# 공통 하이퍼파라미터 (최근 sweep 에 사용한 winning config)
COMMON_ARGS="--lr_backbone 3e-5 --lr_head 3e-4 --min_epochs 10 --normal_threshold 0.5 --num_workers 8 --prefetch_factor 8"

# ----------------------------------------------------------------------------
# 실험 목록: "method|normal_ratio|seed" 형태
# method = p20 (strict + patience 20) or avg5 (avg_last_n 5)
# ----------------------------------------------------------------------------
EXPERIMENTS=(
    # seed=1: n sweep (2 methods each)
    "p20|700|1"
    "p20|1400|1"
    "p20|2100|1"
    "p20|2800|1"
    "avg5|700|1"
    "avg5|1400|1"
    "avg5|2100|1"
    "avg5|2800|1"

    # n=700: seed sweep (2 methods each)
    "p20|700|2"
    "p20|700|4"
    "p20|700|42"
    "avg5|700|2"
    "avg5|700|4"
    "avg5|700|42"

    # 추가 seed × n 조합 (확장 시 여기에 추가)
    "avg5|1400|2"
    "avg5|2800|2"
    "avg5|1400|42"
    "avg5|2800|42"
)

# ----------------------------------------------------------------------------
# 단일 실험 실행 함수
# $1 = gpu_id, $2 = method, $3 = normal_ratio, $4 = seed
# ----------------------------------------------------------------------------
run_one() {
    local gpu=$1
    local method=$2
    local n=$3
    local s=$4

    # method 별 추가 옵션
    local extra=""
    local tag=""
    case "$method" in
        p20)
            extra="--patience 20"
            tag="p20"
            ;;
        avg5)
            extra="--avg_last_n 5"
            tag="avg5"
            ;;
        *)
            echo "[ERROR] unknown method: $method"
            return 1
            ;;
    esac

    local log_dir="logs/v9_lr3_n${n}_s${s}_${tag}"

    # Resume-safe: best_info.json 있으면 skip
    if [ -f "${log_dir}/best_info.json" ]; then
        echo "[GPU${gpu}] SKIP ${log_dir} (already complete)"
        return 0
    fi

    mkdir -p "${log_dir}"
    # train.py 스냅샷 저장 (기록용)
    cp train.py "${log_dir}/train.py"

    echo "[GPU${gpu}] START ${log_dir}"
    CUDA_VISIBLE_DEVICES=$gpu python train.py \
        --normal_ratio $n --seed $s \
        $COMMON_ARGS $extra \
        --log_dir "${log_dir}" \
        > "${log_dir}_run.log" 2>&1

    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[GPU${gpu}] DONE  ${log_dir}"
    else
        echo "[GPU${gpu}] FAIL  ${log_dir} (rc=${rc})"
    fi
}

# ----------------------------------------------------------------------------
# 메인: 2 experiments at a time (GPU 0 + GPU 1)
# ----------------------------------------------------------------------------
total=${#EXPERIMENTS[@]}
echo "Total experiments: ${total}"
echo "Starting 2-GPU parallel sweep at $(date)"
echo ""

i=0
while [ $i -lt $total ]; do
    # GPU 0 에 i 번째 실험 할당
    exp0="${EXPERIMENTS[$i]}"
    IFS='|' read -r m0 n0 s0 <<< "$exp0"
    run_one 0 "$m0" "$n0" "$s0" &
    pid0=$!

    # GPU 1 에 i+1 번째 실험 할당 (있으면)
    if [ $((i+1)) -lt $total ]; then
        exp1="${EXPERIMENTS[$((i+1))]}"
        IFS='|' read -r m1 n1 s1 <<< "$exp1"
        run_one 1 "$m1" "$n1" "$s1" &
        pid1=$!
    else
        pid1=""
    fi

    # 둘 다 끝날 때까지 대기
    wait $pid0
    if [ -n "$pid1" ]; then
        wait $pid1
    fi

    i=$((i+2))
    echo "--- Batch done. Progress: ${i}/${total} ---"
    echo ""
done

echo "=== ALL DONE at $(date) ==="
echo ""
echo "=== SUMMARY ==="
python -c "
import os, json
rows = []
for d in sorted(os.listdir('logs')):
    if not d.startswith('v9_lr3'): continue
    bf = f'logs/{d}/best_info.json'
    if not os.path.exists(bf): continue
    try:
        j = json.load(open(bf))
        fn = round((1 - j['test_metrics']['abnormal']['recall']) * 750)
        fp = round((1 - j['test_metrics']['normal']['recall']) * 750)
        rows.append((d, fn, fp, fn+fp, j.get('test_f1', 0)))
    except Exception:
        pass
rows.sort(key=lambda r: r[3])
print(f'{\"experiment\":42s} {\"FN\":>4} {\"FP\":>4} {\"TOT\":>4} {\"test_f1\":>8}')
for r in rows:
    print(f'{r[0]:42s} {r[1]:4d} {r[2]:4d} {r[3]:4d} {r[4]:>8.4f}')
"
