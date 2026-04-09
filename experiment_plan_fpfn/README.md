# experiment_plan_fpfn — FP/FN 최소화 실험

**목표**: test FP ≤ 2 AND test FN ≤ 2, multi-seed 검증.

자세한 배경 + 단계별 상세: [`plan.md`](plan.md)

---

## 폴더 구조

```
experiment_plan_fpfn/
├── README.md              ← 이 파일 (실행 가이드)
├── plan.md                ← 전체 실험 계획 (5 phase)
├── spike_findings.md      ← 139 spike / 82 runs 분석 결과
├── run_phase0_patch.sh    ← Phase 0: train_tie.py 패치 검증
├── run_phase1_baseline.sh ← Phase 1: EMA spike-proof baseline (5 seed)
├── run_phase2_lr_warmup.sh ← Phase 2: LR/warmup 변주 (9 runs)
├── run_phase3_long.sh     ← Phase 3: epoch/patience 확장 (6 runs)
├── run_phase4_reg.sh      ← Phase 4: regularization (6 runs)
├── monitor.py             ← 실시간 모니터: 나쁜 run 자동 kill
├── analyze.py             ← phase 후 결과 분석 + 다음 config 제안
└── results.md             ← (자동 생성) 각 phase 결과 누적
```

---

## 로컬 실행 (Windows bash, RTX 4060 Ti)

```bash
cd D:/project/anomaly-detection

# 0) 사전 조건 확인
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
ls data/scenarios.csv images/train weights/convnextv2_tiny.fcmae_ft_in22k_in1k.pth

# 1) Phase 1 실행 (5 seed, ~100분)
bash experiment_plan_fpfn/run_phase1_baseline.sh

# 2) monitor 를 별도 터미널에서 실행 (옵션 — 자동 kill)
python experiment_plan_fpfn/monitor.py --watch logs/v9_phase1_*

# 3) 결과 분석
python experiment_plan_fpfn/analyze.py --phase 1

# 4) Phase 1 결과 보고 Phase 2 실행 여부 결정
bash experiment_plan_fpfn/run_phase2_lr_warmup.sh  # 9 runs ~180분
python experiment_plan_fpfn/analyze.py --phase 2

# 5) 이하 동일
bash experiment_plan_fpfn/run_phase3_long.sh
bash experiment_plan_fpfn/run_phase4_reg.sh

# 6) 최종 ensemble
python experiment_plan_fpfn/ensemble_infer.py --top_n 5
```

### 백그라운드 실행 + 로그 tail

```bash
# 백그라운드로 phase 1 돌리고 stdout 을 파일로
bash experiment_plan_fpfn/run_phase1_baseline.sh > logs/phase1_stdout.log 2>&1 &
echo "PID: $!"

# 진행상황 보기
tail -f logs/phase1_stdout.log
```

---

## 회사 서버 (Ubuntu H200 × 2) 실행

```bash
# 1) repo 업로드 (git pull 또는 rsync)
git pull  # 또는
rsync -avz --exclude logs/ --exclude images/ --exclude data/ \
  local:/d/project/anomaly-detection/ server:/path/to/anomaly-detection/

# 2) weights 업로드 (첫 1회만)
rsync -avz weights/ server:/path/to/anomaly-detection/weights/

# 3) 서버에서 데이터 생성 (필요시)
cd /path/to/anomaly-detection
python generate_data.py --workers 24
python generate_images.py --workers 24

# 4) Phase 1 ~ 4 순차 실행
bash experiment_plan_fpfn/run_phase1_baseline.sh
bash experiment_plan_fpfn/run_phase2_lr_warmup.sh
bash experiment_plan_fpfn/run_phase3_long.sh
bash experiment_plan_fpfn/run_phase4_reg.sh

# 또는 run_pipeline.sh 에 추가해서 한 방에
bash run_pipeline.sh  # skip-data skip-img 포함
```

### H200 병렬 가속

H200 2장이면 `run_experiments.py` 의 서버 프로파일 + `--gpus 2` 로 2배 병렬.
Phase 1/2/3/4 의 runs 를 `run_experiments.py` 의 `build_experiments()` 에 추가하면 자동 분산.

### 예상 시간

| 환경 | Phase 1 | Phase 2 | Phase 3 | Phase 4 | 총 |
|---|---:|---:|---:|---:|---:|
| RTX 4060 Ti (1장) | 100분 | 180분 | 180분 | 120분 | **~11 시간** |
| H200 × 2 (병렬) | ~25분 | ~45분 | ~45분 | ~30분 | **~2.5 시간** |

---

## 중단 + 재개

모든 실험은 **log_dir 단위 독립** 이라 중간에 Ctrl+C 해도 다른 run 영향 없음.
재개할 때는 이미 완료된 폴더 (`best_info.json` 존재) 는 run script 가 자동 skip.

**원칙**: 이미 돌린 run 의 `logs/<run_dir>` 은 **절대 삭제 금지**. 새 시도는 새 폴더명 (예: `_v2`, `_retry` suffix).

---

## 결과 수집 (로컬 ← 서버)

```bash
# 서버에서 결과만 tar (약 100MB)
tar czf results_fpfn.tar.gz logs/v9_phase*_*/best_info.json logs/v9_phase*_*/history.json logs/v9_phase*_*/test_history.json logs/v9_phase*_*/confusion_matrix*.png

# 로컬로 전송
scp server:/path/to/anomaly-detection/results_fpfn.tar.gz .
tar xzf results_fpfn.tar.gz -C D:/project/anomaly-detection/
```

---

## Troubleshooting

| 증상 | 원인 | 해결 |
|---|---|---|
| `CUDA OOM` | bs 32 초과 | `--batch_size 16` |
| `FileNotFoundError: weights/...` | weight 미업로드 | `python download.py` |
| Phase 중간에 kill 됨 | monitor 가 성능 낮음 감지 | `logs/<run>/killed.txt` 확인 후 config 수정 |
| `test_f1` 이 history 보다 낮음 | tie-update overwrite (Phase 0 패치 전) | train_tie.py 가 patched 인지 확인 |
| Phase 1 에서 spike 빈번 | EMA 적용 안 됨 | `best_info.json.hparams.ema_decay` 확인 |
