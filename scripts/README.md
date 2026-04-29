# `scripts/` 안내

이 폴더에는 학습 파이프라인을 돌리는 entrypoint, 큐 가공·실행 도구, 결과 분석 도구, 추론 보조 스크립트가 들어있습니다. 단계별 sweep wrapper(예: `02_lr.sh`)는 `scripts/sweeps_server/`에 따로 있고, 그 안의 `README.md`에서 stage 매핑을 봅니다.

`logs/` 폴더 구조는 단일 학습이면 `logs/<YYMMDD_HHMMSS>_<topic>_F<f1>_R<recall>/`, batch면 `logs/<YYYYMMDD_HHMMSS>_run_paper/<YYMMDD_HHMMSS>_<topic>_F<f1>_R<recall>/`입니다. `<run>` 표기는 그 폴더 하나를 가리킵니다.

---

## 0. `bash scripts/sweeps_server/00_all.sh` 돌렸을 때 어디에 뭐가 쌓이나

### `logs/` 그룹 폴더 (시간 prefix가 앞에 와서 `ls logs/` 시간순 정렬)

```
logs/
└── 20260430_120000_run_paper/                              # 이번 batch group
    ├── 260430_120015_fresh0412_v11_rawbase_lr1e5_n700_s42_F0.9963_R0.9963/
    │   ├── best_model.pth, best_info.json, history.json
    │   ├── confusion_matrix.png, training_curves.png
    │   ├── data_config_used.yaml, train_config_used.yaml
    │   └── predictions/
    ├── 260430_120135_fresh0412_v11_rawbase_lr1e5_n700_s1_F0.9941_R0.9941/
    ├── 260430_120251_fresh0412_v11_rawbase_warmup3_n700_s42_F0.9957_R0.9957/
    ├── ... (한 batch에서 launch된 모든 train run)
```

stage별로 별도로 직접 부른 경우 (예: `bash scripts/sweeps_server/14_backbone.sh`) group 명에 stage 이름이 들어갑니다:

```
logs/20260430_140000_backbone/        # 14_backbone.sh 단독 실행
logs/20260430_150000_sample_skip/     # 13_sample_skip.sh 단독 실행
logs/20260430_153000_bkm_combined/    # 17_bkm_combined.sh 단독 실행
logs/20260430_160000_logical_train/   # 15_logical_train.sh 단독 실행
logs/<YYMMDD_HHMMSS>_<topic>_F_R/     # 단일 python train.py (group 없음)
```

`--log-dir-group my_run` 으로 직접 지정하면 그 이름이 그대로 쓰임.

### `validations/` 결과 (성능 표 + plot + 종합 .md)

stage가 끝날 때마다 controller가 매 run 결과를 `*_results.json`/`*_results.md`에 누적 갱신합니다. postprocess stage가 끝나면 종합 리포트와 축별 plot이 추가로 만들어집니다.

| stage | 진행 중 (live) | 끝나고 (요약) |
|---|---|---|
| 01 baseline | `01_baseline_results.{json,md}` | (별도 plot 없음) |
| 02~12, 16 axes | `02_sweep_results.{json,md}` | postprocess가 `02_sweep_report.md` + `02_sweep_plots/<axis>.png` 14개 |
| 13 sample_skip | `03_sample_skip_results.{json,md}` | stage가 `03_sample_skip_plot.png` |
| 14 backbone | `04_backbone_results.{json,md}` | stage가 `04_backbone_plot.png` |
| 15 logical_train | `server_paper_logical_member_v11_train_summary.{json,md}` | (별도 plot 없음) |
| 17 bkm_combined | `05_bkm_combined_results.{json,md}` | stage가 `05_bkm_combined_plot.png` |
| postprocess | `instability_cases.{csv,json,md}`, `prediction_trend_latest.{csv,json,md}` | 종합 |
| 전체 | `run.log` (run_paper_server_all.sh stdout) | |

### 한 group 폴더만 정리해서 보고 싶을 때

```bash
python scripts/generate_group_report.py \
  --group-dir logs/20260430_120000_run_paper

# 결과:
#   logs/20260430_120000_run_paper/group_report.md
#   logs/20260430_120000_run_paper/group_report_f1.png
#   logs/20260430_120000_run_paper/group_report_val_f1_curves.png
```
candidate 평균 표 + per-run 표 + F1 막대 plot + val_f1 곡선 plot.

### 어디를 보면 한 눈에 볼 수 있나

- 한 batch 결과만: `logs/<group>/group_report.md` (위 명령으로 생성)
- 모든 logs 통합 표/plot: `validations/log_history_report_rawbase.md` + `*_candidate_f1.png`, `*_val_f1_curves.png`, `*_grad_p99_curves.png` (실행: `python scripts/generate_log_history_report.py --logs-dir logs --out-prefix validations/log_history_report_rawbase --contains rawbase`)
- 축별 strict 비교 + 표/plot: `validations/02_sweep_report.md` + `02_sweep_plots/<axis>.png` (postprocess 자동)
- BKM combined 한 줄짜리 비교: `validations/05_bkm_combined_results.md` + `05_bkm_combined_plot.png`
- backbone 비교: `validations/04_backbone_results.md` + `04_backbone_plot.png`

---

## 1. 파이프라인 entrypoint

### `run_paper_server_all.sh`
논문 실험의 메인 orchestrator. weights 다운로드 → 데이터/이미지 생성(없으면) → baseline 5-seed 재확인 → 축별 sweep → postprocess 까지 한 번에. `_common.sh::detect_profile` 의 결과로 num_workers / prefetch / max_launched 기본값이 자동 결정되고, CLI 옵션으로 덮어씁니다.

```bash
# 전체 한 줄
bash scripts/run_paper_server_all.sh

# 데이터·weights는 이미 있다고 가정하고, sweep만
bash scripts/run_paper_server_all.sh --skip-weights --skip-dataset

# 특정 축만
bash scripts/run_paper_server_all.sh \
  --skip-weights --skip-dataset \
  --round1-include-axes lr,warmup --skip-post
```

`scripts/sweeps_server/00_all.sh` 가 이 스크립트를 stage별로 호출합니다.

### `prepare_server_queue.py`
`02_sweep_queue.json` 같은 **template** 큐를 읽고, 서버 실행용 **active** 큐로 변환합니다. 변환 내용:
- tag 앞에 `rawbase_` 붙여서 옛 학습과 구분
- 활성 축이 아닌 인자는 baseline 강제 (single-factor 원칙)
- gc 축은 12 → 5 조건으로 축소
- 이미 끝난 tag는 큐에서 빼버림 (--skip-completed-summary)
- num_workers / prefetch_factor 주입

```bash
python scripts/prepare_server_queue.py \
  --src validations/02_sweep_queue.json \
  --dst validations/02_sweep_active.json \
  --config dataset.yaml \
  --num-workers 24 --prefetch-factor 4 \
  --skip-completed-summary validations/02_sweep_results.json \
  --include-axes lr,warmup
```

### `adaptive_experiment_controller.py`
active 큐를 한 줄씩 읽어 `train.py`를 sequential 하게 launch. 매 run 끝나면 `*_results.json` 과 `*_results.md` 갱신. `--max-launched N` 으로 N 개만 돌리고 빠져나오는 디버그 모드도 있음.

```bash
python scripts/adaptive_experiment_controller.py \
  --queue validations/02_sweep_active.json \
  --summary validations/02_sweep_results.json \
  --markdown validations/02_sweep_results.md \
  --target-min 5 --target-max 15 --stop-mode never \
  --update-live-summary \
  --log-dir-group run_20260430_120000
```

`--log-dir-group` 가 train.py의 `--log_dir_group` 으로 그대로 전달됩니다.

### `validate_dataset.py`
새로 생성된 `data/scenarios.csv` + `data/timeseries.csv` + `display/` 의 sanity 검사. `run_paper_server_all.sh` 가 데이터 생성 직후 자동으로 호출. 직접 부를 일은 거의 없음.

---

## 2. 학습 보조

### `add_training_from_folders.py`
이미 있는 best_model.pth 를 weight initializer 로 두고, `extra_images/{normal,abnormal}/` 폴더의 이미지로 fine-tune.

```bash
python scripts/add_training_from_folders.py \
  --model-run logs/run_20260430_120000/<run> \
  --image-root extra_images \
  --epochs 3 --lr 1e-5 --scheduler cosine
```
출력: `logs/addtrain_*/best_model.pth`, `best_info.json`, `history.json`, `confusion_matrix.png`. 추가 학습 입력은 모델 입력용 이미지(display 아님).

### `generate_per_member_images.py`
member별 highlighted 이미지로 분리해서 logical attribution 학습 데이터셋을 만듦. `15_logical_train.sh` 가 호출. 생성 결과는 `images_per_member_<suffix>/`, `data_per_member_<suffix>/scenarios_per_member.csv`.

```bash
python scripts/generate_per_member_images.py \
  --config configs/datasets/logical_member_v11_source.yaml \
  --suffix logical_member_v11 --workers 24
```

---

## 3. 추론

### `inference.py`  *(루트에 있음, 가장 자주 씀)*
저장된 best 모델로 dataset 전체를 분류해서 결과를 폴더 + 텍스트 + CSV 로 떨굼.

```bash
python inference.py \
  --model logs/run_20260430_120000/<run>/best_model.pth \
  --output_dir my_inference
```
출력 (`my_inference/`):
```
abnormal/             # 불량 판정 display 이미지
normal/               # 정상 판정 display 이미지
predictions.csv       # 모든 chart, p_abnormal, predicted, p_normal
predictions.txt       # 통합 텍스트 (ABNORMAL 위, NORMAL 아래)
abnormal_list.txt     # 불량만 chart_id 목록
normal_list.txt       # 정상만 chart_id 목록
```

### `generate_inference_images.py`
**현업 CSV 가져왔을 때 가장 먼저 부르는 스크립트.** `dataset.yaml` 안 건드리고 외부 timeseries/scenarios CSV 를 모델 입력 이미지로 변환.

```bash
python scripts/generate_inference_images.py \
  --timeseries fab_export/timeseries.csv \
  --scenarios  fab_export/scenarios.csv \
  --out-dir    inference_inputs
# 결과: inference_inputs/{model_inputs/, display/, manifest.csv}

# 그 다음 inference.py 로 분류
python inference.py --model logs/<run>/best_model.pth \
  --data_dir inference_inputs --output_dir fab_results
```

### `server_batch_predict.py` + `run_server_batch_predict.sh`
서버에서 한 모델 폴더 통째로 batch 추론하는 wrapper. 결과 폴더 구조와 텍스트 리스트를 일괄 생성.

```bash
python scripts/server_batch_predict.py --model-run logs/run_20260430_120000/<run>
# 또는
bash scripts/run_server_batch_predict.sh logs/run_20260430_120000/<run>
```

---

## 4. 결과 분석 / 보고서

### `generate_log_history_report.py`
`logs/`(혹은 `logs/<group>/`) 에서 `history.json` + `best_info.json` 을 모두 모아 통합 표·plot 생성. 큰 그림 보고용.

```bash
python scripts/generate_log_history_report.py \
  --logs-dir logs \
  --out-prefix validations/log_history_report_rawbase \
  --contains rawbase --top-k 30
```
출력: `*.md`, `*_candidates.csv`, `*_runs.csv`, `*_candidate_f1.png`, `*_val_f1_curves.png`, `*_grad_p99_curves.png`, `*_fn_fp.png`.

### `generate_group_report.py`  *(NEW)*
한 batch 한 group 폴더(`logs/run_<YYYYMMDD_HHMMSS>/`)만 분석. 통합 리포트보다 빠르고 한 sweep 결과만 보고 싶을 때.

```bash
python scripts/generate_group_report.py --group-dir logs/run_20260430_120000
# 출력: <group>/group_report.md, group_report_f1.png, group_report_val_f1_curves.png
```
candidate 평균 표 + per-run 표 + F1 막대 + val_f1 곡선.

### `generate_strict_one_factor_report.py`
**postprocess stage가 호출**하는 종합 리포트 + 축별 plot 생성기. `02_sweep_results.json` 을 읽고 `02_sweep_report.md`, `02_sweep_results.md`, `02_sweep_plots/<axis>.png` 14개를 생성.

```bash
python scripts/generate_strict_one_factor_report.py \
  --strict-summary validations/02_sweep_results.json \
  --markdown-out validations/02_sweep_results.md \
  --report-out  validations/02_sweep_report.md \
  --plots-dir   validations/02_sweep_plots
```
직접 부를 일은 드물고 보통 `00_all.sh` 마지막에 자동.

### `generate_stage_comparison.py`  *(NEW)*
backbone / bkm_combined / sample_skip 처럼 **단일-축 비교 stage**용. baseline(`01_baseline_results.json`)과 candidate별 평균을 비교한 표 + bar plot 생성.

```bash
# 14_backbone.sh, 17_bkm_combined.sh, 13_sample_skip.sh 끝에서 자동으로 호출됩니다.
python scripts/generate_stage_comparison.py \
  --results validations/04_backbone_results.json \
  --out-md  validations/04_backbone_results.md \
  --out-plot validations/04_backbone_plot.png \
  --title "Backbone sweep"
```

### `collect_instability_cases.py`
`logs/**/history.json` 을 훑어 collapse / oscillation / optimistic spike(val_f1=1 튐) 패턴을 골라 보고. priority case 만 별도 표시.

```bash
python scripts/collect_instability_cases.py
# 출력: validations/instability_cases.{csv,json,md}
```

### `analyze_prediction_trends.py`
완료된 high-F1 candidate 들의 `predictions/` 폴더를 비교해서 **여러 조건에서 계속 틀리는 hard sample** 과 **조건 따라 흔들리는 sensitive sample** 을 구분.

```bash
python scripts/analyze_prediction_trends.py \
  --config dataset.yaml \
  --candidate-prefix fresh0412_v11 \
  --min-f1 0.99 \
  --out-prefix validations/prediction_trend_latest \
  --report-label fresh0412_v11
# 출력: validations/prediction_trend_latest.{csv,json,md} + _review/ 폴더
```

---

## 5. Grad-CAM

### `gradcam_report.py`
class별로 sample 몇 개씩 골라 trend 이미지 위에 CAM colormap 을 얹은 갤러리 생성. 모델이 어디를 근거로 보는지 사람이 확인하는 용도.

```bash
python scripts/gradcam_report.py \
  --model-run logs/run_20260430_120000/<run> \
  --image-root images/test \
  --out-dir validations/gradcam_probe \
  --include-classes normal,mean_shift,standard_deviation,spike,drift,context \
  --limit-per-class 6 \
  --save-heat-only \
  --heat-threshold 0.0 --heat-min-alpha 0.18 \
  --gallery-out docs/gradcam_class_overlay.png
```
출력: `gradcam.csv`, `summary.md`, `overlays/`, `heat_only/`, `cam_on_image/`, 한 장 짜리 갤러리 PNG. CAM 자체는 후처리 룰로 쓰지 않습니다.

### `gradcam_error_report.py`
FP 또는 FN 예측만 따로 추려서 같은 형식으로 갤러리. `--error-type fp/fn` 으로 선택.

```bash
python scripts/gradcam_error_report.py \
  --model-run logs/run_20260430_120000/<run> \
  --split test --error-type fp \
  --out-dir validations/gradcam_fp_analysis \
  --heat-threshold 0.0 --heat-min-alpha 0.18 \
  --limit 6 --fill-hard-normal \
  --gallery-out docs/gradcam_fp_examples.png
```

---

## `sweeps_server/` 보조

축별 stage wrapper 18개와 `00_all.sh`, `_common.sh`, `README.md` 가 들어있는 폴더. 자세한 stage 매핑 + 환경 자동감지 표는 [`scripts/sweeps_server/README.md`](sweeps_server/README.md).

빠른 예시:
```bash
bash scripts/sweeps_server/00_all.sh                # 전체
bash scripts/sweeps_server/02_lr.sh                 # lr 축만
bash scripts/sweeps_server/01_baseline.sh           # baseline 5-seed만
bash scripts/sweeps_server/14_backbone.sh           # weights/*.pth 자동 검출
bash scripts/sweeps_server/17_bkm_combined.sh       # BKM 8개 동시 적용
```

---

## 비고

- `scripts/analyze_*.py` 중 `analyze_prediction_trends.py` 만 git tracked. 그 외 `analyze_anchor_search.py`, `analyze_error_patterns.py`, `analyze_hunting_intervention.py`, `analyze_pilot_19.py`, `analyze_stress_anchor.py`, `analyze_v11_anchor.py`, `analyze_wd005_axis.py` 는 옛 1회성 분석으로 `.gitignore`에 의해 제외됩니다 (로컬 scratch).
- 단일 학습은 `python train.py --log_dir my_run` — 폴더는 자동으로 `logs/<YYMMDD_HHMMSS>_my_run_F<f1>_R<recall>/`.
- batch 학습(`00_all.sh` 등)은 자동으로 `--log-dir-group run_<YYYYMMDD_HHMMSS>` 가 붙어 모든 run이 한 group 폴더 아래에 모입니다.
