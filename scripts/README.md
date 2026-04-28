# scripts/

`scripts/`는 용도가 섞여 있어서, 아래 기준으로 보면 됩니다.

## 1. Batch/server inference

- `server_batch_predict.py`
  - 여러 제품 폴더를 스캔해서 best model로 일괄 추론
- `run_server_batch_predict.sh`
  - 위 스크립트 shell wrapper

## 2. Dataset / rendering helpers

- `generate_per_member_images.py`
  - per-member 이미지 확장
- `generate_mean_shift_preview.py`
  - preview 생성
- `validate_dataset.py`
  - 생성 데이터 검증

## 3. Experiment orchestration

- `adaptive_experiment_controller.py`
- `adaptive_master_loop.py`
- `run_strict_single_factor_followup_20260426.ps1`
- `select_strict_single_factor_refinements.py`
- `select_paper_core_axis_refinements.py`
- `select_paper_axis_expansions.py`

## 4. Reporting / analysis

- `generate_strict_one_factor_report.py`
- `report_baseline_deltas.py`
- `collect_instability_cases.py`
- `analyze_prediction_trends.py`
- `generate_paper_plots.py`
- `generate_paper_plots_n122.py`
- `rliable_analysis.py`
- `mcnemar_compare.py`
- `build_golden_recipe.py`

## 5. Sweep launchers

- `sweeps_server/`
  - current paper server launchers
- `sweeps_laptop/legacy/`
  - archived laptop exploration launchers

현재 서버 실험은 `sweeps_server/`를 진입점으로 봅니다. `legacy/` 아래 스크립트는 과거 재현 기록입니다.

## 6. Historical one-off scripts

이름에 날짜가 붙거나 특정 실험 조건을 직접 드러내는 파일은 대체로 one-off입니다.

예:
- `aggregate_vd080_bc.py`
- `paper_followup_v11.py`
- `analyze_v11_anchor.py`
- `analyze_wd005_axis.py`

이들은 필요할 때만 보고, 기본 진입점으로 삼지 않습니다.
