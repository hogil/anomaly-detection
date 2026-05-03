# Problem Setting: Defect Image Detection

## Decision

이 프로젝트의 1차 운영 문제는 **binary pass/fail 판정**입니다.

- Primary output: `normal`, `abnormal`
- Primary mode: `python train.py --mode binary`
- Primary optimization target: abnormal recall, FN count, binary F1, AUROC, threshold별 FN/FP
- Secondary analysis: defect type별 recall과 confusion matrix
- Optional baseline: normal 중심 또는 one-class anomaly detection 계열

현업에서 먼저 필요한 판단은 결함 원인명이 아니라 "불량 이미지를 놓치지 않고 선별하는가"입니다. Defect type별 데이터가 충분하지 않거나 새 defect type이 계속 들어오는 상황에서는 단독 multiclass classifier가 rare/unseen defect에 약하므로, 기본 운영안은 binary gate로 둡니다.

## Repository Mapping

현재 repo는 이미 세 가지 supervised mode를 지원합니다.

| mode | label space | use |
| --- | --- | --- |
| `binary` | `normal`, `abnormal` | 주 실험 및 운영 gate |
| `multiclass` | `normal` + defect types | 결함 원인 분석, 리포트, 보조 비교 |
| `anomaly_type` | defect types only, normal 제외 | binary gate 뒤의 defect type classifier 후보 |

주의: repo의 `anomaly_type`은 one-class anomaly detection이 아닙니다. 정상 이미지만으로 학습하는 PatchCore/EfficientAD/SimpleNet류 baseline은 별도 baseline으로 추가해야 합니다.

## Operating Protocol

1. `binary`를 기준 실험으로 둡니다.
2. 같은 dataset split과 seed로 `multiclass`를 보조 실행합니다.
3. Binary 결과는 abnormal recall, FN, FP, F1, AUROC, threshold sweep으로 판단합니다.
4. Multiclass 결과는 전체 accuracy보다 defect type별 recall과 confusion matrix를 봅니다.
5. Grad-CAM 또는 anomaly map으로 모델 근거가 실제 결함 영역 또는 target member 판단과 맞는지 확인합니다.
6. 운영 구조는 `binary gate -> defect type classifier`를 기본 후보로 둡니다.

구현된 2-stage 실행법은 `docs/two_stage_workflow.md`를 기준으로 봅니다. `scripts/two_stage_predict.py`는 1차 binary model이 `abnormal`로 보낸 chart만 2차 `anomaly_type` model에 넣어 `abnormal/<type>` 결과를 저장합니다.

## Recommended Setup

### Primary: binary image-level defect detection

- Output: `normal`, `abnormal`
- Use: pass/fail 운영 gate
- Main metrics: abnormal recall, FN count, FP count, binary F1, AUROC, threshold별 confusion matrix
- Operating threshold: 기본 `normal_threshold=0.9`; FN 비용이 크면 threshold sweep 결과로 조정

권장 binary run:

```bash
python train.py \
  --config dataset.yaml \
  --mode binary \
  --epochs 20 \
  --batch_size 32 \
  --precision fp16 \
  --normal_ratio 700 \
  --seed 42 \
  --log_dir binary_gate_s42
```

### Secondary: multiclass defect type analysis

- Output: `normal` + defect types
- Use: 원인 분석, 리포트, 후처리 routing
- Main metrics: defect type별 recall, confusion matrix
- 운영 gate로 단독 사용하려면 binary 대비 FN 증가가 없어야 합니다.

보조 multiclass 비교:

```bash
python train.py \
  --config dataset.yaml \
  --mode multiclass \
  --epochs 20 \
  --batch_size 32 \
  --precision fp16 \
  --seed 42 \
  --log_dir aux_multiclass_s42
```

### Optional: one-class/anomaly detection baseline

Abnormal sample이 부족하거나 unseen defect 가능성이 크면 PatchCore, EfficientAD, SimpleNet, CutPaste, DRAEM 계열을 같은 split에서 비교합니다. 현재 repo의 `anomaly_type`은 이 baseline이 아니라 정상 제외 defect type classifier입니다.

Labeled inference 결과의 AUROC와 threshold별 FN/FP:

```bash
python scripts/binary_threshold_report.py \
  --predictions <inference_output>/predictions.csv
```

## Test Plan

1. 같은 split/seed에서 `binary`와 `multiclass`를 모두 실행합니다.
2. Binary는 abnormal recall, FN, FP, F1, AUROC, threshold sweep을 봅니다.
3. Multiclass는 전체 accuracy보다 defect type별 recall과 confusion matrix를 봅니다.
4. Labeled inference 결과가 있으면 `scripts/binary_threshold_report.py`로 threshold별 FN/FP를 확인합니다.
5. Grad-CAM 또는 anomaly map으로 모델이 실제 결함 영역 또는 target member를 보는지 확인합니다.
6. 최종 선택 기준:
   - binary가 FN이 가장 낮고 안정적이면 binary를 운영 gate로 사용합니다.
   - multiclass가 type별 recall도 안정적이면 binary 뒤 2단계 classifier로 추가합니다.
   - unseen defect나 label 부족 문제가 크면 one-class baseline을 병행합니다.

## Evidence

| source | relevant signal for this project |
| --- | --- |
| [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) | 산업 검사 benchmark이며 defect-free training images와 정상/불량 test set 구조를 둔다. |
| [PatchCore, CVPR 2022](https://arxiv.org/abs/2106.08265) | 정상 image feature memory bank 기반 one-class/cold-start 문제를 다루며 MVTec AD image-level AUROC 99.6%를 보고한다. |
| [CutPaste, CVPR 2021](https://arxiv.org/abs/2104.04015) | anomalous data 없이 unknown anomaly pattern을 검출하는 normal-only/self-supervised framing이다. |
| [DRAEM, ICCV 2021](https://arxiv.org/abs/2108.07610) | surface anomaly detection을 normal/anomalous decision boundary와 localization 문제로 둔다. |
| [EfficientAD, WACV 2024](https://arxiv.org/abs/2303.14535) | 실시간 산업 비전에서 anomaly detection latency와 처리량을 핵심 제약으로 다룬다. |
| [SimpleNet, CVPR 2023](https://arxiv.org/abs/2303.15140) | feature-space synthetic anomaly와 binary anomaly discriminator로 class-agnostic anomaly detection을 수행한다. |
| [Valeo Anomaly Dataset / SegAD, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Baitieva_Supervised_Anomaly_Detection_for_Complex_Industrial_Images_CVPR_2024_paper.pdf) | real production dataset에서 defect subclass가 많아도 final anomaly score를 supervised AD 목표로 둔다. |
| [Steel surface defect CAD, Scientific Reports 2026](https://www.nature.com/articles/s41598-025-34320-9) | binary detection stage와 defect classification stage를 분리하며, binary detection이 simultaneous detection/classification보다 좋은 결과를 보였다고 보고한다. |
| [Applied Sciences 2022 steel defect hybrid architecture](https://www.mdpi.com/2076-3417/12/12/6004) | 상위 layer에 binary classifier를 두고 이후 scratch segmentation과 다른 defect object detection을 적용하는 계층형 구조를 사용한다. |

## Decision Gates

Multiclass를 운영 1단계로 올리는 조건:

- defect type별 train/val/test sample이 충분하다.
- 새 defect type보다 기존 defect type routing이 더 중요하다.
- binary gate 대비 FN 증가가 없거나 운영상 허용 가능하다.
- type별 recall과 confusion matrix가 안정적이다.

One-class/anomaly detection baseline을 추가하는 조건:

- abnormal sample이 적거나 defect type drift가 크다.
- 신규 defect를 기존 class로 강제 매핑하기 어렵다.
- binary supervised model의 FN이 특정 unseen pattern에 집중된다.
- 생산라인 적용에서 라벨링 속도가 병목이다.

## Assumptions

- 최종 목표는 결함 원인명 분류보다 불량 이미지 선별/pass-fail 판정입니다.
- 결함 type별 데이터는 normal보다 적고 불균형할 가능성이 큽니다.
- 새 defect type이 나올 수 있으므로 multiclass 단독 운영은 기본안에서 제외합니다.
- 현재 repo의 `anomaly_type`은 PatchCore류 one-class anomaly detection이 아니라 정상 제외 defect type classifier입니다.
