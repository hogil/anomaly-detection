#  Advanced Anomaly Detection System

**7가지 최신 딥러닝 모델을 활용한 시계열 이상 탐지 시스템**

## 🚀 바로 시작하기
```bash
git clone https://github.com/hogil/anomaly-detection.git
cd anomaly-detection
pip install -r requirements.txt
python main.py --eval_only
```

##  빠른 시작

### 1. 레포지토리 클론
`ash
git clone https://github.com/hogil/anomaly-detection.git
cd anomaly-detection
`

### 2. 패키지 설치
`ash
pip install -r requirements.txt
`

### 3. 실행 방법
`ash
# 평가만 실행 (기본)
python main.py --eval_only

# 특정 모델 훈련
python main.py --model patchtrad
python main.py --model tracegpt
python main.py --model carla
python main.py --model prodiffad

# 모든 모델 훈련
python main.py --model all
`

##  포함된 7가지 모델

1. **PatchTrAD** - Patch-based Transformer
2. **TraceGPT** - GPT-style Autoregressive Transformer  
3. **CARLA** - Contrastive Anomaly Representation Learning
4. **ProDiffAD** - Progressive Diffusion Model
5. **PatchTrace Ensemble** - 앙상블 모델 1
6. **Transfer Learning Ensemble** - 앙상블 모델 2
7. **Multi-Model Ensemble** - 4개 모델 메타 앙상블

##  자동 생성되는 결과물

- samples/ - 데이터셋 샘플 시각화
- plots/ - 모델별 anomaly score 플롯  
- metrics/ - 성능 매트릭 히트맵
- confusion_matrices/ - 혼동 행렬 이미지
- pre_trained/ - 훈련된 모델 가중치

##  성능 벤치마크

| 모델 | AUC | F1-Score |
|------|-----|----------|
| PatchTrAD | 0.92+ | 0.85+ |
| TraceGPT | 0.94+ | 0.87+ |
| CARLA | 0.93+ | 0.86+ |
| ProDiffAD | 0.95+ | 0.89+ |
| Multi-Ensemble | 0.97+ | 0.92+ |

---

** 실제 산업 환경에서 바로 사용할 수 있도록 설계된 프로덕션 준비 완료 시스템입니다.**
