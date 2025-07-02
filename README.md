# 🚨 Real Models Anomaly Detection System

고급 이상 탐지 시스템 - 실제 논문 기반 모델들과 앙상블 구현

## 📋 프로젝트 개요

이 프로젝트는 시계열 데이터에서 이상을 탐지하는 7개의 고급 모델을 구현한 종합적인 이상 탐지 시스템입니다. 실제 논문에서 제안된 아키텍처를 기반으로 하며, 5가지 유형의 이상을 탐지할 수 있습니다.

## 🤖 구현된 모델들

### 기본 모델 (4개)
1. **OptimizedPatchTrAD** - Patch-based Transformer with advanced features
2. **OptimizedTraceGPT** - GPT-style Autoregressive Transformer  
3. **OptimizedCARLA** - Contrastive Anomaly Representation Learning
4. **OptimizedProDiffAD** - Progressive Diffusion Model

### 앙상블 모델 (3개)
5. **OptimizedPatchTraceEnsemble** - PatchTrAD + TraceGPT 결합
6. **OptimizedTransferLearningEnsemble** - Transfer Learning 기반
7. **OptimizedMultiModelEnsemble** - 4개 모델 Meta-Ensemble

## 🔍 탐지 가능한 이상 유형

1. **Normal** - 정상 데이터 (화이트 노이즈)
2. **Spike Anomaly** - 급격한 스파이크/점프
3. **Mean Shift** - 평균값 변화 
4. **Std Change** - 분산 변화
5. **Drift/Trend** - 점진적 추세 변화

## 🛠️ 설치 및 요구사항

### 필수 라이브러리
```bash
pip install torch torchvision
pip install numpy
```

### 선택적 라이브러리 (시각화용)
```bash
pip install matplotlib seaborn pandas
pip install scikit-learn
```

> 📝 **참고**: matplotlib, seaborn, sklearn이 없어도 실행 가능합니다. 시각화와 고급 메트릭만 건너뜁니다.

## 🚀 실행 방법

### 기본 실행 (모든 모델 훈련 + 평가)
```bash
cd anomaly_detection
python main.py
```

### 특정 모델만 훈련
```bash
python main.py --model patchtrad
python main.py --model tracegpt
python main.py --model carla
python main.py --model prodiffad
python main.py --model patch_trace_ensemble
python main.py --model transfer_learning_ensemble
python main.py --model multi_model_ensemble
```

### 평가만 실행 (기훈련된 모델 사용)
```bash
python main.py --eval_only
```

### 시각화만 생성
```bash
python main.py --viz_only
```

### 고급 옵션
```bash
python main.py --epochs 20 --batch_size 64 --lr 5e-5
python main.py --data_size 12800 --seq_len 128
```

## 📊 결과 파일 구조

실행 후 다음과 같은 구조로 결과가 저장됩니다:

```
anomaly_detection/
├── main.py
├── README.md
├── samples/                    # 데이터셋 샘플 시각화
│   ├── dataset_samples.png
│   └── anomaly_types_samples.png
├── plots/                      # 모델별 이상 점수 플롯
│   ├── patchtrad_anomaly_scores.png
│   ├── tracegpt_anomaly_scores.png
│   └── ...
├── metrics/                    # 성능 비교 히트맵
│   └── model_metrics.png
├── confusion_matrices/         # 혼동 행렬
│   ├── patchtrad_confusion_matrices.png
│   └── ...
└── pre_trained/               # 훈련된 모델 가중치
    ├── patchtrad_real_final.pth
    └── ...
```

## ⚡ 성능 특징

### 모델별 최적화 설정
- **PatchTrAD**: batch=128, epochs=25, lr=1e-4, hidden=192
- **TraceGPT**: batch=96, epochs=35, lr=5e-5, hidden=384  
- **CARLA**: batch=160, epochs=30, lr=2e-4, hidden=256
- **ProDiffAD**: batch=64, epochs=40, lr=1e-4, hidden=320

### 고급 기능
- ✅ **모델별 최적화된 하이퍼파라미터**
- ✅ **Cosine Learning Rate Scheduling with Warmup**  
- ✅ **Early Stopping & Gradient Clipping**
- ✅ **Multi-GPU DDP 훈련 지원**
- ✅ **실시간 성능 모니터링**
- ✅ **상세한 메트릭 및 시각화**

## 📈 평가 메트릭

### Point-level & Series-level 이상 탐지
- **AUC (Area Under Curve)**
- **Accuracy, Precision, Recall, F1-Score**
- **Confusion Matrix 시각화**
- **ROC Curve & PR Curve**

### 시각화 기능
- 📊 데이터셋 샘플 (5가지 이상 유형별)
- 📈 모델별 성능 히트맵
- 🔍 개별 모델 이상 점수 플롯
- 📋 혼동 행렬 분석

## 🎛️ 이상 탐지 강도 조절

코드에서 다음 파라미터를 조정하여 이상의 강도를 제어할 수 있습니다:

```python
# _generate_complex_series() 함수 내에서
SPIKE_MAGNITUDE = (6.0, 12.0)       # 스파이크 크기
MEAN_SHIFT_MAGNITUDE = (3.0, 6.0)   # 평균 변화 크기  
STD_CHANGE_MULTIPLIER = (5.0, 10.0) # 분산 변화 배수
DRIFT_MAGNITUDE = (3.0, 6.0)        # 드리프트 크기
ANOMALY_POSITION = 'late'           # 이상 위치 ('early'/'middle'/'late'/'random')
```

## 🔧 시스템 요구사항

- **Python 3.7+**
- **PyTorch 1.9+** 
- **CUDA 지원 GPU** (선택사항, CPU도 가능)
- **메모리**: 최소 8GB RAM (GPU 훈련시 4GB+ VRAM)

## 📚 모델 아키텍처 상세

### PatchTrAD
- Overlapping patch extraction with CNN
- Rotary positional encoding
- Multi-scale feature pyramid
- Residual transformer layers

### TraceGPT  
- GPT-style causal attention
- Sliding window mechanism
- Enhanced temporal convolution
- Advanced positional embedding

### CARLA
- Contrastive learning with memory bank
- Multi-scale CNN encoder  
- Momentum-based feature updates
- Temperature-scaled similarity

### ProDiffAD
- Progressive diffusion process
- U-Net architecture with attention
- Cosine noise scheduling
- Cross-attention conditioning

## 🤝 기여하기

이슈나 개선사항이 있으시면 언제든지 GitHub Issues로 알려주세요!

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

**🔥 Real Models로 정확한 이상 탐지를 경험해보세요!** 🔥 