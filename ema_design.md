# EMA Implementation Design (Phase 3)

**주의: Phase 1-A, 1-B, 2가 모두 끝난 후 train.py에 반영할 것. 그 전에는 절대 train.py 수정 금지 (체인 일관성).**

## 목표
- Val 1.0 spike 자체 차단 (smoothed median은 우회, EMA는 weight space에서 제거)
- Smoothed median과 disentangle: seed 1 기준 EMA 단독 성능 비교
- 기존 winning config 최소 침범 (additive, default off)

## 설계 원칙
1. `--ema_decay` default 0.0 → 미사용 (기존 동작 보존)
2. `--ema_decay 0.9999` 지정 시 활성화
3. EMA 모델은 매 step 갱신
4. 평가(val/test)는 EMA 모델로
5. Best model 저장도 EMA 상태 저장
6. 나머지 smoothed median + patience 로직은 그대로 유지 (상호 보완)

## ModelEMA 클래스 (timm/ConvNeXt-V2 official 스타일)

```python
import copy
import torch

class ModelEMA:
    """Exponential Moving Average of model weights.
    
    Reference:
    - Tarvainen & Valpola 2017 (Mean Teacher): arXiv:1703.01780
    - Izmailov et al. 2018 (SWA paper mentions EMA): arXiv:1803.05407
    - ConvNeXt-V2 official repo: decay 0.9999
    - timm.utils.ModelEmaV2
    """
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
    
    @torch.no_grad()
    def update(self, model):
        # state_dict includes BN running stats — must copy those too
        for ema_v, v in zip(self.module.state_dict().values(),
                            model.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(v.detach().to(ema_v.dtype),
                                             alpha=1.0 - self.decay)
            else:
                ema_v.copy_(v)
```

## train_one_epoch 수정

```python
def train_one_epoch(..., ema=None):
    ...
    for batch in pbar:
        ...
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # EMA update (after optimizer.step)
        if ema is not None:
            ema.update(model)
        ...
```

## main() 수정

```python
# 1. Argparse
parser.add_argument("--ema_decay", type=float, default=0.0,
                    help="EMA of weights decay (0=disabled, 0.9999 recommended)")

# 2. 모델 생성 직후
ema = None
if args.ema_decay > 0:
    ema = ModelEMA(model, decay=args.ema_decay)
    print(f"  EMA enabled (decay={args.ema_decay})")

# 3. train_one_epoch 호출 시 ema 전달
train_loss, train_acc = train_one_epoch(
    model, train_loader, criterion, optimizer, device, epoch, args.epochs,
    scaler=scaler, use_mixup=args.use_mixup, ohem_ratio=args.ohem_ratio,
    ema=ema
)

# 4. 평가 시 EMA 모델 사용
eval_model = ema.module if ema is not None else model
val_loss, val_acc, val_recall, val_f1, val_metrics, ... = evaluate(
    eval_model, val_loader, ...
)

# 5. Test도 마찬가지
test_..., ... = evaluate(eval_model, test_loader, ...)

# 6. Best model 저장 시 EMA weights
save_target = ema.module.state_dict() if ema is not None else model.state_dict()
torch.save(save_target, str(log_dir / "best_model.pth"))
```

## 검증 실험 (Phase 3)

### Exp 3-1: EMA 단독 효과 (seed 1, v9 data)
- `--ema_decay 0.9999 --smooth_window 0 --patience 10 --min_epochs 10`
- smoothed median OFF → EMA만으로 val 1.0 spike 제거 확인
- 폴더: `logs/v9_ema_only_n700_s1`

### Exp 3-2: EMA + smoothed median 조합 (seed 1)
- `--ema_decay 0.9999` (smooth_window 3 median 기본값)
- 폴더: `logs/v9_ema_smooth_n700_s1`

### Exp 3-3: Ablation 비교 표
| Variant | smooth_window | ema_decay | seed 1 test FN+FP |
|---------|---------------|-----------|-------------------|
| baseline | 0 | 0 | 11 (기존 sharp) |
| smooth only | 3 | 0 | 2 (winning config) |
| ema only | 0 | 0.9999 | ? |
| smooth + ema | 3 | 0.9999 | ? |

### Exp 3-4: EMA 재현성 (5 seeds)
- EMA 단독이 효과 있으면 5 seeds 검증
- 폴더: `logs/v9_ema_n700_s{42,1,2,3,4}`

## Memory 업데이트 (Phase 3 완료 후)

- `project_research_overfitting.md` 에 EMA 실제 효과 숫자 업데이트
- `project_winning_config.md` 에 EMA 포함 새 winning config 반영
- `feedback_smoothed_val_selection.md` EMA와의 관계 명시
