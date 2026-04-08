"""
학습 에이전트

ConvNeXtV2-Tiny (pretrained, file-based) + Focal Loss
- tqdm으로 batch/epoch 진행 실시간 표시
- epoch별 클래스별 성능 테이블 출력
- val recall 기준 best 저장 → best 갱신 시 test 평가
- epoch별 loss/성능 plot 저장
- best confusion matrix 저장
- 모델 + 하이퍼파라미터 + 조건 저장

Usage:
    python train.py
    python train.py --config config.yaml --epochs 50
"""

import argparse
import json
import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import timm
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.focal_loss import FocalLoss


# =============================================================================
# Worker init (module level — Windows multiprocessing pickle)
# =============================================================================
_GLOBAL_WORKER_SEED = 42


def _worker_init_fn(worker_id):
    """DataLoader worker별 seed 고정 (재현성)"""
    import random
    seed = _GLOBAL_WORKER_SEED + worker_id
    np.random.seed(seed)
    random.seed(seed)


# =============================================================================
# EMA of weights (Mean Teacher / ConvNeXt-V2 스타일)
# =============================================================================

class ModelEMA:
    """Exponential Moving Average of model weights with dynamic decay warmup.

    매 step 마다 ema_w = decay_t * ema_w + (1 - decay_t) * w
    decay_t = min(target_decay, (1 + step) / (10 + step))

    초기에는 낮은 decay (step=0 → 0.1) 로 EMA 가 빠르게 current 에 접근,
    점차 target decay (예: 0.999) 로 수렴 → spike 완화 효과 유지하면서
    "EMA 가 초기 pretrained 에 머물러 있는" 문제 해결.

    Reference:
    - Tarvainen & Valpola 2017 (Mean Teacher): arXiv:1703.01780
    - Izmailov et al. 2018 (SWA): arXiv:1803.05407
    - ConvNeXt-V2 official: decay 0.9999 (하지만 큰 dataset 전제)
    - timm.utils.ModelEmaV2: dynamic decay warmup formula

    작은 dataset (n=700, ~132 iter/ep, 20 ep = 2640 steps) 에서
    target 0.9999 는 너무 높음 (0.9999^2640 = 0.768, 76.8% init 유지).
    target 0.999 권장: 0.999^2640 = 0.071, 92.9% trained.
    """
    def __init__(self, model, decay=0.999):
        import copy
        self.target_decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        # Dynamic decay: ramp from 0.1 (step 0) to target_decay
        decay_t = min(self.target_decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        for ema_v, v in zip(self.module.state_dict().values(),
                            model.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(decay_t).add_(v.detach().to(ema_v.dtype),
                                          alpha=1.0 - decay_t)
            else:
                ema_v.copy_(v)


def _compute_binary_confusion(preds, labels):
    """Binary (0=normal, 1=abnormal) confusion: TN, FN, FP, TP"""
    import numpy as np
    p = np.asarray(preds)
    l = np.asarray(labels)
    tn = int(((p == 0) & (l == 0)).sum())
    fn = int(((p == 0) & (l == 1)).sum())
    fp = int(((p == 1) & (l == 0)).sum())
    tp = int(((p == 1) & (l == 1)).sum())
    return dict(tn=tn, fn=fn, fp=fp, tp=tp)


# =============================================================================
# Dataset
# =============================================================================

class ChartImageDataset(Dataset):

    def __init__(self, image_dir: Path, scenarios_df: pd.DataFrame,
                 classes: list, transform=None, mode="multiclass"):
        """
        mode:
          "multiclass" - 6클래스 (normal, mean_shift, std, spike, drift, context)
          "binary" - 2클래스 (0=normal, 1=abnormal)
          "anomaly_type" - 5클래스 (abnormal만, normal 제외)
        """
        self.transform = transform
        self.mode = mode
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.samples = []
        self.original_classes = []  # 원래 defect type 저장
        self.sample_weights = []  # label별 weight
        for _, row in scenarios_df.iterrows():
            cls = row["class"]
            split = row["split"]
            chart_id = row["chart_id"]
            img_path = image_dir / split / cls / f"{chart_id}.png"
            if not img_path.exists():
                continue

            if mode == "binary":
                label = 0 if cls == "normal" else 1
            elif mode == "anomaly_type":
                if cls == "normal":
                    continue  # normal 제외
                anomaly_classes = [c for c in classes if c != "normal"]
                label = anomaly_classes.index(cls)
            else:
                label = self.class_to_idx[cls]

            self.samples.append((str(img_path), label))
            self.original_classes.append(cls)
            self.sample_weights.append(1.0)  # 기본값, 나중에 set_label_weights로 설정

    def __len__(self):
        return len(self.samples)

    def set_label_weights(self, weight_map: dict):
        """label별 sample weight 설정. weight_map: {defect_type: weight}"""
        for i, cls in enumerate(self.original_classes):
            self.sample_weights[i] = weight_map.get(cls, 1.0)
        print(f"  Sample weights set: {weight_map}")

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, self.sample_weights[idx]

    def get_path(self, idx):
        return self.samples[idx][0]


# =============================================================================
# Model
# =============================================================================

def create_model(num_classes: int, model_name: str, device: torch.device,
                  dropout: float = 0.5):
    """timm 모델 생성. pretrained=False + weights/{model_name}.pth 로드.

    파일명은 HF model id 그대로:
        convnextv2_tiny.fcmae_ft_in22k_in1k → weights/convnextv2_tiny.fcmae_ft_in22k_in1k.pth
    파일 없으면 FileNotFoundError — 먼저 `python download.py` 실행할 것.
    """
    weights_path = f"weights/{model_name}.pth"
    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"가중치 파일 없음: {weights_path}\n"
            f"먼저 'python download.py' 실행하여 weights/ 폴더에 다운로드"
        )

    model = timm.create_model(model_name, pretrained=False)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    print(f"  가중치 로드: {weights_path}")

    # Classification head 교체 (timm 모델마다 head 구조 다름)
    if hasattr(model, 'head') and hasattr(model.head, 'fc'):
        in_features = model.head.fc.in_features
        model.head.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, 512),
            nn.ReLU(inplace=True), nn.Linear(512, num_classes),
        )
    elif hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, 512),
            nn.ReLU(inplace=True), nn.Linear(512, num_classes),
        )
    elif hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, 512),
            nn.ReLU(inplace=True), nn.Linear(512, num_classes),
        )
    else:
        in_features = model.get_classifier().in_features
        model.reset_classifier(0)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, 512),
            nn.ReLU(inplace=True), nn.Linear(512, num_classes),
        )

    print(f"  모델: {model_name} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    return model.to(device)


# =============================================================================
# Train / Eval
# =============================================================================

def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs,
                    scaler=None, use_mixup=False, ohem_ratio=0.0, amp_dtype=None, ema=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]",
                leave=False, ncols=100)
    for batch in pbar:
        if len(batch) == 3:
            images, labels, weights = batch
            weights = weights.to(device)
        else:
            images, labels = batch
            weights = None
        images, labels = images.to(device), labels.to(device)

        if use_mixup and np.random.random() < 0.5:
            images, labels_a, labels_b, lam = mixup_data(images, labels)
        else:
            labels_a, labels_b, lam = labels, labels, 1.0

        optimizer.zero_grad()

        def compute_loss(outputs):
            if ohem_ratio > 0:
                # sample별 loss → top-K
                import torch.nn.functional as F
                ce_a = F.cross_entropy(outputs, labels_a, reduction='none')
                ce_b = F.cross_entropy(outputs, labels_b, reduction='none')
                losses = lam * ce_a + (1 - lam) * ce_b
                k = max(1, int(len(losses) * ohem_ratio))
                topk_losses, _ = losses.topk(k)
                return topk_losses.mean()
            else:
                return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)

        if scaler is not None:
            # fp16 path with GradScaler
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(images)
                loss = compute_loss(outputs)
                if weights is not None and ohem_ratio == 0:
                    loss = (loss * weights).mean() if loss.dim() > 0 else loss
            scaler.scale(loss).backward()
            # gradient clipping (cliff fall 안전망)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        elif amp_dtype is not None and device.type == "cuda":
            # bf16 path — scaler 불필요 (range가 fp32와 동일)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs = model(images)
                loss = compute_loss(outputs)
                if weights is not None and ohem_ratio == 0:
                    loss = (loss * weights).mean() if loss.dim() > 0 else loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            # fp32 path
            outputs = model(images)
            loss = compute_loss(outputs)
            if weights is not None and ohem_ratio == 0:
                loss = (loss * weights).mean() if loss.dim() > 0 else loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # EMA update (after optimizer.step)
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels_a).sum().item()
        total += labels_a.size(0)

        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.3f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, classes, desc="Eval",
             normal_threshold=None, tta=False, amp_dtype=None):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    total = 0

    # eval은 default fp16 (속도). bf16 학습이면 bf16 eval.
    eval_dtype = amp_dtype if amp_dtype is not None else torch.float16
    use_amp_eval = device.type == "cuda"

    pbar = tqdm(loader, desc=desc, leave=False, ncols=100)
    for batch in pbar:
        images, labels = batch[0], batch[1]
        images, labels = images.to(device), labels.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp_eval, dtype=eval_dtype):
            if tta:
                # TTA: 5가지 변형 평균
                # 1. 원본
                probs_list = [F.softmax(model(images), dim=1)]
                # 2. brightness +10%
                probs_list.append(F.softmax(model((images * 1.10).clamp(-3, 3)), dim=1))
                # 3. brightness -10%
                probs_list.append(F.softmax(model((images * 0.90).clamp(-3, 3)), dim=1))
                # 4. contrast +10% (mean 중심으로 ±10%)
                mean = images.mean(dim=[2,3], keepdim=True)
                probs_list.append(F.softmax(model(((images - mean) * 1.10 + mean).clamp(-3, 3)), dim=1))
                # 5. contrast -10%
                probs_list.append(F.softmax(model(((images - mean) * 0.90 + mean).clamp(-3, 3)), dim=1))
                probs = torch.stack(probs_list).mean(dim=0)
                outputs = torch.log(probs + 1e-10)
                loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        total += labels.size(0)

        if normal_threshold is not None:
            probs = F.softmax(outputs, dim=1)
            normal_prob = probs[:, 0]
            is_normal = normal_prob > normal_threshold
            anomaly_preds = torch.argmax(probs[:, 1:], dim=1) + 1
            predicted = torch.where(is_normal, torch.zeros_like(anomaly_preds), anomaly_preds)
        else:
            _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 클래스별 성능
    class_metrics = {}
    for i, cls in enumerate(classes):
        mask = all_labels == i
        cnt = int(mask.sum())
        if cnt == 0:
            class_metrics[cls] = {"recall": 0.0, "precision": 0.0, "f1": 0.0, "count": 0}
            continue
        tp = int(((all_preds == i) & (all_labels == i)).sum())
        fp = int(((all_preds == i) & (all_labels != i)).sum())
        fn = int(((all_preds != i) & (all_labels == i)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        class_metrics[cls] = {"recall": rec, "precision": prec, "f1": f1, "count": cnt}

    avg_loss = total_loss / total
    accuracy = float((all_preds == all_labels).mean())
    avg_recall = float(np.mean([m["recall"] for m in class_metrics.values()]))
    avg_f1 = float(np.mean([m["f1"] for m in class_metrics.values()]))

    return avg_loss, accuracy, avg_recall, avg_f1, class_metrics, all_preds, all_labels


def print_class_table(metrics, title=""):
    """클래스별 성능 테이블 출력"""
    print(f"\n  {title}")
    print(f"  {'Class':>20s} | {'Recall':>7s} | {'Prec':>7s} | {'F1':>7s} | {'N':>4s}")
    print(f"  {'-'*20}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*4}")
    for cls, m in metrics.items():
        print(f"  {cls:>20s} | {m['recall']:7.3f} | {m['precision']:7.3f} | {m['f1']:7.3f} | {m['count']:4d}")
    avg_r = np.mean([m["recall"] for m in metrics.values()])
    avg_f = np.mean([m["f1"] for m in metrics.values()])
    print(f"  {'-'*20}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*4}")
    print(f"  {'AVERAGE':>20s} | {avg_r:7.3f} |         | {avg_f:7.3f} |")


def _confusion_matrix_np(y_true, y_pred, n_classes):
    """sklearn.metrics.confusion_matrix 대체 (numpy only).

    폐쇄망 Ubuntu 24 + numpy 2.x + 구버전 scipy/sklearn ABI 충돌 회피용.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    mat = np.zeros((n_classes, n_classes), dtype=np.int64)
    # bincount trick — O(N), no python loop
    idx = y_true * n_classes + y_pred
    counts = np.bincount(idx, minlength=n_classes * n_classes)
    return counts.reshape(n_classes, n_classes)


def save_confusion_matrix(labels, preds, classes, save_path):
    """Confusion matrix 저장"""
    mat = _confusion_matrix_np(labels, preds, len(classes))
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    short = [c[:8] for c in classes]
    sns.heatmap(mat, annot=True, fmt="d", cmap="Blues",
                xticklabels=short, yticklabels=short, ax=ax)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title("Confusion Matrix (Best Model)", fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_predictions(dataset, preds, labels, classes, save_dir, correct_cap=100):
    """4분면 예측 저장 (display 이미지). TN/TP는 correct_cap까지만.

    Binary mode 가정 (classes = ["normal", "abnormal"]):
      - tn_normal/    : pred=normal, true=normal      (정상 통과, 최대 cap개)
      - fn_abnormal/  : pred=normal, true=abnormal    (놓친 불량, 전부) ★
      - fp_normal/    : pred=abnormal, true=normal    (false alarm, 전부) ★
      - tp_abnormal/  : pred=abnormal, true=abnormal  (잘 잡음, 최대 cap개)

    Multiclass도 지원: 같은 패턴으로 클래스별 4분면.
    """
    import shutil
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    counts = {"tn": 0, "fn": 0, "fp": 0, "tp": 0}
    saved_correct = {}  # (label) -> 누적된 correct 개수 (cap 체크용)

    for i in range(len(preds)):
        true_cls = classes[labels[i]]
        pred_cls = classes[preds[i]]
        is_correct = (preds[i] == labels[i])

        # 카테고리 결정
        if is_correct:
            # TN (true=normal) 또는 TP (true=abnormal/anything else)
            category = "tn" if true_cls == "normal" else "tp"
            # cap 체크
            saved_correct.setdefault(category, 0)
            if saved_correct[category] >= correct_cap:
                counts[category] += 1
                continue
            saved_correct[category] += 1
            dest_name = f"{category}_{true_cls}"
        else:
            # FN (true=abnormal, pred=normal) 또는 FP (true=normal, pred=abnormal)
            if true_cls == "normal":
                category = "fp"
                dest_name = f"fp_{true_cls}"
            else:
                category = "fn"
                dest_name = f"fn_{true_cls}"
        counts[category] += 1

        src_path = Path(dataset.get_path(i))
        fname = src_path.name
        display_path = Path(str(src_path).replace("images", "display", 1))
        copy_src = display_path if display_path.exists() else src_path

        dest_dir = save_dir / dest_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(copy_src, dest_dir / f"pred_{pred_cls}_{fname}")

    saved_tn = min(counts["tn"], correct_cap)
    saved_tp = min(counts["tp"], correct_cap)
    print(f"  Predictions saved: TN={saved_tn}/{counts['tn']} (cap {correct_cap}), "
          f"FN={counts['fn']}, FP={counts['fp']}, "
          f"TP={saved_tp}/{counts['tp']} (cap {correct_cap})")
    print(f"  -> {save_dir}/")


def save_training_plots(history, save_dir):
    """epoch별 loss/recall+f1/lr plot 저장"""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_recall = [h["val_recall"] for h in history]
    val_f1 = [h["val_f1"] for h in history]
    lrs = [h.get("lr", 0) for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(epochs, train_loss, "b-", label="Train")
    axes[0].plot(epochs, val_loss, "r-", label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Recall & F1
    axes[1].plot(epochs, val_recall, "r-", label="Val Recall")
    axes[1].plot(epochs, val_f1, "b-", label="Val F1")
    axes[1].set_title("Recall & F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning rate
    axes[2].plot(epochs, lrs, "m-", label="LR")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_yscale("log")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_dir / "training_curves.png", dpi=150)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_backbone", type=float, default=2e-5,
                        help="Backbone peak LR. ConvNeXtV2-Tiny: 2e-5 (5e-5는 spike 유발). 다른 backbone도 2e-5부터 시작.")
    parser.add_argument("--lr_head", type=float, default=2e-4,
                        help="Head peak LR. backbone의 10x 유지.")
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--normal_threshold", type=float, default=0.5,
                        help="Inference threshold. 0.5=standard, 0.7=약간 conservative. 극한값(0.999+) 금지 — test-peeking 이고 production 에서 fragile.")
    parser.add_argument("--model_name", type=str, default="convnextv2_tiny.fcmae_ft_in22k_in1k")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "step", "plateau"])
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--step_gamma", type=float, default=0.5)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--use_mixup", action="store_true", default=False)
    parser.add_argument("--mode", type=str, default="binary",
                        choices=["multiclass", "binary", "anomaly_type"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--ema_decay", type=float, default=0.0,
                        help="EMA of weights decay. 0=disabled. ConvNeXt-V2 official: 0.9999. 추천: 0.999~0.9999")
    parser.add_argument("--max_per_class", type=int, default=0,
                        help="학습 데이터 클래스당 최대 수 (0=전체)")
    parser.add_argument("--normal_ratio", type=int, default=0,
                        help="Binary mode: normal 학습 샘플 수 (0=전체, abnormal은 고정)")
    parser.add_argument("--ohem_ratio", type=float, default=0.0,
                        help="OHEM ratio (0=off, 0.75=top75%, 0.5=top50%)")
    parser.add_argument("--focal_gamma", type=float, default=0.0,
                        help="Focal Loss gamma (0 = 사실상 CrossEntropy)")
    parser.add_argument("--abnormal_weight", type=float, default=1.0,
                        help="Binary mode: abnormal class weight 배수 (1.0=균등, 0=inverse freq 자동)")
    parser.add_argument("--label_weights", type=str, default="",
                        help="Multiclass label별 weight (예: normal=0.5,spike=3.0,context=2.0)")
    parser.add_argument("--min_epochs", type=int, default=1,
                        help="최소 학습 epoch — best 갱신 허용 기준. 1=ep 1부터 best update + test_history 기록")
    parser.add_argument("--avg_last_n", type=int, default=0,
                        help="[deprecated] post-hoc weight 평균 (0=비활성)")
    parser.add_argument("--smooth_window", type=int, default=3,
                        help="best 기준 val_f1을 최근 N epoch 통계로 smooth (0=비활성)")
    parser.add_argument("--smooth_method", type=str, default="median",
                        choices=["mean", "median"],
                        help="smooth_window 통계 방법 (median: spike에 robust)")
    parser.add_argument("--eval_test_every_epoch", action="store_true",
                        help="매 epoch 끝에 test 평가 + history에 기록")
    parser.add_argument("--seed", type=int, default=42,
                        help="학습 random seed (재현성)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker 수 (Windows: 0~4, Linux 서버: 8~16 권장)")
    parser.add_argument("--precision", type=str, default="fp16",
                        choices=["fp16", "bf16", "fp32"],
                        help="학습 정밀도 (H100/H200: bf16 권장 — overflow 없음, GradScaler 불필요)")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile 활성화 (H100/H200에서 20~50%% 가속)")
    parser.add_argument("--prefetch_factor", type=int, default=4,
                        help="DataLoader prefetch_factor (서버: 8~16)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 모든 random seed 고정 (재현성)
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    # OLD v9_noise_sparse 스타일 — 빠른 non-deterministic 경로 (benchmark=True)
    # 재현성 약간 손해 보되 학습 dynamics 안정성 복원
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print(f"  Random seed: {args.seed} (cudnn.benchmark=True, non-deterministic)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Precision 결정 — bf16은 GradScaler 불필요 (H100/H200 권장)
    # use_amp 호환: fp16이 default. --precision bf16/fp32로 override.
    if args.precision == "bf16" and device.type == "cuda":
        amp_dtype = torch.bfloat16
        scaler = None  # bf16은 scaler 불필요
    elif args.precision == "fp16" and args.use_amp and device.type == "cuda":
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda")
    else:
        amp_dtype = None  # fp32
        scaler = None

    print(f"\n{'='*60}")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        # H100/H200 같은 SM 9.0은 TF32 + bf16이 가장 빠름
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"  Precision: {args.precision} (amp_dtype={amp_dtype}, scaler={'on' if scaler else 'off'})")
    print(f"  Compile: {args.compile}")
    print(f"{'='*60}")

    all_classes = config["dataset"]["classes"]
    if args.mode == "binary":
        classes = ["normal", "abnormal"]
    elif args.mode == "anomaly_type":
        classes = [c for c in all_classes if c != "normal"]
    else:
        classes = all_classes
    num_classes = len(classes)
    image_dir = Path(config["output"]["image_dir"])
    data_dir = Path(config["output"]["data_dir"])

    # 하이퍼파라미터
    hparams = {
        "model": "ConvNeXtV2-Tiny",
        "pretrained": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr_backbone": args.lr_backbone,
        "lr_head": args.lr_head,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "normal_threshold": args.normal_threshold,
        "warmup_epochs": args.warmup_epochs,
        "loss": f"FocalLoss(gamma={args.focal_gamma})",
        "optimizer": "AdamW",
        "scheduler": "Warmup(5ep) + CosineAnnealing",
        "num_classes": num_classes,
        "classes": classes,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "mode": args.mode,
        "amp": args.use_amp,
        "mixup": args.use_mixup,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "dropout": args.dropout,
        "focal_gamma": args.focal_gamma,
        "smooth_window": args.smooth_window,
        "smooth_method": args.smooth_method,
        "min_epochs": args.min_epochs,
        "precision": args.precision,
        "compile": args.compile,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
    }
    print("\n  Hyperparameters:")
    for k, v in hparams.items():
        print(f"    {k}: {v}")

    # 데이터
    sc_df = pd.read_csv(data_dir / "scenarios.csv")
    train_df = sc_df[sc_df["split"] == "train"]
    val_df = sc_df[sc_df["split"] == "val"]
    test_df = sc_df[sc_df["split"] == "test"]

    # 클래스당 최대 수 제한
    if args.max_per_class > 0:
        train_df = train_df.groupby("class").apply(
            lambda x: x.sample(n=min(len(x), args.max_per_class), random_state=42)
        ).reset_index(drop=True)

    # Normal 샘플 수 조절 (abnormal은 유지)
    if args.normal_ratio > 0 and args.mode == "binary":
        normal_df = train_df[train_df["class"] == "normal"]
        abnormal_df = train_df[train_df["class"] != "normal"]
        n_target = args.normal_ratio
        if n_target < len(normal_df):
            normal_df = normal_df.sample(n=n_target, random_state=42)
        elif n_target > len(normal_df):
            # 부족하면 복제 (oversample)
            extra = normal_df.sample(n=n_target - len(normal_df), random_state=42, replace=True)
            normal_df = pd.concat([normal_df, extra])
        train_df = pd.concat([normal_df, abnormal_df]).reset_index(drop=True)
        print(f"  Normal ratio: {len(normal_df)} normal / {len(abnormal_df)} abnormal")

    print(f"\n  Dataset: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # 모델별 정규화 자동 추출 (timm data config)
    # tmp 모델 생성 (pretrained X, 매우 빠름) → mean/std 추출
    _tmp_model = timm.create_model(args.model_name, pretrained=False)
    _data_cfg = timm.data.resolve_model_data_config(_tmp_model)
    input_mean = list(_data_cfg.get("mean", (0.485, 0.456, 0.406)))
    input_std = list(_data_cfg.get("std", (0.229, 0.224, 0.225)))
    input_size = _data_cfg.get("input_size", (3, 224, 224))
    img_hw = (input_size[1], input_size[2])
    del _tmp_model
    print(f"  Input: size={img_hw}, mean={input_mean}, std={input_std}")

    train_transform = transforms.Compose([
        transforms.Resize(img_hw),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        transforms.ToTensor(),
        transforms.Normalize(input_mean, input_std),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(img_hw),
        transforms.ToTensor(),
        transforms.Normalize(input_mean, input_std),
    ])

    train_ds = ChartImageDataset(image_dir, train_df, all_classes, train_transform, mode=args.mode)
    val_ds = ChartImageDataset(image_dir, val_df, all_classes, val_transform, mode=args.mode)
    test_ds = ChartImageDataset(image_dir, test_df, all_classes, val_transform, mode=args.mode)
    print(f"  Images: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    num_workers = args.num_workers
    # 전역 변수로 seed 전달 (multiprocessing pickle 위해)
    global _GLOBAL_WORKER_SEED
    _GLOBAL_WORKER_SEED = args.seed

    common = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **common)
    print(f"  DataLoader: num_workers={num_workers}, pin_memory=True, prefetch_factor=4, persistent_workers={num_workers > 0}")

    # 모델 — weights/<short>.pth 자동 매핑 (없으면 FileNotFoundError)
    model = create_model(num_classes, args.model_name, device, dropout=args.dropout)
    params_M = sum(p.numel() for p in model.parameters()) / 1e6

    # torch.compile (H100/H200 권장: 20~50% 가속)
    if args.compile:
        if hasattr(torch, "compile") and device.type == "cuda":
            print(f"  torch.compile: enabled (mode=max-autotune)")
            try:
                model = torch.compile(model, mode="max-autotune")
            except Exception as e:
                print(f"  torch.compile failed, falling back: {e}")
        else:
            print(f"  torch.compile requested but unsupported (cpu or old torch)")

    # EMA of weights (Mean Teacher / ConvNeXt-V2 스타일)
    ema = None
    if args.ema_decay > 0:
        ema = ModelEMA(model, decay=args.ema_decay)
        print(f"  EMA enabled: decay={args.ema_decay}")
    else:
        print(f"  EMA disabled (ema_decay=0)")

    # Freeze backbone (선택)
    if args.freeze_backbone_epochs > 0:
        for n, p in model.named_parameters():
            if "head" not in n and "classifier" not in n:
                p.requires_grad = False
        print(f"  Backbone frozen for {args.freeze_backbone_epochs} epochs")

    # Label별 sample weight 생성 (binary에서도 원래 defect type별 weight 적용)
    sample_weights = None
    if args.label_weights and args.mode == "binary":
        lw_map = {}
        for pair in args.label_weights.split(","):
            k, v = pair.strip().split("=")
            lw_map[k.strip()] = float(v)
        # train dataset의 각 sample에 원래 defect type 기반 weight 부여
        weights = []
        for orig_cls in train_ds.original_classes:
            weights.append(lw_map.get(orig_cls, 1.0))
        sample_weights = torch.FloatTensor(weights)
        print(f"  Label weights (per sample): {lw_map}")

    # Focal Loss (+ label smoothing)
    if args.mode == "binary":
        n_normal = len(train_df[train_df["class"] == "normal"])
        n_abnormal = len(train_df[train_df["class"] != "normal"])
        if args.abnormal_weight > 0 and not args.label_weights:
            # 수동 지정: normal=1.0, abnormal=지정값
            alpha = [1.0, args.abnormal_weight]
        else:
            alpha = [n_abnormal / (n_normal + n_abnormal) * 2,
                     n_normal / (n_normal + n_abnormal) * 2]
        print(f"  Binary class weights: normal={alpha[0]:.2f}, abnormal={alpha[1]:.2f}")
    elif args.mode == "anomaly_type":
        anomaly_df = train_df[train_df["class"] != "normal"]
        anomaly_classes = [c for c in all_classes if c != "normal"]
        counts = anomaly_df["class"].value_counts()
        alpha = [1.0 / counts.get(c, 1) for c in anomaly_classes]
        alpha_sum = sum(alpha)
        alpha = [a / alpha_sum * len(anomaly_classes) for a in alpha]
    else:
        class_counts = train_df["class"].value_counts()
        alpha = [1.0 / class_counts.get(c, 1) for c in classes]
        alpha_sum = sum(alpha)
        alpha = [a / alpha_sum * num_classes for a in alpha]

        # label별 weight 수동 지정 (예: --label_weights "normal=0.5,spike=3.0")
        if args.label_weights:
            for pair in args.label_weights.split(","):
                k, v = pair.strip().split("=")
                if k in classes:
                    idx = classes.index(k)
                    alpha[idx] = float(v)
            print(f"  Label weights: {dict(zip(classes, [f'{a:.2f}' for a in alpha]))}")
    if args.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(alpha, dtype=torch.float32).to(device),
            label_smoothing=args.label_smoothing
        )
        print(f"  Loss: CrossEntropy + LabelSmoothing({args.label_smoothing})")
    else:
        criterion = FocalLoss(alpha=alpha, gamma=args.focal_gamma).to(device)
        print(f"  Loss: FocalLoss(gamma={args.focal_gamma})")

    # Optimizer
    backbone_params = [p for n, p in model.named_parameters() if "head" not in n]
    head_params = [p for n, p in model.named_parameters() if "head" in n]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=args.weight_decay)

    # Scheduler
    # warmup start_factor 0.05 — 매우 낮은 시작값으로 gradient spike 방지
    # 기존 0.1은 초기 LR이 너무 높아 seed 4 같은 경우에 ep 4-8에서 spike 발생
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.05, total_iters=args.warmup_epochs
    )
    if args.scheduler == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6
        )
    elif args.scheduler == "step":
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.step_gamma
        )
    elif args.scheduler == "plateau":
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.step_gamma, patience=3
        )

    if args.scheduler != "plateau":
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler],
            milestones=[args.warmup_epochs]
        )
    else:
        scheduler = warmup_scheduler  # plateau는 별도 처리
    print(f"  Scheduler: warmup({args.warmup_epochs}ep) + {args.scheduler}")

    # 로그
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = log_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    history = []

    best_val_recall = 0.0
    best_epoch = 0
    best_test_metrics = None
    best_test_recall = 0.0
    best_test_f1 = 0.0
    patience_counter = 0
    # test_history: 매 best update (NEW BEST 또는 TIE save) 시마다 test 측정값 누적
    test_history = []

    print(f"\n{'='*60}")
    print(f"  학습 시작 ({args.epochs} epochs)")
    print(f"{'='*60}\n")

    train_start_time = time.time()
    epoch_times = []

    # Model averaging (post-hoc, deprecated): 마지막 N epoch state 보관
    from collections import deque
    avg_buffer = deque(maxlen=args.avg_last_n) if args.avg_last_n > 0 else None

    # Smoothed val_f1 (online): 최근 N epoch val_f1 값 보관
    val_f1_window = deque(maxlen=args.smooth_window) if args.smooth_window > 0 else None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Backbone unfreeze
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs + 1:
            for n, p in model.named_parameters():
                p.requires_grad = True
            print(f"\n  * Backbone unfrozen at epoch {epoch}")

        # Train (EMA 업데이트 포함)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs,
            scaler=scaler, use_mixup=args.use_mixup, ohem_ratio=args.ohem_ratio,
            amp_dtype=amp_dtype, ema=ema,
        )

        # 평가 모델: EMA 있으면 EMA, 없으면 원본
        eval_model = ema.module if ema is not None else model

        # Val
        val_loss, val_acc, val_recall, val_f1, val_metrics, val_preds, val_labels = evaluate(
            eval_model, val_loader, criterion, device, classes,
            desc=f"Epoch {epoch}/{args.epochs} [Val]",
            amp_dtype=amp_dtype,
        )

        # Test (옵션: 매 epoch 평가)
        ep_test_metrics = None
        if args.eval_test_every_epoch:
            _, _, ep_test_recall, ep_test_f1, ep_test_metrics, _, _ = evaluate(
                eval_model, test_loader, criterion, device, classes,
                desc=f"Epoch {epoch}/{args.epochs} [TestEpoch]",
                amp_dtype=amp_dtype,
            )

        if args.scheduler == "plateau" and epoch > args.warmup_epochs:
            main_scheduler.step(val_recall)
        else:
            scheduler.step()
        elapsed = time.time() - t0

        # 로그 기록
        epoch_log = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_recall": round(val_recall, 4),
            "val_f1": round(val_f1, 4),
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed": round(elapsed, 1),
        }
        if ep_test_metrics is not None:
            ep_log_extras = {
                "test_f1": round(ep_test_f1, 4),
                "test_recall": round(ep_test_recall, 4),
                "test_nor_R": round(ep_test_metrics["normal"]["recall"], 4),
                "test_abn_R": round(ep_test_metrics["abnormal"]["recall"], 4),
                "val_nor_R": round(val_metrics["normal"]["recall"], 4),
                "val_abn_R": round(val_metrics["abnormal"]["recall"], 4),
            }
            epoch_log.update(ep_log_extras)
        history.append(epoch_log)

        # 매 epoch마다 history.json + training_curves.png 갱신
        with open(log_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        try:
            save_training_plots(history, log_dir)
        except Exception:
            pass

        # Epoch 요약
        print(f"\n{'─'*60}")
        print(f"  Epoch {epoch}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  Train: loss={train_loss:.4f}  acc={train_acc:.3f}")
        print(f"  Val:   loss={val_loss:.4f}  acc={val_acc:.3f}  recall={val_recall:.3f}  f1={val_f1:.3f}")

        # 클래스별 성능 테이블
        print_class_table(val_metrics, title="Val Class Performance")

        # Best model 선택 기준: val F1 score (smooth_window 적용 시 N epoch 통계)
        # 동점이면 최신 epoch으로 업데이트 (>=)
        # min_epochs 이전에는 best 갱신하지 않음 (warmup 보장)
        if val_f1_window is not None:
            val_f1_window.append(val_f1)
            if args.smooth_method == "median":
                vals = sorted(val_f1_window)
                n = len(vals)
                val_target_recall = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2
            else:
                val_target_recall = sum(val_f1_window) / len(val_f1_window)
        else:
            val_target_recall = val_f1

        # Best 선택 로직 (TIE-UPDATE, train_tie.py 2026-04-09):
        #   - strict improvement (>): 모델 저장 + test 평가 + patience reset
        #   - tie (==):               모델 저장 + test 평가 + patience 증가 (last-tie wins)
        #   - decrease (<):           patience 증가만
        # min_epochs 이전에는 best 갱신 안 함
        # Rationale: val_f1 포화 plateau 에서 strict > 만 쓰면 첫 도달 epoch 에 freeze.
        # Tie 에서도 업데이트하여 더 수렴한 weights 선택하되, patience counter 는 누적되어
        # 5 연속 non-strict (tie or decrease) 발생 시 early stop.
        is_candidate = epoch >= args.min_epochs and val_target_recall >= best_val_recall
        is_strict_improvement = is_candidate and val_target_recall > best_val_recall

        if is_candidate:
            if is_strict_improvement:
                patience_counter = 0
            else:
                patience_counter += 1  # tie: 업데이트하되 counter 누적
            best_val_recall = val_target_recall
            best_epoch = epoch
            # best_model.pth 저장: EMA 있으면 EMA weights, 없으면 raw
            save_state = ema.module.state_dict() if ema is not None else model.state_dict()
            torch.save(save_state, str(log_dir / "best_model.pth"))
            tag = "NEW BEST" if is_strict_improvement else f"TIE update (stag {patience_counter}/{args.patience})"
            print(f"\n  * {tag} (target={val_target_recall:.4f}) -> model saved{' (EMA)' if ema is not None else ''}")

            # Best 시 test 평가 (eval_model 사용 — EMA 있으면 EMA)
            test_loss, test_acc, test_recall, test_f1, test_metrics, test_preds, test_labels = evaluate(
                eval_model, test_loader, criterion, device, classes,
                desc=f"Epoch {epoch}/{args.epochs} [Test]",
                amp_dtype=amp_dtype,
            )
            best_test_metrics = test_metrics
            best_test_recall = test_recall
            best_test_f1 = test_f1
            print_class_table(test_metrics, title="Test Class Performance (Best)")

            # test_history 누적 — FN/FP/TN/TP + 기본 메트릭
            tag_short = "NEW_BEST" if is_strict_improvement else "TIE"
            cm = _compute_binary_confusion(test_preds, test_labels) if args.mode == "binary" else {}
            test_entry = {
                "epoch": epoch,
                "event": tag_short,
                "patience": patience_counter,
                "val_f1": round(val_f1, 4),
                "val_recall": round(val_recall, 4),
                "val_target_smoothed": round(val_target_recall, 4),
                "test_f1": round(test_f1, 4),
                "test_recall": round(test_recall, 4),
                "test_nor_R": round(test_metrics["normal"]["recall"], 4),
                "test_abn_R": round(test_metrics["abnormal"]["recall"], 4),
                **cm,  # tn, fn, fp, tp (binary only)
            }
            test_history.append(test_entry)
            if cm:
                print(f"    test FN={cm['fn']} FP={cm['fp']} TN={cm['tn']} TP={cm['tp']}  test_f1={test_f1:.4f}")

            # 오분류 이미지 저장
            save_predictions(test_ds, test_preds, test_labels, classes, predictions_dir, correct_cap=100)

            # Confusion matrix 저장
            save_confusion_matrix(test_labels, test_preds, classes, log_dir / "confusion_matrix.png")

            # Normal threshold 다중 평가
            nt_thresholds = [0.5, 0.9, 0.99, 0.999, 0.9999]
            nt_results = {}
            best_nt = args.normal_threshold

            print(f"\n  Normal Threshold 평가:")
            for nt in nt_thresholds:
                _, nt_acc, nt_recall, nt_f1, nt_metrics_t, nt_preds_t, nt_labels_t = evaluate(
                    eval_model, test_loader, criterion, device, classes,
                    desc=f"NT={nt}", normal_threshold=nt, amp_dtype=amp_dtype,
                )
                nt_results[str(nt)] = {
                    "acc": round(nt_acc, 4), "recall": round(nt_recall, 4),
                    "f1": round(nt_f1, 4), "metrics": nt_metrics_t,
                }
                marker = " ★" if nt == best_nt else ""
                print(f"    NT={nt}: acc={nt_acc:.3f} recall={nt_recall:.3f} f1={nt_f1:.3f}{marker}")

                if nt == best_nt:
                    save_confusion_matrix(nt_labels_t, nt_preds_t, classes, log_dir / "confusion_matrix_nt.png")

            # Best 조건 저장
            best_info = {
                "epoch": best_epoch,
                "val_recall": round(best_val_recall, 4),
                "val_f1": round(val_f1, 4),
                "val_acc": round(val_acc, 4),
                "test_recall": round(test_recall, 4),
                "test_f1": round(test_f1, 4),
                "test_acc": round(test_acc, 4),
                "test_metrics": test_metrics,
                "normal_threshold_results": nt_results,
                "test_history": test_history,
                "hparams": hparams,
            }
            with open(log_dir / "best_info.json", "w", encoding="utf-8") as f:
                json.dump(best_info, f, indent=2, ensure_ascii=False)
            # test_history 별도 파일로도 저장 (가독성)
            with open(log_dir / "test_history.json", "w", encoding="utf-8") as f:
                json.dump(test_history, f, indent=2, ensure_ascii=False)
        else:
            # Decrease (smoothed < best) or pre-min_epochs: patience 증가
            patience_counter += 1
            print(f"\n  No improvement ({patience_counter}/{args.patience})")

        # epoch 시간 누적
        epoch_times.append(time.time() - t0)

        # Model averaging: 매 epoch 끝에 state_dict snapshot 저장
        if avg_buffer is not None:
            import copy as _copy
            avg_buffer.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

        # Early stopping (최소 epoch 이후에만)
        # smooth_window 활성: smoothed val_f1 기준으로 patience 누적 (이미 위 best 갱신 로직에서 처리됨)
        # avg_last_n (deprecated) 활성 시: 비활성화
        if args.avg_last_n == 0 and patience_counter >= args.patience and epoch >= args.min_epochs:
            print(f"\n  ! Early stopping at epoch {epoch}")
            break

    # 전체 학습 시간
    total_train_time = time.time() - train_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0

    # ===== Model Averaging (last N epochs) =====
    # 마지막 N epoch weight 평균하여 best 대체
    if avg_buffer is not None and len(avg_buffer) > 0:
        n_avg = len(avg_buffer)
        print(f"\n{'='*60}")
        print(f"  Model Averaging: 마지막 {n_avg} epoch 평균")
        print(f"{'='*60}")

        # 평균 state_dict 계산
        avg_state = {}
        keys = list(avg_buffer[0].keys())
        for k in keys:
            stacked = torch.stack([s[k].float() for s in avg_buffer])
            avg_state[k] = stacked.mean(dim=0).to(avg_buffer[0][k].dtype)

        # averaged 모델로 평가
        model.load_state_dict(avg_state)

        # val
        val_loss_a, val_acc_a, val_recall_a, val_f1_a, val_metrics_a, _, _ = evaluate(
            model, val_loader, criterion, device, classes, desc="Avg [Val]",
            amp_dtype=amp_dtype,
        )
        # test
        test_loss_a, test_acc_a, test_recall_a, test_f1_a, test_metrics_a, test_preds_a, test_labels_a = evaluate(
            model, test_loader, criterion, device, classes, desc="Avg [Test]",
            amp_dtype=amp_dtype,
        )

        print(f"\n  Averaged val:  f1={val_f1_a:.4f}  recall={val_recall_a:.4f}")
        print(f"  Averaged test: f1={test_f1_a:.4f}  recall={test_recall_a:.4f}")
        print_class_table(test_metrics_a, title="Averaged Test Performance")

        # averaged 모델 저장 → best_model.pth 대체
        torch.save({k: v.to(device) for k, v in avg_state.items()},
                   str(log_dir / "best_model.pth"))
        # confusion matrix 갱신
        save_confusion_matrix(test_labels_a, test_preds_a, classes,
                              log_dir / "confusion_matrix.png")
        # predictions 갱신
        save_predictions(test_ds, test_preds_a, test_labels_a, classes,
                         predictions_dir, correct_cap=100)

        # NT sweep on averaged
        nt_results_a = {}
        for nt in [0.5, 0.6, 0.7, 0.8, 0.9]:
            _, nt_acc, nt_recall, nt_f1, nt_metrics_t, nt_preds_t, nt_labels_t = evaluate(
                model, test_loader, criterion, device, classes,
                desc=f"AvgNT={nt}", normal_threshold=nt, amp_dtype=amp_dtype,
            )
            nt_results_a[str(nt)] = {
                "acc": round(nt_acc, 4), "recall": round(nt_recall, 4),
                "f1": round(nt_f1, 4), "metrics": nt_metrics_t,
            }
            if nt == args.normal_threshold:
                save_confusion_matrix(nt_labels_t, nt_preds_t, classes,
                                      log_dir / "confusion_matrix_nt.png")

        # best metrics를 averaged로 교체
        best_test_metrics = test_metrics_a
        best_test_recall = test_recall_a
        best_test_f1 = test_f1_a
        best_val_recall = val_f1_a
        best_epoch = f"avg_last_{n_avg}"  # 표시용
        # nt_results 위해 변수 보관
        avg_nt_results = nt_results_a
    else:
        avg_nt_results = None

    # Training plots
    save_training_plots(history, log_dir)

    # best_info.json 갱신/생성 (timing + averaging 정보)
    info_path = log_dir / "best_info.json"
    if info_path.exists():
        with open(info_path, encoding="utf-8") as f:
            bi = json.load(f)
    else:
        # NEW BEST 없었으면 새로 생성
        bi = {"hparams": hparams}

    # averaging 적용 시 상위 메트릭 averaged 값으로 교체
    if avg_buffer is not None and len(avg_buffer) > 0:
        bi["epoch"] = best_epoch
        bi["val_f1"] = round(best_val_recall, 4)
        bi["val_recall"] = round(best_val_recall, 4)
        bi["val_acc"] = round(val_acc_a, 4)
        bi["test_f1"] = round(best_test_f1, 4)
        bi["test_recall"] = round(best_test_recall, 4)
        bi["test_acc"] = round(test_acc_a, 4)
        bi["test_metrics"] = best_test_metrics
        bi["normal_threshold_results"] = avg_nt_results
        bi["averaging"] = {"enabled": True, "last_n": len(avg_buffer)}

    bi["timing"] = {
        "total_time_sec": round(total_train_time, 2),
        "total_time_min": round(total_train_time / 60, 2),
        "avg_epoch_sec": round(avg_epoch_time, 2),
        "num_epochs_run": len(epoch_times),
    }
    bi["params_M"] = round(params_M, 2)
    bi["input_size"] = list(input_size)
    bi["normalize"] = {"mean": input_mean, "std": input_std}
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(bi, f, indent=2, ensure_ascii=False)

    # 최종 요약
    print(f"\n{'='*60}")
    print(f"  학습 완료")
    print(f"{'='*60}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Val:  recall={best_val_recall:.4f}")
    if best_test_metrics:
        print(f"  Test: recall={best_test_recall:.4f}  f1={best_test_f1:.4f}")
        print_class_table(best_test_metrics, title="Final Test Performance (Best Model)")
    print(f"  Time: total={total_train_time/60:.1f}min, avg_epoch={avg_epoch_time:.1f}s")
    print(f"  Params: {params_M:.2f}M")
    print(f"\n  저장:")
    print(f"    모델: {log_dir}/best_model.pth")
    print(f"    로그: {log_dir}/best_info.json")
    print(f"    곡선: {log_dir}/training_curves.png")
    print(f"    예측: {predictions_dir}/ (TN/TP cap 100)")
    print(f"    CM:   {log_dir}/confusion_matrix.png")

    # 전체 히스토리 저장
    with open(log_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
