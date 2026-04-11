"""NT sweep inference on saved models (no retraining)"""
import torch, json, os, sys
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
sys.path.insert(0, '.')
from train import ChartImageDataset, create_model, _compute_binary_confusion


def main():
    scen_df = pd.read_csv('data/scenarios.csv')
    test_df = scen_df[scen_df['split'] == 'test']
    all_classes = ['normal', 'mean_shift', 'standard_deviation', 'spike', 'drift', 'context']

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_ds = ChartImageDataset(Path('images'), test_df, all_classes, tfm, mode='binary')
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    device = torch.device('cuda')

    def load_and_infer(path):
        model = create_model(
            2, 'convnextv2_tiny.fcmae_ft_in22k_in1k', device,
            dropout=0.0,
        )
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        all_probs_nor = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                imgs = batch[0].to(device)
                lbls = batch[1]
                with torch.amp.autocast('cuda'):
                    logits = model(imgs)
                    probs = F.softmax(logits, dim=1)
                all_probs_nor.extend(probs[:, 0].cpu().tolist())
                all_labels.extend(lbls.tolist())
        return all_probs_nor, all_labels

    runs = []
    for sub in ['_ep10', '']:
        for s in [42, 1, 2, 3, 4]:
            d = f'logs/v9{sub}_n2800_s{s}'
            if not os.path.exists(f'{d}/best_model.pth'):
                continue
            runs.append(d)

    nts = [0.5, 0.9, 0.99, 0.999, 0.9999]
    header = ' '.join(f"{nt:>7}" for nt in nts)
    print(f"{'run':<42} {header}")
    print('-' * 95)
    for d in runs:
        probs_nor, labels = load_and_infer(f'{d}/best_model.pth')
        results = {}
        for nt in nts:
            preds = [0 if p > nt else 1 for p in probs_nor]
            cm = _compute_binary_confusion(preds, labels)
            results[nt] = (cm['fn'] + cm['fp'], cm['fn'], cm['fp'])
        row = ' '.join(f"{results[nt][0]:>3}({results[nt][1]}/{results[nt][2]})" for nt in nts)
        print(f"{d.replace('logs/v9', ''):<42} {row}")


if __name__ == '__main__':
    main()
