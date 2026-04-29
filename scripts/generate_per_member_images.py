"""Design A: Per-member image rendering.

실제 fab inference 방식 — 1 chart 에서 fleet 의 각 member 를 한 번씩 highlighted member로
이미지 N 장 생성 (N = fleet size).

Labels:
  - Non-anomaly highlighted member → 'normal'
  - Anomaly injected highlighted member → original defect class

Output:
  - images_per_member_{suffix}/train|val|test/{class}/ch_{id}_{member}.png
  - display_per_member_{suffix}/train|val|test/{class}/ch_{id}_{member}.png
  - data_per_member_{suffix}/scenarios.csv  — expanded (chart_id, highlighted_member, class) rows

Usage:
  python scripts/generate_per_member_images.py --config dataset.yaml --suffix vd080 --workers 6
"""
import argparse
import os
import sys
import pickle
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.image_renderer import ImageRenderer
from src.data.schema import highlighted_member as read_highlighted_member
from src.data.schema import legend_axis as read_legend_axis
from src.data.schema import members as read_members
from src.data.schema import target as read_target


_worker_cache = {}


def _init_worker(ts_pickle_path, config, x_col, images_dir, display_dir):
    with open(ts_pickle_path, 'rb') as f:
        _worker_cache['ts_grouped'] = pickle.load(f)
    _worker_cache['renderer'] = ImageRenderer(config)
    _worker_cache['images_dir'] = Path(images_dir)
    _worker_cache['display_dir'] = Path(display_dir)
    _worker_cache['x_col'] = x_col
    img_cfg = config.get('image', {})
    _worker_cache['title_columns'] = img_cfg.get('title_columns', ['device', 'step', 'item'])
    _worker_cache['x_label'] = img_cfg.get('x_label') or x_col
    _worker_cache['y_label'] = img_cfg.get('y_label', 'Measurement Value (nm)')


def _render_per_member(row_dict):
    """1 chart → len(members) images. Returns per-image metadata rows."""
    ts_grouped = _worker_cache['ts_grouped']
    renderer = _worker_cache['renderer']
    images_dir = _worker_cache['images_dir']
    display_dir = _worker_cache['display_dir']
    x_col = _worker_cache['x_col']
    title_columns = _worker_cache['title_columns']
    x_label = _worker_cache['x_label']
    y_label = _worker_cache['y_label']

    sid = row_dict['chart_id']
    chart_class = row_dict['class']
    split = row_dict['split']
    anomaly_member = read_highlighted_member(row_dict)
    members = read_members(row_dict)
    legend_axis = read_legend_axis(row_dict)
    if not anomaly_member or not members or not legend_axis:
        return []

    raw_ds = row_dict.get('defect_start_idx', None)
    defect_start = None
    if raw_ds is not None:
        try:
            if isinstance(raw_ds, (int, float)):
                if not (np.isnan(raw_ds) or raw_ds < 0):
                    defect_start = float(raw_ds)
            else:
                defect_start = raw_ds
        except (TypeError, ValueError):
            defect_start = None

    sc_ts = ts_grouped.get(sid)
    if sc_ts is None:
        return []

    fleet_data = {}
    for mid in members:
        member_ts = sc_ts[sc_ts[legend_axis].astype(str) == str(mid)].sort_values(x_col)
        if member_ts.empty:
            continue
        x_vals = member_ts[x_col].to_numpy()
        y_vals = member_ts['value'].to_numpy()
        if len(x_vals) == 0:
            continue
        fleet_data[mid] = (x_vals, y_vals)

    if not fleet_data:
        return []

    title_parts = [str(row_dict[col]) for col in title_columns if col in row_dict and row_dict[col] is not None]
    chart_title = ' / '.join(title_parts) if title_parts else sid

    target = read_target(row_dict)

    # Render N images — one per fleet member as highlighted_member
    outputs = []
    for highlighted_member in fleet_data.keys():
        # Determine per-image label
        if highlighted_member == anomaly_member and chart_class != 'normal':
            img_class = chart_class  # inherits defect type
            anomalous_ids = [highlighted_member]
            disp_defect_start = defect_start if chart_class != 'context' else None
        else:
            img_class = 'normal'
            anomalous_ids = []
            disp_defect_start = None

        filename = f"{sid}_{highlighted_member}.png"
        train_path = images_dir / split / img_class / filename
        disp_path = display_dir / split / img_class / filename
        train_path.parent.mkdir(parents=True, exist_ok=True)
        disp_path.parent.mkdir(parents=True, exist_ok=True)

        renderer.render_overlay(fleet_data, highlighted_member, str(train_path),
                                target=target)
        renderer.render_overlay_display(
            fleet_data, highlighted_member, str(disp_path),
            anomalous_ids=anomalous_ids,
            defect_start_idx=disp_defect_start,
            title=chart_title,
            x_label=x_label,
            y_label=y_label,
            target=target,
        )
        outputs.append({
            'family_id': row_dict.get('family_id', sid),
            'chart_id': sid,
            'highlighted_member': highlighted_member,
            'class': img_class,
            'split': split,
            'legend_axis': legend_axis,
            'members': ','.join(members),
            'defect_start_idx': row_dict.get('defect_start_idx'),
            'target': target,
            'image_name': filename,
            'image_relpath': f"{split}/{img_class}/{filename}",
            'original_chart_class': chart_class,
            'source_highlighted_member': anomaly_member,
            'device': row_dict.get('device'),
            'step': row_dict.get('step'),
            'item': row_dict.get('item'),
        })
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='dataset.yaml')
    parser.add_argument('--suffix', default='vd080',
                        help='dataset suffix — data_{suffix}/ folder')
    parser.add_argument('--workers', type=int, default=0,
                        help='0 = auto, 1 = sequential')
    parser.add_argument('--x_col', type=str, default=None)
    parser.add_argument('--limit', type=int, default=None,
                        help='optional: limit to first N scenarios for quick test')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Source data
    data_dir = Path(f'data_{args.suffix}')
    if not data_dir.exists():
        data_dir = Path(config['output']['data_dir'])
    scenarios_path = data_dir / 'scenarios.csv'
    timeseries_path = data_dir / 'timeseries.csv'

    # Target output dirs
    images_dir = Path(f'images_per_member_{args.suffix}')
    display_dir = Path(f'display_per_member_{args.suffix}')
    pm_data_dir = Path(f'data_per_member_{args.suffix}')
    pm_data_dir.mkdir(exist_ok=True)

    print(f"[per-member] config: {args.config}")
    print(f"[per-member] source data: {data_dir}")
    print(f"[per-member] output: {images_dir}")
    print(f"[per-member] scenarios: {scenarios_path}")

    sc_df = pd.read_csv(scenarios_path)
    if args.limit:
        sc_df = sc_df.head(args.limit)
    print(f"[per-member] scenarios to process: {len(sc_df)}")

    # Determine x_col
    ts_df = pd.read_csv(timeseries_path, nrows=100)
    x_col = args.x_col or ('time_index' if 'time_index' in ts_df.columns else ts_df.columns[1])

    # Load + group timeseries (heavy)
    print(f"[per-member] loading timeseries ({timeseries_path.stat().st_size / 1e9:.2f} GB)...")
    ts_df = pd.read_csv(timeseries_path)
    ts_grouped = {sid: g for sid, g in ts_df.groupby('chart_id')}

    # pickle for worker sharing
    import tempfile
    ts_pickle = Path(tempfile.gettempdir()) / f'ts_grouped_{args.suffix}.pkl'
    with open(ts_pickle, 'wb') as f:
        pickle.dump(ts_grouped, f)
    print(f"[per-member] pickled timeseries to {ts_pickle}")

    # Parallel workers
    if args.workers == 0:
        nw = max(1, cpu_count() - 2)
    else:
        nw = args.workers
    print(f"[per-member] workers: {nw}")

    rows = sc_df.to_dict(orient='records')

    all_outputs = []
    if nw > 1:
        with Pool(processes=nw,
                  initializer=_init_worker,
                  initargs=(str(ts_pickle), config, x_col, str(images_dir), str(display_dir))) as pool:
            for outs in tqdm(pool.imap_unordered(_render_per_member, rows, chunksize=10),
                             total=len(rows), desc='render'):
                all_outputs.extend(outs)
    else:
        _init_worker(str(ts_pickle), config, x_col, str(images_dir), str(display_dir))
        for row in tqdm(rows, desc='render'):
            all_outputs.extend(_render_per_member(row))

    # Save expanded scenarios
    pm_df = pd.DataFrame(all_outputs)
    pm_scenarios_path = pm_data_dir / 'scenarios_per_member.csv'
    pm_df.to_csv(pm_scenarios_path, index=False)
    print(f"[per-member] wrote {len(pm_df)} rows to {pm_scenarios_path}")

    # Summary by class
    print("\n[per-member] class distribution (per-image):")
    print(pm_df.groupby(['split', 'class']).size().unstack(fill_value=0))

    # Cleanup
    ts_pickle.unlink(missing_ok=True)
    print(f"\n[per-member] DONE")


if __name__ == '__main__':
    main()
