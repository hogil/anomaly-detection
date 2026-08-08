#!/usr/bin/env python
"""데이터셋 검토용 HTML 시트를 만든다 — 정상 하위유형과 대응 불량을 나란히.

생성한 데이터가 의도대로인지 눈으로 확인하는 용도. 표본을 새로 뽑아 렌더하고,
이미지를 base64 로 박아 **파일 하나로 완결**되게 만든다 (외부 요청 0, 오프라인 열람 가능).

  python scripts/build_dataset_review.py --config configs/datasets/dataset_v29.yaml

기본 출력은 docs/dataset_review_<version>.html. 사내망처럼 외부가 막힌 곳에서는
repo 를 pull 받아 이 파일을 브라우저로 바로 열면 된다.

표본은 임시 폴더에 만들고 끝나면 지운다 — 실제 학습 데이터는 건드리지 않는다.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

# (제목, 심볼, config key, 설명, [(라벨 조건, 카드 수, note 포맷)])
# 해당 variant 가 데이터에 없으면 그 절은 통째로 건너뛴다.
SECTIONS = [
    ("②", "한 멤버가 중간에서 변하는 건 양호, 우측 끝만 불량", "mid_shift",
     "target 한 멤버만 시계열 <b>중간</b>에서 레벨이 바뀌고 그대로 유지된다. 불량 mean_shift 는 "
     "우측 끝 구간에만 들어가므로 <b>바뀐 뒤 안정 구간이 얼마나 긴가</b>가 구분 신호가 된다 — "
     "절대 위치가 아니라 폭 특징이라 global average pooling 구조에서도 배우기 쉽다.",
     [("normal_variant", "mid_shift", 4,
       lambda d: f'<b>{d.get("start_ratio", 0):.0%}</b> 지점에서 <b>{d.get("shift_sigma")}σ</b> '
                 f'변화 → 이후 <b>{d.get("points_after")}점</b> 안정'),
      ("class", "mean_shift", 2,
       lambda d: '<span class="hl-abn">불량</span> — 우측 끝 구간에서 변화, 안정 구간 없음')]),

    ("→", "우측 끝이 조금 움직인 건 양호", "right_minor",
     "기존에는 우측이 거의 평평하도록 강제돼 “우측이 움직이면 불량”이 돼 버렸다. "
     "불량 하한(2.2σ) 아래 구간을 정상으로 채운다 — 이동 0.5~1.2σ 또는 산포 1.15~1.5배.",
     [("normal_variant", "right_minor", 3,
       lambda d: f'우측 {d.get("start_ratio", 0):.0%}~ · <b>{d.get("kind")}</b> '
                 f'{d.get("shift_sigma") or d.get("spread_scale")}'),
      ("class", "mean_shift", 1,
       lambda d: '<span class="hl-abn">비교용 불량</span> — 같은 우측 구간, 훨씬 큰 변동'),
      ("class", "standard_deviation", 1,
       lambda d: '<span class="hl-abn">비교용 불량</span> — 같은 우측 구간, 훨씬 큰 산포')]),

    ("👥", "멤버가 1~3대뿐인 chart", "small_fleet",
     "기존 데이터는 멤버가 최소 4대라 이런 그림이 아예 없었다. 멤버가 <b>1개면</b> 비교 대상이 "
     "없어 판정 불가 → 정상. <b>2~3대</b>면 fleet 이 밴드가 아니라 기준선 하나라 작은 차이도 "
     "크게 보인다 — 현업 오검이 여기 몰려 있어 비중을 크게 잡았다.",
     [("variant", "single_legend", 2, lambda d: '멤버 <b>1개</b> — 비교 대상 없음'),
      ("variant", "smallfleet2", 2, lambda d: '멤버 <b>2대</b> — 기준선이 하나뿐'),
      ("abn_small", "mean_shift", 1,
       lambda d: '<span class="hl-abn">비교용 불량</span> — 2~3대에서도 우측 변동이 크면 불량'),
      ("abn_small", "context", 1,
       lambda d: '<span class="hl-abn">비교용 불량</span> — 이웃 하나뿐이어도 명확히 멀면 불량')]),

    ("↩", "중간에 났다가 되돌아온 건 양호", "recovered",
     "mean_shift · std · drift · spike 를 시계열 <b>중간 구간</b>에만 넣고 그 뒤는 baseline 으로 "
     "복귀시킨다. 진폭은 <b>진짜 불량과 같은 설정</b>이다 — 약하게 넣으면 모델이 “작으면 정상”을 "
     "배울 뿐 복귀 여부를 안 본다. <b>우측 끝이 깨끗한가</b>만으로 갈려야 한다.",
     [("recovered_kind", "drift_return", 2, None), ("recovered_kind", "mean_shift", 1, None),
      ("recovered_kind", "drift", 1, None), ("recovered_kind", "standard_deviation", 1, None),
      ("recovered_kind", "spike", 1, None),
      ("class", "mean_shift", 1,
       lambda d: '<span class="hl-abn">비교용 불량</span> — 우측 끝에서 변하고 <b>복귀 없음</b>'),
      ("class", "drift", 1,
       lambda d: '<span class="hl-abn">비교용 불량</span> — 우측 끝 drift, 복귀 없음')]),

    ("①", "spike 2~3개도 불량으로 나옴", "few_spike",
     "판별 기준은 <b>크기가 아니라 개수</b>다. 크기를 작게 두면 모델은 “작은 튐은 정상”만 배우고 "
     "현업의 크고 개수 적은 2~3매는 계속 잡는다 — 그래서 불량 spike 와 <b>같은 스케일</b>로 두고 "
     "<b>개수</b>로만 가른다.",
     [("normal_variant", "few_spike", 4,
       lambda d: f'튄 점 <b>{d.get("num_spikes")}개</b> · 평균 <b>{d.get("avg_magnitude_sigma")}σ</b>'),
      ("class", "spike", 2,
       lambda d: f'<span class="hl-abn">불량</span> — 튄 점 <b>{d.get("num_spikes")}개</b>, '
                 f'우측 구간 집중')]),

    ("⏹", "최근에 안 돌린 설비는 양호", "early_stop",
     "target 이 <b>왼쪽에만</b> 있고 최근(우측)에는 없는 경우. 불량은 전부 우측 끝 구간에 "
     "들어가므로 그 구간에 점이 아예 없으면 <b>판정 대상이 아니다</b> — context(avg/std)처럼 "
     "전 구간을 보는 것도 같은 규칙으로 정상 처리한다. fleet 은 전 구간 유지해서 "
     "“혼자만 최근에 안 돌았다”가 드러나게 한다. 불량 클래스에는 적용하지 않는다.",
     [("variant", "early_stop", 2,
       lambda d: 'target 만 <b>왼쪽에서 끊김</b>' + (
           f' · fleet 대비 {d["offset_sigma"]:.2f}σ 치우침' if d.get("offset_sigma") else '')),
      ("variant", "degraded", 2, lambda d: '<b>열화된 채로 멈춤</b> — 최근 데이터가 없으니 정상'),
      ("class", "drift", 1,
       lambda d: '<span class="hl-abn">비교용 불량</span> — 우측 끝까지 데이터가 있고 거기서 열화')]),

    ("⏱", "그 설비만 최근에 시작한 건 양호", "late_start",
     "2달치 데이터인데 특정 설비만 <b>마지막 며칠</b>만 진행된 경우. target 의 점이 전부 "
     "불량 구간(우측)에 몰려 있어 오검이 많이 났다. fleet 은 전 구간 그대로 두고 target 만 "
     "우측 끝 3~25% 에서 시작하게 만든다. 불량 클래스에서는 꼬리에 점이 10개 미만이면 "
     "되돌린다 — <b>점 몇 개로는 판정하지 않는다.</b>",
     [("variant", "late_start_target", 2,
       lambda d: '<b>그 설비만</b> 늦게 시작 — 다른 eqp 는 계속 진행'),
      ("variant", "late_start_all", 2,
       lambda d: '<b>다 같이</b> 늦게 시작 — chart 전체가 우측에만'),
      ("abn_late", "spike", 1,
       lambda d: f'<span class="hl-abn">비교용 불량</span> — 늦게 시작해도 우측에 '
                 f'<b>{d.get("num_spikes")}개</b>면 불량'),
      ("abn_late", "context", 1,
       lambda d: '<span class="hl-abn">비교용 불량</span> — 점이 적어도 fleet 에서 '
                 '<b>명확히 멀면</b> 불량')]),

    ("④", "계측 모수가 작으면 불량으로만 나옴", "sparse_chart",
     "<b>차트 전체</b>가 성긴 경우와 <b>특정 설비 하나만</b> 성긴 경우를 모두 만든다. 클래스와 "
     "무관하게 적용하는 것이 핵심 — 정상에만 걸면 “점이 적으면 정상”이라는 새 지름길이 생긴다. "
     "다만 불량 구간에 점이 최소치 미만으로 남으면 target 은 원래 밀도로 되돌린다 — "
     "<b>점 3개짜리 불량은 판정 근거가 안 된다.</b>",
     [("fewest", "sparse_member", 2, lambda d: '정상 — <b>특정 설비만</b> 계측이 성김'),
      ("fewest", "sparse_chart", 2, lambda d: '정상 — 차트 전체가 성김'),
      ("abn_sparse", "spike", 1,
       lambda d: f'<span class="hl-abn">비교용 불량</span> — 성겨도 우측에 '
                 f'<b>{d.get("num_spikes")}개</b>가 남는다'),
      ("abn_sparse", "context", 1,
       lambda d: '<span class="hl-abn">비교용 불량</span> — 점이 적어도 이격이 명확')]),

    ("≠", "fleet 에서 조금 떨어진 정상 — eqp 2대짜리 포함", "context_like",
     "정상은 target 전체 평균이 fleet 평균의 <b>0.6σ 이내</b>로 강제되고 context 불량은 그보다 "
     "훨씬 멀어야 했다. 그 사이가 비어 <b>조금만 치우쳐도 context 불량</b>이 됐다. "
     "0.8~2.0σ 구간을 정상으로 채우고 context 하한도 올렸다. 멤버가 <b>2대</b>면 fleet 이 밴드가 "
     "아니라 기준선 1개라 작은 차이도 크게 보이므로, 2대짜리 chart 비중을 올렸다.",
     [("normal_variant", "context_like", 3,
       lambda d: f'fleet 평균에서 <b>{d.get("offset_sigma", 0):.2f}σ</b> 이격 · 전 구간 균일'),
      ("variant", "smallfleet2", 2, lambda d: '멤버 <b>2대</b> — 기준선이 하나뿐'),
      ("class", "context", 1,
       lambda d: '<span class="hl-abn">context 불량</span> — <b>훨씬</b> 멀리 떨어진다')]),

    ("＋", "이웃보다 산포가 큰 정상 — 단, 월등히 크면 불량", "loose_target",
     "정상은 fleet 대비 <b>조금만</b> 넓게. 월등히 넓은 것은 context 불량이 맡고, 그쪽 하한을 "
     "올려 간격을 벌렸다.",
     [("normal_variant", "loose_target", 3,
       lambda d: f'fleet 대비 <b>{d.get("vs_fleet_within", 0):.2f}배</b> · 전 구간 균일, 평균 유지'),
      ("class", "context", 2,
       lambda d: '<span class="hl-abn">context 불량</span> — 이웃보다 <b>월등히</b> 넓거나 치우침')]),

    ("기준", "비교용 — 변형 없는 정상과 기존 불량", "clean",
     "새 유형이 기존 그림을 밀어내지 않았는지 확인하는 기준선.",
     [("clean", "", 2, lambda d: "변형 없음 — 기존 정상과 동일"),
      ("class", "drift", 1, lambda d: '<span class="hl-abn">drift 불량</span> — 우측 끝 주입'),
      ("class", "standard_deviation", 1,
       lambda d: '<span class="hl-abn">std 불량</span> — 우측 구간만 확대')]),
]


def sample_dataset(cfg_path: Path, work: Path, n_normal: int, n_abn: int,
                   workers: int, python: str) -> Path:
    """표본용 소규모 데이터셋을 임시 폴더에 생성하고 data 폴더를 반환."""
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    version = f"review_{cfg_path.stem}"
    cfg["dataset"]["version"] = version
    cfg["dataset"]["samples_per_class"] = {
        k: (n_normal if k == "normal" else n_abn)
        for k in cfg["dataset"]["samples_per_class"]}
    cfg["output"] = {"data_dir": str(work / "data"), "image_dir": str(work / "images"),
                     "display_dir": str(work / "display")}
    tmp_cfg = work / "config.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    snap_dir = ROOT / "configs" / "datasets"
    before = set(snap_dir.glob("*.yaml")) if snap_dir.is_dir() else set()
    # generate_data.py 는 latest.yaml 포인터를 갱신한다. 표본 생성으로 그게 바뀌면
    # "마지막에 만든 데이터셋" 이 표본용 config 를 가리키게 되므로 원래대로 되돌린다.
    latest = snap_dir / "latest.yaml"
    latest_backup = latest.read_bytes() if latest.exists() else None

    for script in ("generate_data.py", "generate_images.py"):
        print(f"  [review] {script}")
        rc = subprocess.run([python, script, "--config", str(tmp_cfg),
                             "--workers", str(workers)], cwd=ROOT).returncode
        if rc != 0:
            raise SystemExit(f"{script} 실패 (rc={rc})")

    # 표본용 스냅샷 yaml 정리 (보존 가치 없음)
    for path in (set(snap_dir.glob("*.yaml")) - before):
        if path.stem.startswith(f"dataset_{version}"):
            path.unlink(missing_ok=True)
    if latest_backup is not None:
        latest.write_bytes(latest_backup)
    elif latest.exists():
        latest.unlink()
    return work


def embed(path: Path, width: int, quality: int) -> str:
    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def build_cards(df: pd.DataFrame, work: Path, specs) -> str:
    cards = []
    for kind, key, count, note_fn in specs:
        if kind == "normal_variant":
            mask = df["dp"].map(lambda d: d.get("normal_variant") == key)
        elif kind == "variant":
            mask = df["variant"].str.contains(key) & (df["class"] == "normal")
        elif kind == "recovered_kind":
            mask = df["dp"].map(lambda d: d.get("normal_variant") == "recovered"
                                and d.get("kind") == key)
            note_fn = (lambda d: f'<b>{d.get("kind")}</b> 가 {d.get("start_ratio", 0):.0%}~'
                                 f'{d.get("end_ratio", 0):.0%} 구간에 났다가 이후 '
                                 f'<b>{d.get("points_after")}점</b> 복귀')
        elif kind == "fewest":
            base = df["variant"].str.contains(key) & (df["class"] == "normal")
            sub = df[base].sort_values("npoints")
            for _, row in sub.head(count).iterrows():
                card = _card(row, work, note_fn)
                if card:
                    cards.append(card)
            continue
        elif kind == "abn_small":
            nmem = df["members"].astype(str).str.count(",") + 1
            mask = (df["class"] == key) & (nmem <= 3)
        elif kind == "abn_late":
            mask = (df["class"] == key) & df["variant"].str.contains("late_start")
        elif kind == "abn_sparse":
            mask = (df["class"] == key) & df["sparse"] & ~df["variant"].str.contains("fleetonly")
        elif kind == "clean":
            mask = df["variant"] == "clean"
        else:
            mask = df["class"] == key
        for _, row in df[mask].head(count).iterrows():
            card = _card(row, work, note_fn)
            if card:
                cards.append(card)
    return "".join(cards)


def _card(row, work: Path, note_fn) -> str:
    """카드 1장. 이미지가 없으면 빈 문자열."""
    disp = work / "display" / row["split"] / row["class"] / f'{row["chart_id"]}.png'
    trn = work / "images" / row["split"] / row["class"] / f'{row["chart_id"]}.png'
    if not disp.exists():
        return ""
    tag_kind = "abn" if row["class"] != "normal" else "nor"
    tags = f'<span class="tag tag-{tag_kind}">{row["class"]}</span>'
    if row["sparse"]:
        mode = "member" if "member" in row["variant"] else "chart"
        tags += f'<span class="tag tag-sparse">sparse·{mode}</span>'
    npts = row.get("npoints")
    if npts is not None:
        tags += f'<span class="tag tag-pts">{int(npts)}점</span>'
    thumb = ""
    if trn.exists():
        thumb = (f'<figure class="thumb"><img src="data:image/jpeg;base64,'
                 f'{embed(trn, 224, 82)}" alt="학습 입력">'
                 f'<figcaption>모델 입력</figcaption></figure>')
    return f"""
      <article class="card">
        <header class="card-head">{tags}<code class="cid">{row["chart_id"]}</code></header>
        <img class="chart" src="data:image/jpeg;base64,{embed(disp, 540, 74)}" alt="{row['class']}">
        <div class="card-foot"><p class="note">{note_fn(row["dp"])}</p>{thumb}</div>
      </article>"""


STYLE = """
:root {
  --paper:#FAFBFC; --panel:#FFF; --ink:#141820; --ink-2:#5A616E;
  --line:#E2E6EC; --line-2:#EFF2F6; --accent:#3D6BC4; --accent-soft:#E8EFFB;
  --abn:#C0392F; --abn-soft:#FBEAE8; --warn:#8A6A1F; --warn-soft:#FBF2DC;
}
@media (prefers-color-scheme: dark) { :root {
  --paper:#0D1117; --panel:#151B24; --ink:#E6EAF0; --ink-2:#9AA3B2;
  --line:#242C38; --line-2:#1C232E; --accent:#7BA5E8; --accent-soft:#1A2740;
  --abn:#E8837A; --abn-soft:#3A1F1C; --warn:#D8B25C; --warn-soft:#332A16; } }
:root[data-theme="dark"] {
  --paper:#0D1117; --panel:#151B24; --ink:#E6EAF0; --ink-2:#9AA3B2;
  --line:#242C38; --line-2:#1C232E; --accent:#7BA5E8; --accent-soft:#1A2740;
  --abn:#E8837A; --abn-soft:#3A1F1C; --warn:#D8B25C; --warn-soft:#332A16; }
:root[data-theme="light"] {
  --paper:#FAFBFC; --panel:#FFF; --ink:#141820; --ink-2:#5A616E;
  --line:#E2E6EC; --line-2:#EFF2F6; --accent:#3D6BC4; --accent-soft:#E8EFFB;
  --abn:#C0392F; --abn-soft:#FBEAE8; --warn:#8A6A1F; --warn-soft:#FBF2DC; }
* { box-sizing:border-box; }
body { margin:0; background:var(--paper); color:var(--ink); font-size:15px; line-height:1.6;
  font-family:"Pretendard","Malgun Gothic",ui-sans-serif,system-ui,-apple-system,sans-serif;
  -webkit-font-smoothing:antialiased; }
code, .num, .cid { font-family:ui-monospace,"Cascadia Mono",Consolas,monospace;
  font-variant-numeric:tabular-nums; }
.wrap { max-width:1180px; margin:0 auto; padding:44px 24px 80px; }
.masthead { border-bottom:2px solid var(--ink); padding-bottom:20px; }
.eyebrow { font-size:11px; letter-spacing:.16em; text-transform:uppercase; color:var(--ink-2); margin:0 0 10px; }
h1 { font-size:clamp(26px,3.4vw,38px); line-height:1.15; margin:0 0 12px; text-wrap:balance; letter-spacing:-.015em; }
.lede { margin:0; max-width:68ch; color:var(--ink-2); }
.lede b { color:var(--ink); font-weight:600; }
.legend { display:flex; flex-wrap:wrap; gap:20px; margin-top:22px; padding:14px 16px;
  background:var(--panel); border:1px solid var(--line); border-radius:4px; font-size:13px; color:var(--ink-2); }
.legend span { display:flex; align-items:center; gap:7px; }
.dot { width:11px; height:11px; border-radius:50%; flex:none; }
.block { margin-top:52px; }
.block-head { display:flex; gap:16px; align-items:flex-start; border-top:1px solid var(--line);
  padding-top:18px; margin-bottom:20px; }
.sym { flex:none; width:38px; height:38px; display:grid; place-items:center; background:var(--accent-soft);
  color:var(--accent); border-radius:3px; font-size:17px; font-weight:700; }
.block-head h2 { font-size:19px; margin:0 0 5px; letter-spacing:-.01em; }
.sub { margin:0; font-size:13.5px; color:var(--ink-2); max-width:76ch; }
.sub b { color:var(--ink); font-weight:600; }
.sub code { font-size:12.5px; background:var(--line-2); padding:1px 5px; border-radius:3px; color:var(--ink); }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:18px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:5px; overflow:hidden;
  display:flex; flex-direction:column; }
.card-head { display:flex; align-items:center; gap:7px; padding:9px 12px; border-bottom:1px solid var(--line-2); }
.tag { font-size:10.5px; letter-spacing:.06em; text-transform:uppercase; font-weight:700;
  padding:2.5px 7px; border-radius:3px; }
.tag-nor { background:var(--accent-soft); color:var(--accent); }
.tag-abn { background:var(--abn-soft); color:var(--abn); }
.tag-sparse { background:var(--warn-soft); color:var(--warn); }
.tag-pts { background:var(--line-2); color:var(--ink-2); }
.cid { margin-left:auto; font-size:11px; color:var(--ink-2); }
.chart { width:100%; display:block; background:#fff; }
.card-foot { display:flex; gap:12px; align-items:center; padding:11px 12px;
  border-top:1px solid var(--line-2); margin-top:auto; }
.note { margin:0; font-size:12.5px; color:var(--ink-2); flex:1; }
.note b { color:var(--ink); font-weight:600; font-family:ui-monospace,Consolas,monospace; }
.hl-abn { color:var(--abn); font-weight:700; }
.thumb { margin:0; flex:none; width:62px; text-align:center; }
.thumb img { width:62px; height:62px; display:block; border:1px solid var(--line); border-radius:3px; background:#fff; }
.thumb figcaption { font-size:9.5px; color:var(--ink-2); margin-top:3px; letter-spacing:.04em; }
.tablewrap { overflow-x:auto; margin-top:16px; }
table { border-collapse:collapse; width:100%; min-width:520px; font-size:13.5px; }
th, td { text-align:left; padding:9px 12px; border-bottom:1px solid var(--line-2); }
th { font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-2);
  border-bottom:1px solid var(--line); }
td.num { text-align:right; font-family:ui-monospace,Consolas,monospace; font-variant-numeric:tabular-nums; }
footer { margin-top:56px; padding-top:18px; border-top:1px solid var(--line); font-size:12.5px; color:var(--ink-2); }
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/datasets/dataset_v29.yaml")
    parser.add_argument("--out", default=None, help="기본 docs/dataset_review_<version>.html")
    parser.add_argument("--normal", type=int, default=160, help="표본 정상 장수")
    parser.add_argument("--abnormal", type=int, default=26, help="표본 불량 클래스당 장수")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--keep-samples", action="store_true", help="임시 표본 폴더를 지우지 않는다")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    if not cfg_path.exists():
        raise SystemExit(f"--config 를 찾지 못했습니다: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    version = str(cfg.get("dataset", {}).get("version", cfg_path.stem))
    out = Path(args.out) if args.out else ROOT / "docs" / f"dataset_review_{version}.html"
    if not out.is_absolute():
        out = ROOT / out

    work = Path(tempfile.mkdtemp(prefix="dsreview_"))
    try:
        sample_dataset(cfg_path, work, args.normal, args.abnormal, args.workers, args.python)
        df = pd.read_csv(work / "data" / "scenarios.csv")
        df["variant"] = df["variant"].fillna("") if "variant" in df.columns else ""
        df["dp"] = df["defect_params"].map(
            lambda s: json.loads(s) if isinstance(s, str) and s != "{}" else {})
        df["sparse"] = df["variant"].str.contains("sparse")
        ts = pd.read_csv(work / "data" / "timeseries.csv")
        by_chart = {cid: sub for cid, sub in ts.groupby("chart_id")}

        def _npoints(row):
            sub = by_chart.get(row["chart_id"])
            if sub is None:
                return 0
            return int((sub[row["legend_axis"]].astype(str)
                        == str(row["highlighted_member"])).sum())

        df["npoints"] = df.apply(_npoints, axis=1)

        blocks = []
        for sym, title, key, body, specs in SECTIONS:
            cards = build_cards(df, work, specs)
            if not cards:
                print(f"  [review] 건너뜀 — 표본에 {key} 없음")
                continue
            blocks.append(f"""
  <section class="block">
    <header class="block-head"><span class="sym">{sym}</span>
      <div><h2>{title}</h2><p class="sub"><code>{key}</code> — {body}</p></div></header>
    <div class="grid">{cards}</div>
  </section>""")

        spc = cfg["dataset"]["samples_per_class"]
        weights = (cfg.get("normal_variants") or {}).get("weights") or {}
        n_norm = spc.get("normal", 0)
        qty = "".join(
            f'<tr><td><code>{k}</code></td><td class="num">{w:.0%}</td>'
            f'<td class="num">{round(n_norm * w):,}</td></tr>'
            for k, w in weights.items() if w)
        n_abn = sum(v for k, v in spc.items() if k != "normal")

        html = f"""<title>데이터셋 검토 — {version}</title>
<style>{STYLE}</style>
<div class="wrap">
  <header class="masthead">
    <p class="eyebrow">{version} · 생성 데이터 검토</p>
    <h1>정상 하위유형과 대응 불량</h1>
    <p class="lede">현업에서 정상을 불량으로 잡던 케이스를 <b>정상 클래스에 추가</b>한 결과다.
    새 정상 유형과 구분 대상이 되는 불량을 나란히 놓았고, 카드마다 사람이 보는 차트와
    <b>모델이 실제로 받는 224px 입력</b>을 같이 실었다.</p>
    <div class="legend">
      <span><i class="dot" style="background:#4878CF"></i>target (판정 대상)</span>
      <span><i class="dot" style="background:#B0B0B0"></i>fleet (회색 멤버)</span>
      <span><i class="dot" style="background:#C0392F"></i>주입된 불량 구간</span>
      <span>표본 — 정상 {args.normal} · 불량 {args.abnormal}×{len(spc) - 1}</span>
    </div>
  </header>
{''.join(blocks)}

  <section class="block">
    <header class="block-head"><span class="sym">#</span>
      <div><h2>전체 학습셋 수량</h2>
      <p class="sub">정상 <code>{n_norm:,}</code> · 불량 <code>{n_abn:,}</code>
      → 정상:불량 = <b>{n_norm / max(n_abn, 1):.1f} : 1</b></p></div></header>
    <div class="tablewrap"><table>
      <thead><tr><th>정상 유형</th><th class="num">비율</th><th class="num">장수</th></tr></thead>
      <tbody>{qty}</tbody></table></div>
  </section>

  <footer>
    <code>python scripts/build_dataset_review.py --config {cfg_path.relative_to(ROOT).as_posix()}</code>
    로 다시 만들 수 있다. 이미지가 파일 안에 들어 있어 외부 접속 없이 열린다.
  </footer>
</div>"""
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        print(f"[review] {out.relative_to(ROOT).as_posix()}  ({out.stat().st_size / 1e6:.2f} MB)")
    finally:
        if args.keep_samples:
            print(f"[review] 표본 유지: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
