$utf8 = New-Object System.Text.UTF8Encoding($false)

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Content
    )
    [System.IO.File]::WriteAllText($Path, $Content, $utf8)
}

$root = 'C:\Users\hgcho\.codex\skills'

Write-Utf8NoBom -Path (Join-Path $root 'anomaly-paper-evidence\SKILL.md') -Content @'
---
name: anomaly-paper-evidence
description: Route anomaly-detection paper work across literature, planning, analysis, live runs, and claim gates.
---

# Anomaly Paper Evidence

## Core Role

Orchestrate the full paper-evidence workflow around a frozen operating baseline. The goal is not to crown a single winner. The goal is to prove whether each axis has a real effect, where the best neighborhood is, where saturation starts, and which settings create instability.

## Active Operating Reference

Use `fresh0412_v11_n700_existing` as the operating baseline on 2026-04-25:

- 5 seeds: `42,1,2,3,4`
- mean `F1=0.9901`
- mean `FN=9.8`
- mean `FP=5.0`
- band-hit `3/5`
- limitation: `s42 FP=3`, `s2 FP=4`

Do not restart reference search unless the user explicitly reopens it.

## Child-Skill Routing

- `anomaly-literature-research`: primary papers, related-work matrix, benchmark saturation and robustness context.
- `anomaly-paper-purpose`: claim wording, maturity, limits, and paper story.
- `anomaly-experiment-planner`: axis grids, seeds, queues, and stop rules.
- `anomaly-result-analysis`: per-seed tables, curve shape, deltas, and instability evidence.
- `anomaly-live-orchestrator`: active process control, queue revision, and follow-up relaunch.

## Success Condition

For every important axis, try to answer:

- Does the axis change `FN`, `FP`, `F1`, worst seed, or stability?
- How large is the delta vs the operating baseline for `FN`, `FP`, and `F1`?
- Do repeated wrong predictions cluster on the same `chart_id` values across strong runs?
- Are some errors likely label or annotation problems rather than tunable model failures?
- Is there a sweet spot or best neighborhood?
- Does performance degrade when moving away from that neighborhood?
- Is the shape roughly quadratic-like, saturation-like, monotonic, or inconclusive?
- Which collapse or oscillation cases explain the effect?

## Evidence Gates

- `E0 unusable`: crash, missing metrics, wrong protocol, or unverifiable artifact.
- `E1 exploratory`: one seed, incomplete run, or weak source.
- `E2 directional`: same protocol plus enough points to show an axis trend or local neighborhood effect.
- `E3 paper-grade`: same protocol, planned seeds, per-seed table, mean/std, shape interpretation, instability handling, and explicit limitation.

## Operating Rules

1. Do not talk as if the goal is "candidate selection". Talk about axis effect, best neighborhood, saturation, or no effect.
2. Preserve `validations/instability_cases_report.md`, `.csv`, and `.json`. Collapse, oscillation, and optimistic spikes are evidence, not trash.
3. If a run family becomes too clean, classify it as saturation or over-suppression evidence.
4. Never hide neighboring values. A good claim needs the center and the values around it.
5. Prefer paired seeds and same-protocol comparisons.
6. Track prediction-trend evidence with `scripts/analyze_prediction_trends.py` and separate `config_sensitive`, `persistent_hard_sample`, and `label_or_annotation_suspect`.
7. Every paper claim must map to a baseline, tested levels, observed shape, and limitation.

## Workflow

1. Read `references/current-project-state.md`.
2. Confirm the operating baseline and active queues.
3. Route literature, planning, live control, and analysis in that order when needed.
4. Keep the live runner filling missing levels near the suspected sweet spot and at the edges.
5. Convert only `E2/E3` findings into paper text.

## Reference

Read `references/current-project-state.md` first. Treat the current baseline as operating `E2`, not final `E3`.
'@

Write-Utf8NoBom -Path (Join-Path $root 'anomaly-paper-purpose\SKILL.md') -Content @'
---
name: anomaly-paper-purpose
description: Frame anomaly-detection paper purpose, contribution claims, maturity, and limitations.
---

# Anomaly Paper Purpose

## Role Boundary

Turn experiment evidence into defensible paper claims. Do not plan runs, parse logs, or search literature directly.

## Claim Target

Describe what an axis does, not just which point scored highest. Strong claims look like:

- sweet spot around a value or narrow neighborhood
- saturation after a data scale or sample-count range
- stabilization within a GC, WD, warmup, or smoothing range
- a consistent FN/FP tradeoff shift
- no clear or only weak effect

Each claim must tie to the operating baseline and to neighboring levels.

## Claim Frames

- `Local optimum`: one region is best and nearby values remain strong while farther values degrade.
- `Saturation`: more data helps up to a point, then gains flatten or FP/FN become over-suppressed.
- `Stabilization`: GC, WD, smoothing, or warmup reduces collapse, oscillation, gradient spikes, or bad checkpoint selection.
- `Tradeoff movement`: an option moves FN and FP in opposite directions.
- `Null effect`: the axis barely moves the outcome; say so clearly.

## Claim Discipline

- Do not call something effective from a lone point.
- Show the center, the neighboring levels, and the direction away from the center.
- Use per-seed FP/FN, mean/std, worst seed, and instability tags.
- Use prediction-trend tags to separate tunable misses from likely annotation or ambiguity issues.
- If the best score is too clean, frame it as saturation or ceiling behavior.
- Novelty language requires literature backing. Otherwise use empirical-characterization wording.

## Maturity Labels

- `Exploratory`: useful for follow-up design, not final paper text.
- `Appendix-ready`: the shape is visible, but seed count or statistical support is limited.
- `Main-table-ready`: same protocol, same seeds, mean/std, per-seed table, and limits are available.
- `Core-claim-ready`: main-table-ready plus literature positioning and robustness support.

## Required Claim Elements

For each important claim, include:

- operating baseline and why it was chosen
- baseline mean `F1`, `FN`, `FP`
- tested levels and seed coverage
- delta vs baseline at each important level
- best neighborhood or saturation region
- how results degrade away from that region, if they do
- affected failure mode: `FN`, `FP`, variance, oscillation, gradient spike, or checkpoint-selection error
- whether repeated wrong charts are `config_sensitive`, `persistent_hard_sample`, or `label_or_annotation_suspect`
- limitation or null-result note if the effect is weak

## Active Framing

Use `fresh0412_v11_n700_existing` as the current observable-error baseline. Describe it as the operating baseline, not the final perfect reference.
'@

Write-Utf8NoBom -Path (Join-Path $root 'anomaly-experiment-planner\SKILL.md') -Content @'
---
name: anomaly-experiment-planner
description: Plan controlled anomaly-detection refs, ablations, saturation sweeps, seeds, and queues.
---

# Anomaly Experiment Planner

## Role Boundary

Design experiments and queues that can prove axis behavior. Do not write final paper claims or deep-dive completed logs.

## Active Plan

The live sweep is frozen on `fresh0412_v11_n700_existing` as the operating baseline. Anchor search is stopped. Read `references/experiment-plan-fresh0412.md` before proposing new runs or changing the level grid.

## Planning Goal

Plan enough levels to answer whether an axis has a meaningful effect, where the best neighborhood is, and where behavior degrades or saturates.

## Axis Families

Use different grid logic by axis type:

- `Sweet-spot axes`: `LR`, `warmup`, `WD`, `AW`, `GC`, `smoothing`, `EMA`. Sample a plausible center plus levels on both sides.
- `Scale axes`: `normal_ratio`, `per-class`, total `n`, `max_per_class`. Use widening levels to detect saturation or over-suppression.
- `Rendering axes`: color, background, no-fleet, variants. Keep the same seeds and training recipe so only the visual condition changes.

## Grid Rules

1. Keep seeds paired across levels whenever possible.
2. For sweet-spot axes, include the expected good region and at least one clearly smaller and one clearly larger level.
3. For scale axes, use enough spread to show early gains, mid-range behavior, and plateau or over-suppression.
4. If collapse or strong oscillation appears, add nearby rescue levels instead of only expanding the clean side.
5. If prediction-trend review shows repeated misses on only a subset of settings, treat that as a config-sensitive follow-up opportunity.
6. Do not over-expand a level that is only "best so far". First prove the neighborhood.

## Primary Questions Per Axis

Before launching a queue, declare which question it answers:

- best neighborhood or sweet spot
- saturation or diminishing returns
- FN/FP tradeoff movement
- collapse boundary
- stability region under oscillation or gradient spikes

## Reproducibility Protocol

Record for every run:

- axis name and tested level
- seed
- command and config
- dataset version
- log directory
- primary question
- prediction-trend expectation, if relevant
- stop or continue rule

## Stop and Expand Rules

- Expand to all 5 seeds when the level is needed to define the curve shape or a paper figure.
- Skip or stop a family when it becomes too clean and no longer helps the observable-error story; reclassify it as saturation evidence.
- Preserve collapse and oscillation runs and schedule nearby levels that can prove recovery or stabilization.
- When the same charts stay wrong across many strong runs, route them to prediction-trend review before blaming the axis.
- Avoid declaring an optimum from a single level with no neighbors.

## Controller

Use `scripts/adaptive_experiment_controller.py` and the queued validation JSON files for sequential execution. Prefer PowerShell and direct Python on this Windows workspace.
'@

Write-Utf8NoBom -Path (Join-Path $root 'anomaly-result-analysis\SKILL.md') -Content @'
---
name: anomaly-result-analysis
description: Analyze anomaly experiment logs into FP/FN/F1 tables, statistics, gates, and evidence summaries.
---

# Anomaly Result Analysis

## Role Boundary

Turn completed artifacts into axis-effect evidence. Do not launch training except for small analysis scripts.

## Active Baseline

Use `fresh0412_v11_n700_existing` with seeds `42,1,2,3,4` as the operating baseline for paired comparisons. Always disclose the current limitation: band-hit `3/5`, with low-FP seeds on `s42` and `s2`.

## Required Outputs

For each axis, produce:

- per-level, per-seed table: `level`, `seed`, `F1`, `FN`, `FP`, best epoch, status
- aggregate table: mean/std, worst seed, hit count
- baseline-delta table: `delta_F1`, `delta_FN`, `delta_FP`, paired deltas where seeds overlap
- axis-shape summary: sweet spot, saturation, monotonic tradeoff, collapse boundary, or no clear effect
- best-neighborhood statement, if supported
- limitation sentence

## Metric Extraction

Prefer `best_info.json`. If FP/FN are not stored directly, compute:

- `FN = abnormal.count - round(abnormal.recall * abnormal.count)`
- `FP = normal.count - round(normal.recall * normal.count)`

Use histories for trajectory analysis and label the source clearly when the evidence is weaker.

## Shape Interpretation Rules

- `Quadratic-like sweet spot`: one level or narrow neighborhood is best, and moving away on either side worsens performance or stability.
- `Saturation`: improvement flattens after a range, or extra scale suppresses errors too much to stay useful.
- `Monotonic tradeoff`: one side steadily reduces `FN` or `FP` while harming the other.
- `Collapse boundary`: a region where runs destabilize, spike, or fail.
- `No clear effect`: overlapping noise or inconsistent directions.

Do not force a shape when the data do not support it.

## Stability and Failure Evidence

Always inspect and preserve:

- `history.json` for `val_f1`, `test_f1`, and `val_loss` oscillation
- `step_grad_norms.json` for max, p95, and p99 gradient norms
- `validations/instability_cases_report.md`, `.csv`, `.json`
- `validations/prediction_trend_latest.md`, `.csv`, `.json`

Count not only final collapses, but also:

- optimistic spikes
- late-epoch drift
- missed peaks from checkpoint selection
- recoverable vs unrecoverable instability

## Prediction Trend Review

Run `scripts/analyze_prediction_trends.py` on completed runs and inspect repeated wrong `chart_id` values.

Use these labels:

- `label_or_annotation_suspect`: many strong runs across diverse axes still disagree with the label. Treat as manual-review required before making a tuning claim.
- `persistent_hard_sample`: strong runs often miss it. Usually hard visibility or boundary difficulty.
- `config_sensitive`: some strong settings recover it and others miss it. Use this to explain axis movement.
- `mostly_unstable_only`: mostly wrong only in bad or oscillatory runs.

Do not blame the model for likely label noise, and do not excuse clearly config-sensitive misses as unavoidable.

## Reporting Rule

Never report only the single best F1. Always anchor the report to the operating baseline and state `delta_F1`, `delta_FN`, and `delta_FP`, plus what changed around the tested neighborhood and away from it.
'@

Write-Utf8NoBom -Path (Join-Path $root 'anomaly-live-orchestrator\SKILL.md') -Content @'
---
name: anomaly-live-orchestrator
description: Monitor live anomaly experiments, stop weak runs, revise queues, and relaunch follow-ups.
---

# Anomaly Live Orchestrator

## Role

Act as the master loop for live experiments. Keep the queue moving while preserving evidence quality and curve coverage.

## Active Baseline

The operating baseline is frozen to `fresh0412_v11_n700_existing` as of 2026-04-25. Anchor search is stopped. Do not relaunch it unless the user explicitly asks.

## First Checks

Before changing anything, inspect:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*train.py*' -or $_.CommandLine -like '*adaptive_experiment_controller.py*' }
Get-ChildItem logs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 20 Name,LastWriteTime
Get-Content validations\adaptive_ref_summary.md -ErrorAction SilentlyContinue
```

## Live Objective

Do not manage runs as if the only goal is to find one winner. Manage runs so each important axis gets enough evidence to call:

- best neighborhood
- degradation away from that neighborhood
- saturation or plateau
- collapse boundary
- stabilization region

## Live Decision Rules

- If a promising level has no neighbors, schedule nearby levels before over-expanding it.
- If an edge level collapses or oscillates badly, preserve it and consider GC, smoothing, or warmup rescue levels around that region.
- If a family becomes too clean, stop treating it as a search target and move it into saturation evidence.
- Continue to all 5 seeds only when needed to define the curve shape or a paper figure.
- Never discard optimistic spikes or mid-run oscillation. Add them to the instability bank.
- Refresh prediction-trend review after major batches and before declaring repeated FP/FN as unavoidable.

## State Ledger

Maintain after each batch:

- current active process
- latest completed level and seed
- current axis picture so far
- prediction-trend status for repeated wrong charts
- decision: continue, add neighbors, stop family, or relaunch rescue level
- exact next command or file change

## Reporting During Runs

Each status update should say:

- what is running now
- what just completed
- what the numbers imply about the axis
- what the latest `delta_F1`, `delta_FN`, and `delta_FP` are vs baseline
- whether repeated FP/FN look config-sensitive or label-suspect
- what change, if any, was made to the queue
'@

Write-Utf8NoBom -Path (Join-Path $root 'anomaly-literature-research\SKILL.md') -Content @'
---
name: anomaly-literature-research
description: Find primary anomaly-detection papers and build related-work matrices for claim positioning.
---

# Anomaly Literature Research

## Role Boundary

Find and structure literature that supports or limits the project's axis-effect claims. Do not run experiments or write final claim text without project results.

## Search Rules

- Browse for current literature when the query could have changed.
- Prefer primary sources: arXiv, CVF/OpenAccess, OpenReview, IEEE/ACM/Springer, official dataset or method pages.
- Use surveys only to discover leads, then verify the original papers.
- Do not claim novelty unless the matrix clearly shows the gap.

## Literature Angles For This Project

Search and organize papers around:

- benchmark saturation and ceiling behavior
- industrial visual anomaly detection baselines
- synthetic or rendered anomaly generation
- data-scaling and sample-count effects
- robustness to color, background, or visual shifts
- stabilization, reproducibility, and reporting norms

## Matrix Fields

For each paper, record:

- citation key
- problem and method family
- dataset and metric
- evidence relevant to this project
- limitation
- which axis or claim it informs

## Claim Support Mapping

Map literature back to experiment axes:

- `normal_ratio` and per-class sweeps -> saturation and diminishing returns
- `GC`, `WD`, `warmup`, `smoothing` -> stabilization and collapse suppression
- rendering or color changes -> visual-shift robustness
- seed-wise FP/FN reporting -> reproducibility and evaluation norms

## Output Format

Return:

- related-work matrix
- gap statement
- claim-support map by axis
- citation risks or papers that need closer reading

## Starter Sources

Use these as starting points, then update with current search:

- MVTec AD, CVPR 2019: https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html
- PatchCore, CVPR 2022/arXiv: https://arxiv.org/abs/2106.08265
- CutPaste, CVPR 2021/arXiv: https://arxiv.org/abs/2104.04015
- DRAEM, ICCV 2021/arXiv: https://arxiv.org/abs/2108.07610
- EfficientAD, WACV 2024/arXiv: https://arxiv.org/abs/2303.14535
- SimpleNet, CVPR 2023/arXiv: https://arxiv.org/abs/2303.15140
- MVTec AD 2, arXiv 2025: https://arxiv.org/abs/2503.21622
'@

Write-Utf8NoBom -Path (Join-Path $root 'anomaly-paper-evidence\agents\openai.yaml') -Content @'
interface:
  display_name: "Anomaly Paper Evidence"
  short_description: "Orchestrate axis-effect evidence."
  default_prompt: "Use this skill to route anomaly-detection work into baseline-controlled axis-effect evidence across literature, planning, analysis, and live orchestration."
  context_note: "Baseline is fresh0412_v11_n700_existing (E2): mean F1 0.9901, FN 9.8, FP 5.0, hit 3/5. Prove axis effects, best neighborhoods, saturation, and stabilization rather than hunting a single winner. Keep instability cases in validations/instability_cases_report.* and review repeated wrong chart_id trends in validations/prediction_trend_latest.* before calling errors unavoidable."
'@

Write-Utf8NoBom -Path (Join-Path $root 'anomaly-paper-purpose\agents\openai.yaml') -Content @'
interface:
  display_name: "Anomaly Paper Purpose"
  short_description: "Frame axis-effect claims."
  default_prompt: "Use this skill to turn anomaly-detection results into paper claims about sweet spots, saturation, stabilization, tradeoffs, and limits."
  context_note: "Baseline is fresh0412_v11_n700_existing (E2): mean F1 0.9901, FN 9.8, FP 5.0, hit 3/5. Claims should describe axis behavior and best neighborhoods with neighboring levels, and every claim should include delta_F1/delta_FN/delta_FP vs baseline plus config-sensitive vs label/annotation judgement."
'@

Write-Utf8NoBom -Path (Join-Path $root 'anomaly-experiment-planner\agents\openai.yaml') -Content @'
interface:
  display_name: "Anomaly Experiment Planner"
  short_description: "Plan axis-response sweeps."
  default_prompt: "Use this skill to design anomaly-detection sweeps that can prove sweet spots, saturation, tradeoffs, collapse boundaries, and stability regions."
  context_note: "Baseline is fresh0412_v11_n700_existing (E2): mean F1 0.9901, FN 9.8, FP 5.0, hit 3/5. Plan neighboring levels around promising regions, preserve collapse zones, and use prediction-trend review to decide whether repeated misses need tuning follow-up or manual label review."
'@

Write-Utf8NoBom -Path (Join-Path $root 'anomaly-result-analysis\agents\openai.yaml') -Content @'
interface:
  display_name: "Anomaly Result Analysis"
  short_description: "Analyze axis-effect evidence."
  default_prompt: "Use this skill to turn anomaly logs into per-level FP/FN/F1 tables, curve-shape summaries, instability evidence, and paper-grade limits."
  context_note: "Baseline is fresh0412_v11_n700_existing (E2): mean F1 0.9901, FN 9.8, FP 5.0, hit 3/5. Analyze shape, neighboring-level degradation, instability bank evidence, repeated wrong chart_id trends, and always report delta_F1/delta_FN/delta_FP vs baseline."
'@

Write-Utf8NoBom -Path (Join-Path $root 'anomaly-live-orchestrator\agents\openai.yaml') -Content @'
interface:
  display_name: "Anomaly Live Orchestrator"
  short_description: "Run the axis-evidence loop."
  default_prompt: "Use this skill to keep anomaly experiments running, fill missing levels, preserve instability cases, and adapt queues until each important axis has a defensible shape."
  context_note: "Baseline is fresh0412_v11_n700_existing (E2): mean F1 0.9901, FN 9.8, FP 5.0, hit 3/5. Live decisions should target curve coverage, sweet spots, saturation, stabilization evidence, repeated wrong chart_id review, and baseline-relative deltas rather than a single winner."
'@

Write-Utf8NoBom -Path (Join-Path $root 'anomaly-literature-research\agents\openai.yaml') -Content @'
interface:
  display_name: "Anomaly Literature Research"
  short_description: "Map papers to axis claims."
  default_prompt: "Use this skill to find primary anomaly-detection literature that supports or limits axis-effect claims such as saturation, stabilization, rendering robustness, and reporting norms."
  context_note: "Baseline is fresh0412_v11_n700_existing (E2): mean F1 0.9901, FN 9.8, FP 5.0, hit 3/5. Map papers back to experiment axes and avoid novelty claims unless the gap is explicit."
'@
