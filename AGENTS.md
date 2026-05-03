# Project Agent Notes

These notes are project-specific for `D:/project/anomaly-detection` and supplement the user's global Codex principles.

## Problem Framing

- Treat the primary operating problem as binary image-level pass/fail: `normal` vs `abnormal`.
- Do not claim that binary is universally better than multiclass. The project claim is narrower: production gate and defect-type diagnosis have different jobs.
- `multiclass` is for auxiliary comparison and defect/process analysis.
- `anomaly_type` is not one-class anomaly detection. In this repo it means supervised abnormal-only defect type classification.

## Two-Stage Workflow

- Stage 1 is trained with `python train.py --mode binary`.
- Stage 2 is trained separately with `python train.py --mode anomaly_type`.
- Use `scripts/two_stage_predict.py` for actual two-stage inference.
- Stage 1 sends a sample to normal only when `p_normal > normal_threshold`; otherwise it sends the sample to abnormal.
- Stage 2 only runs for samples predicted abnormal by Stage 1. Treat this as **predicted positive only** routing: `stage2_evaluated = TP_abnormal + FP_normal`, not all true abnormal samples.
- Stage 2 cannot rescue runtime binary `FN` cases because Stage 1 normal predictions do not reach Stage 2.

## Reporting Rules

- Always report binary `FN`, `FP`, abnormal recall, and threshold behavior before discussing type accuracy.
- For labeled validation/test, break binary `FN/FP` down by `true_class`.
- Interpret Stage 2 type accuracy only on the subset where `stage2_ran=true` and the ground-truth abnormal type exists in the Stage 2 class list.
- Use defect-type error patterns to guide data/rendering/difficulty changes:
  - `mean_shift` FN: increase minimum gap or amplitude.
  - `spike` FN: increase spike height, width, or visibility.
  - `drift` FN: increase slope/span floor or reduce normal-overlap ambiguity.
  - `context` FN: improve fleet/context visibility and highlighted-member design.
  - normal FP: expand normal variation and inspect hard-normal Grad-CAM before increasing abnormal bias.

## Key Docs

- `docs/problem_setting.md`: binary-first framing and literature positioning.
- `docs/two_stage_workflow.md`: detailed training, inference, outputs, and FP/FN diagnosis workflow.
- `scripts/two_stage_predict.py`: implementation of binary gate plus second-stage type classifier.

## Continuous Recording

- When a run changes project knowledge, update repo docs, project skills, and memory in the same turn when feasible.
- For every real 2-stage run, record: binary run path, type run path, command, split, `normal_threshold`, output dir, `TN/FN/FP/TP`, `stage2_evaluated`, Stage 2 subset definition, Stage 2 type accuracy, major type confusion, and key FP/FN chart IDs.
- Always state that Stage 2 is evaluated on Stage 1 predicted positives. If there are normal FPs, record their Stage 2 fake type labels as FP contamination.
- If code behavior changes, record the changed file and verification command. Keep smoke-only dummy-checkpoint results separate from real performance evidence.
- Persistent knowledge locations:
  - `docs/summary.md` for project-level status and actual performance.
  - `docs/two_stage_workflow.md` for detailed commands, output interpretation, and operational rules.
  - `~/.codex/skills/anomaly-paper-evidence/references/current-project-state.md` for team-agent continuity.
  - `~/.codex/skills/anomaly-result-analysis/SKILL.md` for reusable analysis procedure.
  - Memory for short cross-session observations.
