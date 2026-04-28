# Paper Server Scripts

These wrappers split `scripts/run_paper_server_all.sh` into stage-level commands while keeping the pipeline logic in one place.

## Commands

```bash
bash scripts/sweeps_server/20_paper_all.sh
bash scripts/sweeps_server/21_paper_refcheck.sh
bash scripts/sweeps_server/22_paper_round1.sh
bash scripts/sweeps_server/23_paper_round1_after_gc.sh
bash scripts/sweeps_server/24_paper_round2.sh
bash scripts/sweeps_server/25_paper_post.sh
```

All options are passed through to `scripts/run_paper_server_all.sh`.

Useful resume/debug examples:

```bash
bash scripts/sweeps_server/23_paper_round1_after_gc.sh --max-launched 1
bash scripts/sweeps_server/22_paper_round1.sh --round1-start-after-candidate fresh0412_v11_gc50_n700 --max-launched 1
bash scripts/sweeps_server/24_paper_round2.sh
bash scripts/sweeps_server/25_paper_post.sh --candidate-prefix fresh0412_v11
```

Defaults are inherited from `run_paper_server_all.sh`: data/image generation workers `24`, training DataLoader workers `24`, and prefetch factor `4`.
