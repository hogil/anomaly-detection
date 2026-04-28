$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$python = (Get-Command python).Source
$selectedConfig = 'logs\260412_044744_fresh0412_v11_n700_s42_F0.9920_R0.9920\data_config_used.yaml'
$prefix = 'fresh0412'
$baseN = '700'
$stamp = Get-Date -Format 'yyyy-MM-ddTHH:mm:ss'

$state = [ordered]@{
  started_at = $stamp
  selected_reference = 'fresh0412_v11_n700_existing'
  selected_config = $selectedConfig
  prefix = $prefix
  base_n = [int]$baseN
  status = 'running'
  stages = @(
    'main_axes: sweep lr gc wd smooth reg rescue',
    'paper_followup: build table and launch combo if winners exist',
    'perclass: max_per_class saturation sweep'
  )
}
$state | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath 'validations\paper_all_experiments_state.json' -Encoding UTF8

function Write-State($status, $stage) {
  $payload = [ordered]@{
    updated_at = (Get-Date).ToString('s')
    selected_reference = 'fresh0412_v11_n700_existing'
    selected_config = $selectedConfig
    prefix = $prefix
    base_n = [int]$baseN
    status = $status
    current_stage = $stage
  }
  $payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath 'validations\paper_all_experiments_state.json' -Encoding UTF8
}

Write-State 'running' 'main_axes'
& $python -u run_experiments_v11.py `
  --config $selectedConfig `
  --groups sweep lr gc wd smooth reg rescue `
  --base_n $baseN `
  --name-prefix $prefix `
  --num_workers 0 `
  --gpus 1

Write-State 'running' 'paper_followup'
& $python -u scripts\paper_followup_v11.py `
  --prefix $prefix `
  --base-n $baseN `
  --num-workers 0 `
  --launch-combo

Write-State 'running' 'perclass'
& $python -u run_experiments_v11.py `
  --config $selectedConfig `
  --groups perclass `
  --base_n $baseN `
  --name-prefix $prefix `
  --num_workers 0 `
  --gpus 1

Write-State 'complete' 'done'
