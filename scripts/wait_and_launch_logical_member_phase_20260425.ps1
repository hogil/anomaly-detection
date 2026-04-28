$ErrorActionPreference = 'Stop'

$Root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $Root

$StatePath = Join-Path $Root 'validations\logical_member_phase_state.json'
$SelectedConfig = Join-Path $Root 'logs\260412_044744_fresh0412_v11_n700_s42_F0.9920_R0.9920\data_config_used.yaml'
$LogicalConfig = Join-Path $Root 'configs\datasets\logical_member_v11_20260425.yaml'
$LogicalSuffix = 'logical_v11'
$LogicalDataDir = 'data_logical_v11'
$LogicalImageDir = 'images_per_member_logical_v11'
$LogicalDisplayDir = 'display_per_member_logical_v11'
$LogicalScenariosCsv = Join-Path $Root 'data_per_member_logical_v11\scenarios_per_member.csv'
$LogicalLogDir = Join-Path $Root 'logs\fresh0412_v11_logical_member_baseline_s42'

function Write-State {
    param(
        [string]$Status,
        [string]$Stage,
        [string]$Message = ''
    )
    $payload = [ordered]@{
        updated_at = (Get-Date).ToString('s')
        status = $Status
        stage = $Stage
        message = $Message
        selected_reference = 'fresh0412_v11_n700_existing'
        selected_data_config = 'logs\260412_044744_fresh0412_v11_n700_s42_F0.9920_R0.9920\data_config_used.yaml'
        logical_config = 'configs\datasets\logical_member_v11_20260425.yaml'
        logical_data_dir = 'data_logical_v11'
        logical_image_dir = 'images_per_member_logical_v11'
        logical_scenarios_csv = 'data_per_member_logical_v11\scenarios_per_member.csv'
        logical_log_dir = 'logs\fresh0412_v11_logical_member_baseline_s42'
    }
    $payload | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $StatePath
}

function Get-ActivePaperPipeline {
    $selfPid = $PID
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.ProcessId -ne $selfPid -and
            $_.CommandLine -and
            (
                $_.CommandLine -like '*run_all_paper_experiments_20260425.ps1*' -or
                $_.CommandLine -like '*run_experiments_v11.py*' -or
                $_.CommandLine -like '*paper_followup_v11.py*' -or
                $_.CommandLine -like '*wait_and_run_paper_extra_experiments_20260425.ps1*' -or
                $_.CommandLine -like '*wait_and_run_manual_axis_followup_20260426.ps1*' -or
                $_.CommandLine -like '*train.py*'
            ) -and
            $_.CommandLine -notlike '*wait_and_launch_logical_member_phase_20260425.ps1*' -and
            $_.CommandLine -notlike '*Get-CimInstance*'
        } |
        Select-Object ProcessId, CommandLine
}

function New-LogicalConfig {
    Write-State 'preparing' 'logical_config' 'Creating logical member dataset config.'
    @"
import pathlib
import yaml

src = pathlib.Path(r"$SelectedConfig")
dst = pathlib.Path(r"$LogicalConfig")

with src.open('r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

cfg.setdefault('dataset', {})
cfg['dataset']['all_context_columns_per_chart'] = True
cfg['dataset']['version'] = str(cfg['dataset'].get('version', 'v11')) + '-logical-member'
cfg.setdefault('output', {})
cfg['output']['data_dir'] = '$LogicalDataDir'
cfg['output']['image_dir'] = '$LogicalImageDir'
cfg['output']['display_dir'] = '$LogicalDisplayDir'

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open('w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print(dst)
"@ | python -
    if ($LASTEXITCODE -ne 0) {
        throw "logical config creation failed with exit code $LASTEXITCODE"
    }
}

try {
    Write-State 'waiting' 'paper_pipeline' 'Waiting for the main paper and extra-axis pipeline to finish.'
    while ($true) {
        $active = @(Get-ActivePaperPipeline)
        if ($active.Count -eq 0) {
            break
        }
        $pids = ($active | ForEach-Object { $_.ProcessId }) -join ','
        Write-Host "[$((Get-Date).ToString('s'))] waiting for paper pipeline pid(s): $pids"
        Start-Sleep -Seconds 60
    }

    New-LogicalConfig

    Write-State 'running' 'logical_generate_data' 'Generating logical member dataset with all context columns per family.'
    & python generate_data.py --config $LogicalConfig --workers 0 --all_context_columns_per_chart
    if ($LASTEXITCODE -ne 0) {
        throw "logical generate_data failed with exit code $LASTEXITCODE"
    }

    Write-State 'running' 'logical_per_member_expand' 'Rendering per-member logical images and expanded scenarios.'
    & python scripts\generate_per_member_images.py --config $LogicalConfig --suffix $LogicalSuffix --workers 0
    if ($LASTEXITCODE -ne 0) {
        throw "logical per-member rendering failed with exit code $LASTEXITCODE"
    }

    Write-State 'running' 'logical_baseline_train' 'Launching logical member baseline seed 42 training.'
    & python -u train.py `
        --log_dir $LogicalLogDir `
        --config $LogicalConfig `
        --scenarios_csv $LogicalScenariosCsv `
        --num_workers 0 `
        --mode binary `
        --epochs 20 `
        --patience 5 `
        --smooth_window 3 `
        --smooth_method median `
        --grad_clip 1.0 `
        --ema_decay 0.0 `
        --lr_backbone 2e-5 `
        --lr_head 2e-4 `
        --warmup_epochs 5 `
        --weight_decay 0.02 `
        --seed 42 `
        --train_sampler balanced_binary
    if ($LASTEXITCODE -ne 0) {
        throw "logical baseline train failed with exit code $LASTEXITCODE"
    }

    Write-State 'complete' 'logical_baseline_done' 'Logical member baseline training completed. Target-flip and shortcut-leak evaluation remain the next phase.'
} catch {
    Write-State 'failed' 'error' $_.Exception.Message
    Write-Error $_
    exit 1
}
