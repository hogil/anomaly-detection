$ErrorActionPreference = 'Stop'

$Root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $Root

$Queue = Join-Path $Root 'validations\paper_manual_axis_followup_queue.json'
$Summary = Join-Path $Root 'validations\paper_manual_axis_followup_summary.json'
$Markdown = Join-Path $Root 'validations\paper_manual_axis_followup_summary.md'
$StatePath = Join-Path $Root 'validations\paper_manual_axis_followup_state.json'
$Controller = Join-Path $Root 'scripts\adaptive_experiment_controller.py'

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
        queue = 'validations\paper_manual_axis_followup_queue.json'
        summary = 'validations\paper_manual_axis_followup_summary.json'
        markdown = 'validations\paper_manual_axis_followup_summary.md'
        selected_reference = 'fresh0412_v11_n700_existing'
    }
    $payload | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $StatePath
}

function Get-ActivePaperWaiter {
    $selfPid = $PID
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.ProcessId -ne $selfPid -and
            $_.CommandLine -and
            (
                $_.CommandLine -like '*wait_and_run_paper_extra_experiments_20260425.ps1*' -or
                $_.CommandLine -like '*run_experiments_v11.py*' -or
                $_.CommandLine -like '*train.py*'
            ) -and
            $_.CommandLine -notlike '*wait_and_run_manual_axis_followup_20260426.ps1*' -and
            $_.CommandLine -notlike '*Get-CimInstance*'
        } |
        Select-Object ProcessId,CommandLine
}

try {
    Write-State 'waiting' 'paper_waiter' 'Waiting for the existing paper extra-axis waiter to clear before running manual follow-up.'
    while ($true) {
        $active = @(Get-ActivePaperWaiter)
        if ($active.Count -eq 0) {
            break
        }
        $pids = ($active | ForEach-Object { $_.ProcessId }) -join ','
        Write-Host "[$((Get-Date).ToString('s'))] waiting for paper waiter pid(s): $pids"
        Start-Sleep -Seconds 60
    }

    if (-not (Test-Path $Queue)) {
        throw "manual follow-up queue not found: $Queue"
    }
    $manual = Get-Content -Raw $Queue | ConvertFrom-Json
    $runCount = @($manual.runs).Count
    if ($runCount -le 0) {
        Write-State 'complete' 'no_runs' 'Manual follow-up queue is empty.'
        exit 0
    }

    Write-State 'running' 'manual_axis_followup' "Running $runCount manual follow-up runs."
    & python $Controller `
        --queue $Queue `
        --summary $Summary `
        --markdown $Markdown `
        --target-min 5 `
        --target-max 15 `
        --stop-mode never `
        --candidate-min-runs-before-skip 0
    if ($LASTEXITCODE -ne 0) {
        throw "manual follow-up controller failed with exit code $LASTEXITCODE"
    }

    Write-State 'complete' 'done' 'Manual axis follow-up queue completed.'
} catch {
    Write-State 'failed' 'error' $_.Exception.Message
    Write-Error $_
    exit 1
}
