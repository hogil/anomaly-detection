$ErrorActionPreference = 'Stop'

$Root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $Root

$Queue = Join-Path $Root 'validations\paper_strict_single_factor_queue.json'
$Summary = Join-Path $Root 'validations\paper_strict_single_factor_summary.json'
$Markdown = Join-Path $Root 'validations\paper_strict_single_factor_summary.md'
$Round2Queue = Join-Path $Root 'validations\paper_strict_single_factor_round2_queue.json'
$Round2Decision = Join-Path $Root 'validations\paper_strict_single_factor_round2_decision.md'
$Round2Summary = Join-Path $Root 'validations\paper_strict_single_factor_round2_summary.json'
$Round2Markdown = Join-Path $Root 'validations\paper_strict_single_factor_round2_summary.md'
$StatePath = Join-Path $Root 'validations\paper_strict_single_factor_state.json'
$ReportScript = Join-Path $Root 'scripts\generate_strict_one_factor_report.py'
$Controller = Join-Path $Root 'scripts\adaptive_experiment_controller.py'
$Selector = Join-Path $Root 'scripts\select_strict_single_factor_refinements.py'

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
        queue = 'validations\paper_strict_single_factor_queue.json'
        summary = 'validations\paper_strict_single_factor_summary.json'
        markdown = 'validations\paper_strict_single_factor_summary.md'
        round2_queue = 'validations\paper_strict_single_factor_round2_queue.json'
        round2_summary = 'validations\paper_strict_single_factor_round2_summary.json'
        report_markdown = 'validations\paper_strict_single_factor_summary.md'
        report_plots = 'validations\paper_strict_single_factor_plots'
        policy = 'baseline_fixed_one_factor_only'
    }
    $payload | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $StatePath
}

function Refresh-Report {
    & python $ReportScript | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "strict report refresh failed with exit code $LASTEXITCODE"
    }
}

try {
    if (-not (Test-Path $Queue)) {
        throw "strict single-factor queue not found: $Queue"
    }

    $payload = Get-Content -Raw $Queue | ConvertFrom-Json
    $runCount = @($payload.runs).Count
    if ($runCount -le 0) {
        Write-State 'complete' 'no_runs' 'Strict single-factor queue is empty.'
        exit 0
    }

    Write-State 'running' 'strict_single_factor' "Running $runCount baseline-fixed one-factor follow-up runs."
    & python $Controller `
        --queue $Queue `
        --summary $Summary `
        --markdown $Markdown `
        --target-min 5 `
        --target-max 15 `
        --stop-mode never `
        --candidate-min-runs-before-skip 0
    if ($LASTEXITCODE -ne 0) {
        throw "strict single-factor controller failed with exit code $LASTEXITCODE"
    }
    Refresh-Report

    Write-State 'selecting' 'strict_single_factor_round2' 'Selecting adaptive baseline-fixed one-factor round 2 levels.'
    & python $Selector `
        --summary $Summary `
        --out-queue $Round2Queue `
        --decision-md $Round2Decision
    if ($LASTEXITCODE -ne 0) {
        throw "strict single-factor selector failed with exit code $LASTEXITCODE"
    }

    if (Test-Path $Round2Queue) {
        $round2 = Get-Content -Raw $Round2Queue | ConvertFrom-Json
        $round2Count = @($round2.runs).Count
        if ($round2Count -gt 0) {
            Write-State 'running' 'strict_single_factor_round2' "Running $round2Count adaptive baseline-fixed one-factor round 2 runs."
            & python $Controller `
                --queue $Round2Queue `
                --summary $Round2Summary `
                --markdown $Round2Markdown `
                --target-min 5 `
                --target-max 15 `
                --stop-mode never `
                --candidate-min-runs-before-skip 0
            if ($LASTEXITCODE -ne 0) {
                throw "strict single-factor round 2 controller failed with exit code $LASTEXITCODE"
            }
            Refresh-Report
        }
    }

    Refresh-Report
    Write-State 'complete' 'done' 'Strict single-factor follow-up queue completed.'
} catch {
    Write-State 'failed' 'error' $_.Exception.Message
    Write-Error $_
    exit 1
}
