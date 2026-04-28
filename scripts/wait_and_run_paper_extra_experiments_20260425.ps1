$ErrorActionPreference = 'Stop'

$Root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $Root

$Queue = Join-Path $Root 'validations\paper_extra_axis_queue.json'
$Summary = Join-Path $Root 'validations\paper_extra_axis_summary.json'
$Markdown = Join-Path $Root 'validations\paper_extra_axis_summary.md'
$ExpandQueue = Join-Path $Root 'validations\paper_extra_axis_expand_queue.json'
$DecisionMd = Join-Path $Root 'validations\paper_extra_axis_expansion_decision.md'
$CoreRefineQueue1 = Join-Path $Root 'validations\paper_core_axis_refine_round1_queue.json'
$CoreRefineMd1 = Join-Path $Root 'validations\paper_core_axis_refine_round1_decision.md'
$CoreRefineSummary1 = Join-Path $Root 'validations\paper_core_axis_refine_round1_summary.json'
$CoreRefineMarkdown1 = Join-Path $Root 'validations\paper_core_axis_refine_round1_summary.md'
$CoreRefineQueue2 = Join-Path $Root 'validations\paper_core_axis_refine_round2_queue.json'
$CoreRefineMd2 = Join-Path $Root 'validations\paper_core_axis_refine_round2_decision.md'
$CoreRefineSummary2 = Join-Path $Root 'validations\paper_core_axis_refine_round2_summary.json'
$CoreRefineMarkdown2 = Join-Path $Root 'validations\paper_core_axis_refine_round2_summary.md'
$ManualFollowupQueue = Join-Path $Root 'validations\paper_manual_axis_followup_queue.json'
$ManualFollowupSummary = Join-Path $Root 'validations\paper_manual_axis_followup_summary.json'
$ManualFollowupMarkdown = Join-Path $Root 'validations\paper_manual_axis_followup_summary.md'
$StatePath = Join-Path $Root 'validations\paper_extra_axis_state.json'
$Controller = Join-Path $Root 'scripts\adaptive_experiment_controller.py'
$Selector = Join-Path $Root 'scripts\select_paper_axis_expansions.py'
$CoreRefiner = Join-Path $Root 'scripts\select_paper_core_axis_refinements.py'
$PredTrendAnalyzer = Join-Path $Root 'scripts\analyze_prediction_trends.py'
$BaselineDeltaReporter = Join-Path $Root 'scripts\report_baseline_deltas.py'
$InstabilityCollector = Join-Path $Root 'scripts\collect_instability_cases.py'
$PredTrendLatestPrefix = Join-Path $Root 'validations\prediction_trend_latest'
$BaselineDeltaLatestPrefix = Join-Path $Root 'validations\baseline_delta_latest'
$SelectedConfig = 'logs\260412_044744_fresh0412_v11_n700_s42_F0.9920_R0.9920\data_config_used.yaml'

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
        core_refine_round1_queue = 'validations\paper_core_axis_refine_round1_queue.json'
        core_refine_round2_queue = 'validations\paper_core_axis_refine_round2_queue.json'
        manual_followup_queue = 'validations\paper_manual_axis_followup_queue.json'
        queue = 'validations\paper_extra_axis_queue.json'
        summary = 'validations\paper_extra_axis_summary.json'
        expansion_queue = 'validations\paper_extra_axis_expand_queue.json'
    }
    $payload | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $StatePath
}

function Get-ActiveMainPipeline {
    $selfPid = $PID
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.ProcessId -ne $selfPid -and
            $_.CommandLine -and
            (
                $_.CommandLine -like '*run_all_paper_experiments_20260425.ps1*' -or
                $_.CommandLine -like '*run_experiments_v11.py*' -or
                $_.CommandLine -like '*paper_followup_v11.py*' -or
                $_.CommandLine -like '*train.py*'
            ) -and
            $_.CommandLine -notlike '*wait_and_run_paper_extra_experiments_20260425.ps1*' -and
            $_.CommandLine -notlike '*Get-CimInstance*'
        } |
        Select-Object ProcessId,CommandLine
}

function Run-PredTrendAnalysis {
    param(
        [string]$Label
    )
    $safeLabel = ($Label -replace '[^A-Za-z0-9_-]', '_')
    $stagePrefix = Join-Path $Root ("validations\prediction_trend_{0}" -f $safeLabel)
    $stageReview = "${stagePrefix}_review"
    $latestReview = "${PredTrendLatestPrefix}_review"

    & python $PredTrendAnalyzer `
        --config $SelectedConfig `
        --candidate-prefix fresh0412_v11_ `
        --min-f1 0.99 `
        --out-prefix $stagePrefix `
        --review-k 20 `
        --report-label $Label
    if ($LASTEXITCODE -ne 0) {
        throw "prediction trend analysis failed with exit code $LASTEXITCODE"
    }

    foreach ($ext in @('json', 'csv', 'md')) {
        Copy-Item -LiteralPath ("{0}.{1}" -f $stagePrefix, $ext) -Destination ("{0}.{1}" -f $PredTrendLatestPrefix, $ext) -Force
    }
    if (Test-Path $latestReview) {
        Remove-Item -Recurse -Force $latestReview
    }
    if (Test-Path $stageReview) {
        Copy-Item -Recurse -Force $stageReview $latestReview
    }
}

function Run-BaselineDeltaReport {
    param(
        [string]$Label
    )
    $safeLabel = ($Label -replace '[^A-Za-z0-9_-]', '_')
    $stagePrefix = Join-Path $Root ("validations\baseline_delta_{0}" -f $safeLabel)

    & python $BaselineDeltaReporter `
        --baseline-candidate fresh0412_v11_n700 `
        --baseline-alias fresh0412_v11_n700_existing `
        --candidate-prefix fresh0412_v11_ `
        --out-prefix $stagePrefix `
        --min-complete 1
    if ($LASTEXITCODE -ne 0) {
        throw "baseline delta report failed with exit code $LASTEXITCODE"
    }

    foreach ($ext in @('json', 'csv', 'md')) {
        Copy-Item -LiteralPath ("{0}.{1}" -f $stagePrefix, $ext) -Destination ("{0}.{1}" -f $BaselineDeltaLatestPrefix, $ext) -Force
    }
}

function Run-InstabilityCollection {
    & python $InstabilityCollector --pattern '*fresh0412_v11*'
    if ($LASTEXITCODE -ne 0) {
        throw "instability case collection failed with exit code $LASTEXITCODE"
    }
}

try {
    Write-State 'waiting' 'main_axes' 'Waiting for the main paper runner to finish.'
    while ($true) {
        $active = @(Get-ActiveMainPipeline)
        if ($active.Count -eq 0) {
            break
        }
        $pids = ($active | ForEach-Object { $_.ProcessId }) -join ','
        Write-Host "[$((Get-Date).ToString('s'))] waiting for main pipeline pid(s): $pids"
        Start-Sleep -Seconds 60
    }

    $coreRefineRounds = @(
        @{
            Round = 1
            Queue = $CoreRefineQueue1
            DecisionMd = $CoreRefineMd1
            Summary = $CoreRefineSummary1
            Markdown = $CoreRefineMarkdown1
        },
        @{
            Round = 2
            Queue = $CoreRefineQueue2
            DecisionMd = $CoreRefineMd2
            Summary = $CoreRefineSummary2
            Markdown = $CoreRefineMarkdown2
        }
    )

    foreach ($round in $coreRefineRounds) {
        Write-State 'selecting' "core_refine_round_$($round.Round)" "Selecting adaptive core-axis refinement round $($round.Round)."
        & python $CoreRefiner `
            --prefix fresh0412 `
            --base-n 700 `
            --config $SelectedConfig `
            --out-queue $round.Queue `
            --decision-md $round.DecisionMd `
            --seeds 42 1 2
        if ($LASTEXITCODE -ne 0) {
            throw "core-axis refinement selector round $($round.Round) failed with exit code $LASTEXITCODE"
        }

        $coreQueue = Get-Content -Raw $round.Queue | ConvertFrom-Json
        $coreRunCount = @($coreQueue.runs).Count
        if ($coreRunCount -le 0) {
            Write-Host "[$((Get-Date).ToString('s'))] no new core-axis refinement runs selected for round $($round.Round)"
            break
        }

        Write-State 'running' "core_refine_round_$($round.Round)" "Running $coreRunCount adaptive core-axis refinement runs."
        & python $Controller `
            --queue $round.Queue `
            --summary $round.Summary `
            --markdown $round.Markdown `
            --target-min 5 `
            --target-max 15 `
            --stop-mode never `
            --candidate-min-runs-before-skip 0
        if ($LASTEXITCODE -ne 0) {
            throw "core-axis refinement controller round $($round.Round) failed with exit code $LASTEXITCODE"
        }

        Write-State 'analyzing' "prediction_trends_round_$($round.Round)" "Refreshing prediction-trend review after core refinement round $($round.Round)."
        Run-PredTrendAnalysis "core_refine_round_$($round.Round)"
        Run-BaselineDeltaReport "core_refine_round_$($round.Round)"
        Run-InstabilityCollection
    }

    Write-State 'running' 'extra_axes' 'Running additional paper-required axes.'
    & python $Controller `
        --queue $Queue `
        --summary $Summary `
        --markdown $Markdown `
        --target-min 5 `
        --target-max 15 `
        --stop-mode never `
        --candidate-min-runs-before-skip 0
    if ($LASTEXITCODE -ne 0) {
        throw "extra-axis controller failed with exit code $LASTEXITCODE"
    }

    Write-State 'analyzing' 'prediction_trends_extra_axes' 'Refreshing prediction-trend review after extra axes.'
    Run-PredTrendAnalysis 'extra_axes'
    Run-BaselineDeltaReport 'extra_axes'
    Run-InstabilityCollection

    Write-State 'selecting' 'expansion' 'Selecting promising candidates for seeds 3 and 4.'
    & python $Selector `
        --source-queue $Queue `
        --summary $Summary `
        --out-queue $ExpandQueue `
        --decision-md $DecisionMd `
        --ref-f1 0.9901 `
        --ref-fn 9.8 `
        --ref-fp 5.0 `
        --max-candidates 10 `
        --seeds 3 4
    if ($LASTEXITCODE -ne 0) {
        throw "expansion selector failed with exit code $LASTEXITCODE"
    }

    $expand = Get-Content -Raw $ExpandQueue | ConvertFrom-Json
    $runCount = @($expand.runs).Count
    if ($runCount -gt 0) {
        Write-State 'running' 'expanded_seeds' "Running $runCount expansion runs for paper-grade seeds."
        & python $Controller `
            --queue $ExpandQueue `
            --summary $Summary `
            --markdown $Markdown `
            --target-min 5 `
            --target-max 15 `
            --stop-mode never `
            --candidate-min-runs-before-skip 0
        if ($LASTEXITCODE -ne 0) {
            throw "expansion controller failed with exit code $LASTEXITCODE"
        }

        Write-State 'analyzing' 'prediction_trends_expanded_seeds' 'Refreshing prediction-trend review after expanded seeds.'
        Run-PredTrendAnalysis 'expanded_seeds'
        Run-BaselineDeltaReport 'expanded_seeds'
        Run-InstabilityCollection
    } else {
        Write-Host "[$((Get-Date).ToString('s'))] no expansion runs selected"
    }

    if (Test-Path $ManualFollowupQueue) {
        $manual = Get-Content -Raw $ManualFollowupQueue | ConvertFrom-Json
        $manualRunCount = @($manual.runs).Count
        if ($manualRunCount -gt 0) {
            Write-State 'running' 'manual_axis_followup' "Running $manualRunCount manual confirmation runs for neighborhood checks."
            & python $Controller `
                --queue $ManualFollowupQueue `
                --summary $ManualFollowupSummary `
                --markdown $ManualFollowupMarkdown `
                --target-min 5 `
                --target-max 15 `
                --stop-mode never `
                --candidate-min-runs-before-skip 0
            if ($LASTEXITCODE -ne 0) {
                throw "manual follow-up controller failed with exit code $LASTEXITCODE"
            }

            Write-State 'analyzing' 'prediction_trends_manual_followup' 'Refreshing prediction-trend review after manual follow-up runs.'
            Run-PredTrendAnalysis 'manual_followup'
            Run-BaselineDeltaReport 'manual_followup'
            Run-InstabilityCollection
        } else {
            Write-Host "[$((Get-Date).ToString('s'))] manual follow-up queue exists but has no runs"
        }
    }

    Write-State 'analyzing' 'prediction_trends_final' 'Refreshing final prediction-trend review.'
    Run-PredTrendAnalysis 'final'
    Run-BaselineDeltaReport 'final'
    Run-InstabilityCollection

    Write-State 'complete' 'done' 'Extra paper-axis pipeline completed.'
} catch {
    Write-State 'failed' 'error' $_.Exception.Message
    Write-Error $_
    exit 1
}
