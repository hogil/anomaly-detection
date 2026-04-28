param(
    [ValidateSet("weights", "dataset", "train", "sweep", "perclass", "ablation", "summary", "paper", "all")]
    [string]$Stage = "dataset",
    [string]$Python = "python",
    [string]$Config = "dataset.yaml",
    [int]$Workers = 1,
    [switch]$SkipValidate,
    [int]$NormalRatio = 700,
    [int]$MaxPerClass = 0,
    [int]$Seed = 42,
    [string]$LogName = "company_ref",
    [string]$NamePrefix = "company_run",
    [int]$BaseN = 700,
    [int]$NumWorkers = 1,
    [int]$PrefetchFactor = 4,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Invoke-Step {
    param([string[]]$Args)
    Write-Host ""
    Write-Host ("+ " + ($Python + " " + ($Args -join " ")))
    & $Python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "command failed with exit code $LASTEXITCODE"
    }
}

switch ($Stage) {
    "weights" {
        Invoke-Step -Args (@("download.py") + $ExtraArgs)
    }

    "dataset" {
        Invoke-Step -Args @("generate_data.py", "--config", $Config, "--workers", "$Workers")
        Invoke-Step -Args @("generate_images.py", "--config", $Config, "--workers", "$Workers")
        if (-not $SkipValidate) {
            Invoke-Step -Args @("scripts/validate_dataset.py", "--config", $Config)
        }
    }

    "train" {
        Invoke-Step -Args (@(
            "train.py",
            "--config", $Config,
            "--seed", "$Seed",
            "--num_workers", "$NumWorkers",
            "--prefetch_factor", "$PrefetchFactor",
            "--log_dir", $LogName
        ) + $(if ($NormalRatio -gt 0) { @("--normal_ratio", "$NormalRatio") } else { @() }) `
          + $(if ($MaxPerClass -gt 0) { @("--max_per_class", "$MaxPerClass") } else { @() }) `
          + $ExtraArgs)
    }

    "sweep" {
        Invoke-Step -Args (@(
            "run_experiments_v11.py",
            "--groups", "sweep",
            "--config", $Config,
            "--num_workers", "$NumWorkers",
            "--name-prefix", $NamePrefix
        ) + $ExtraArgs)
    }

    "perclass" {
        Invoke-Step -Args (@(
            "run_experiments_v11.py",
            "--groups", "perclass",
            "--config", $Config,
            "--num_workers", "$NumWorkers",
            "--name-prefix", $NamePrefix
        ) + $ExtraArgs)
    }

    "ablation" {
        Invoke-Step -Args (@(
            "run_experiments_v11.py",
            "--groups", "lr", "gc", "wd", "smooth", "reg",
            "--config", $Config,
            "--base_n", "$BaseN",
            "--num_workers", "$NumWorkers",
            "--name-prefix", $NamePrefix
        ) + $ExtraArgs)
    }

    "summary" {
        Invoke-Step -Args (@(
            "run_experiments_v11.py",
            "--only-summary",
            "--config", $Config,
            "--base_n", "$BaseN",
            "--name-prefix", $NamePrefix
        ) + $ExtraArgs)
    }

    "paper" {
        Invoke-Step -Args (@(
            "scripts/paper_followup_v11.py",
            "--prefix", $NamePrefix,
            "--base-n", "$BaseN",
            "--num-workers", "$NumWorkers"
        ) + $ExtraArgs)
    }

    "all" {
        Invoke-Step -Args @("generate_data.py", "--config", $Config, "--workers", "$Workers")
        Invoke-Step -Args @("generate_images.py", "--config", $Config, "--workers", "$Workers")
        if (-not $SkipValidate) {
            Invoke-Step -Args @("scripts/validate_dataset.py", "--config", $Config)
        }
        Invoke-Step -Args (@(
            "run_experiments_v11.py",
            "--groups", "sweep",
            "--config", $Config,
            "--num_workers", "$NumWorkers",
            "--name-prefix", $NamePrefix
        ) + $ExtraArgs)
        Invoke-Step -Args (@(
            "run_experiments_v11.py",
            "--groups", "lr", "gc", "wd", "smooth", "reg",
            "--config", $Config,
            "--base_n", "$BaseN",
            "--num_workers", "$NumWorkers",
            "--name-prefix", $NamePrefix
        ) + $ExtraArgs)
        Invoke-Step -Args (@(
            "run_experiments_v11.py",
            "--groups", "perclass",
            "--config", $Config,
            "--num_workers", "$NumWorkers",
            "--name-prefix", $NamePrefix
        ) + $ExtraArgs)
        Invoke-Step -Args (@(
            "scripts/paper_followup_v11.py",
            "--prefix", $NamePrefix,
            "--base-n", "$BaseN",
            "--num-workers", "$NumWorkers"
        ) + $ExtraArgs)
    }
}
