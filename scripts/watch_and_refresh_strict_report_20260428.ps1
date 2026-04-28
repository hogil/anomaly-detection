$ErrorActionPreference = 'Stop'

$Root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $Root

$StrictSummary = Join-Path $Root 'validations\paper_strict_single_factor_summary.json'
$Round2Summary = Join-Path $Root 'validations\paper_strict_single_factor_round2_summary.json'
$Generator = Join-Path $Root 'scripts\generate_strict_one_factor_report.py'
$LogPath = Join-Path $Root 'validations\paper_strict_report_watcher.log'

function Get-Stamps {
    $out = @{}
    foreach ($path in @($StrictSummary, $Round2Summary)) {
        if (Test-Path $path) {
            $item = Get-Item $path
            $out[$path] = "$($item.LastWriteTimeUtc.Ticks):$($item.Length)"
        } else {
            $out[$path] = 'missing'
        }
    }
    return $out
}

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date).ToString('s'), $Message
    Add-Content -Encoding UTF8 -Path $LogPath -Value $line
}

$last = Get-Stamps
Write-Log 'strict report watcher started'
try {
    & python $Generator | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Log 'initial report refresh complete'
    } else {
        Write-Log "initial report refresh failed with exit code $LASTEXITCODE"
    }
} catch {
    Write-Log "initial report refresh exception: $($_.Exception.Message)"
}

while ($true) {
    Start-Sleep -Seconds 15
    $now = Get-Stamps
    $changed = $false
    foreach ($key in $now.Keys) {
        if (-not $last.ContainsKey($key) -or $last[$key] -ne $now[$key]) {
            $changed = $true
            break
        }
    }
    if (-not $changed) {
        continue
    }
    $last = $now
    try {
        & python $Generator | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Log 'report refreshed'
        } else {
            Write-Log "report refresh failed with exit code $LASTEXITCODE"
        }
    } catch {
        Write-Log "report refresh exception: $($_.Exception.Message)"
    }
}
