param(
    [string]$Rscript = 'Rscript',
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    $steps = @(
        @{Runner=$Rscript; Script='scripts/01_read_data.R'},
        @{Runner=$Python; Script='scripts/02_clustering.py'},
        @{Runner=$Rscript; Script='scripts/03_clustering_visualization.R'},
        @{Runner=$Rscript; Script='scripts/04_network_analysis.R'},
        @{Runner=$Python; Script='scripts/05_glmy_homology.py'}
    )
    foreach ($step in $steps) {
        $timer = [System.Diagnostics.Stopwatch]::StartNew()
        Write-Host "Running $($step.Script)"
        & $step.Runner $step.Script
        if ($LASTEXITCODE -ne 0) { throw "Step failed: $($step.Script) (exit $LASTEXITCODE)" }
        $timer.Stop()
        Write-Host ("Completed {0} in {1:F2} seconds" -f $step.Script, $timer.Elapsed.TotalSeconds)
    }
} finally { Pop-Location }
