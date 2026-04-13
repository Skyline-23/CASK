param(
    [switch]$SyncReasoningGate,
    [string]$ReportDir = "",
    [string[]]$Datasets = @("aime24_ref6", "aime25_ref6"),
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$resolvedRepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path

Push-Location $resolvedRepoRoot
try {
    if ($SyncReasoningGate) {
        $syncArgs = @("scripts/sync_reasoning_gate.py", "--datasets") + $Datasets
        if ($ReportDir) {
            $syncArgs += @("--report-dir", $ReportDir)
        }
        & $PythonExe @syncArgs
    }

    & $PythonExe scripts/build_paper_figures.py
}
finally {
    Pop-Location
}
