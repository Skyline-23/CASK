param(
    [switch]$SyncReasoningGate = $true
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\Users\Skyline23\Downloads\triattention"
$pythonExe = "C:\Users\Skyline23\AppData\Local\Programs\Python\Python313\python.exe"

Push-Location $repoRoot
try {
    if ($SyncReasoningGate) {
        & $pythonExe scripts/sync_reasoning_gate_from_bg.py
    }

    & $pythonExe scripts/build_paper_figures.py
}
finally {
    Pop-Location
}
