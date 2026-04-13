param(
    [int]$WaitPid
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\Users\Skyline23\Downloads\triattention"
$pythonExe = "C:\Users\Skyline23\AppData\Local\Programs\Python\Python313\python.exe"
$statsFile = "cask\calibration\for_aime25_experiment\qwen3_8b.pt"
$logDir = Join-Path $repoRoot "experiments\reports\bg_replay_20260413"
$queueLog = Join-Path $logDir "queue.log"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Write-QueueLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $queueLog -Value $line
}

function Invoke-ReplayRun {
    param(
        [string]$Label,
        [string]$ReferencePath,
        [string]$Method,
        [int]$Budget
    )

    $stdoutPath = Join-Path $logDir "$Label.out.log"
    $stderrPath = Join-Path $logDir "$Label.err.log"
    $jsonPath = "experiments\reports\bg_replay_20260413\$Label.json"
    $csvPath = "experiments\reports\bg_replay_20260413\$Label.csv"

    if ((Test-Path (Join-Path $repoRoot $jsonPath)) -and (Test-Path (Join-Path $repoRoot $csvPath))) {
        Write-QueueLog "SKIP $Label existing_outputs=true"
        return
    }

    $args = @(
        "scripts/replay_reference_fidelity.py",
        "--reference", $ReferencePath,
        "--model-path", "experiments\models\Qwen3-8B",
        "--method", $Method,
        "--budget", "$Budget",
        "--triattention-stats-file", $statsFile,
        "--max-records", "6",
        "--attn-implementation", "sdpa",
        "--load-dtype", "bfloat16",
        "--json-output", $jsonPath,
        "--csv-output", $csvPath
    )

    Write-QueueLog "START $Label"
    $proc = Start-Process -FilePath $pythonExe `
        -ArgumentList $args `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru
    $proc.WaitForExit()
    $proc.Refresh()
    $exitCode = $proc.ExitCode
    $jsonExists = Test-Path (Join-Path $repoRoot $jsonPath)
    $csvExists = Test-Path (Join-Path $repoRoot $csvPath)
    Write-QueueLog "END $Label exit=$exitCode json=$jsonExists csv=$csvExists"
    if (-not ($jsonExists -and $csvExists)) {
        throw "Replay run failed for $Label because expected outputs were not produced"
    }
}

if ($WaitPid -gt 0) {
    Write-QueueLog "WAIT pid=$WaitPid"
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 10
    }
    Write-QueueLog "WAIT_DONE pid=$WaitPid"
}

$runs = @(
    @{
        Label = "aime24_cask384"
        Reference = "experiments\outputs\aime24\Qwen3-8B\sample1\fullkv\full_h100_aime24_ref6_fidelity_20260410\merged\merged.jsonl"
        Method = "cask"
        Budget = 384
    },
    @{
        Label = "aime25_tri384"
        Reference = "experiments\outputs\aime25\Qwen3-8B\sample1\fullkv\full_h100_aime25_ref6_fidelity_20260410\merged\merged.jsonl"
        Method = "triattention"
        Budget = 384
    },
    @{
        Label = "aime25_cask384"
        Reference = "experiments\outputs\aime25\Qwen3-8B\sample1\fullkv\full_h100_aime25_ref6_fidelity_20260410\merged\merged.jsonl"
        Method = "cask"
        Budget = 384
    }
)

foreach ($run in $runs) {
    Invoke-ReplayRun `
        -Label $run.Label `
        -ReferencePath $run.Reference `
        -Method $run.Method `
        -Budget $run.Budget
}

Write-QueueLog "QUEUE_DONE"
