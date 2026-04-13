param(
    [string]$WaitQueueLog = "C:\Users\Skyline23\Downloads\triattention\experiments\reports\bg_replay_20260413\queue.log"
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\Users\Skyline23\Downloads\triattention"
$pythonExe = "C:\Users\Skyline23\AppData\Local\Programs\Python\Python313\python.exe"
$statsFile = "cask\calibration\for_aime25_experiment\qwen3_8b.pt"
$logDir = Join-Path $repoRoot "experiments\reports\bg_promptheavy_20260413"
$queueLog = Join-Path $logDir "queue.log"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Write-QueueLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $queueLog -Value $line
}

function Wait-ForReasoningQueue {
    param([string]$Path)
    Write-QueueLog "WAIT_REASONING_QUEUE path=$Path"
    while ($true) {
        if (Test-Path $Path) {
            $content = Get-Content $Path -ErrorAction SilentlyContinue
            if ($content -match "QUEUE_DONE") {
                break
            }
        }
        Start-Sleep -Seconds 15
    }
    Write-QueueLog "WAIT_REASONING_QUEUE_DONE path=$Path"
}

function Invoke-ReplayRun {
    param(
        [string]$Task,
        [string]$Method,
        [int]$Budget
    )

    $label = "{0}_{1}{2}" -f $Task, ($(if ($Method -eq "triattention") { "tri" } else { "cask" })), $Budget
    $stdoutPath = Join-Path $logDir "$label.out.log"
    $stderrPath = Join-Path $logDir "$label.err.log"
    $jsonPath = "experiments\reports\bg_promptheavy_20260413\$label.json"
    $csvPath = "experiments\reports\bg_promptheavy_20260413\$label.csv"
    $referencePath = "experiments\longbench_h100_refs\longbench\Qwen3-8B\runs\$Task\merged\merged.jsonl"

    if ((Test-Path (Join-Path $repoRoot $jsonPath)) -and (Test-Path (Join-Path $repoRoot $csvPath))) {
        Write-QueueLog "SKIP $label existing_outputs=true"
        return
    }

    $args = @(
        "scripts/replay_reference_fidelity.py",
        "--reference", $referencePath,
        "--model-path", "experiments\models\Qwen3-8B",
        "--method", $Method,
        "--budget", "$Budget",
        "--triattention-stats-file", $statsFile,
        "--max-records", "1",
        "--attn-implementation", "sdpa",
        "--load-dtype", "bfloat16",
        "--count-prompt-tokens", "true",
        "--slack-budget-trigger", "true",
        "--allow-prefill-compression", "false",
        "--json-output", $jsonPath,
        "--csv-output", $csvPath
    )

    Write-QueueLog "START $label"
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
    Write-QueueLog "END $label exit=$exitCode json=$jsonExists csv=$csvExists"
    if (-not ($jsonExists -and $csvExists)) {
        throw "Prompt-heavy replay run failed for $label because expected outputs were not produced"
    }
}

Wait-ForReasoningQueue -Path $WaitQueueLog

$runs = @(
    @{ Task = "multi_news"; Method = "triattention"; Budget = 256 },
    @{ Task = "multi_news"; Method = "cask"; Budget = 256 },
    @{ Task = "qasper"; Method = "triattention"; Budget = 256 },
    @{ Task = "qasper"; Method = "cask"; Budget = 256 },
    @{ Task = "hotpotqa"; Method = "triattention"; Budget = 256 },
    @{ Task = "hotpotqa"; Method = "cask"; Budget = 256 }
)

foreach ($run in $runs) {
    Invoke-ReplayRun -Task $run.Task -Method $run.Method -Budget $run.Budget
}

Write-QueueLog "QUEUE_DONE"
