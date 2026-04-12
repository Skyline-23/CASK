$ErrorActionPreference = "Stop"

$RepoRoot = "C:\Users\AIUSER\Downloads\CASK"
$PythonExe = Join-Path $RepoRoot ".venv313\Scripts\python.exe"
$WatchLog = Join-Path $RepoRoot "experiments\frontier\h100_cutover_watch_20260412.log"

$RequiredMerged = @(
    "C:\Users\AIUSER\Downloads\CASK\experiments\outputs\aime25\Qwen3-8B\sample1\triattention\budget_384_h100_gate_qwen_aime25_6run_20260412\merged\merged.jsonl",
    "C:\Users\AIUSER\Downloads\CASK\experiments\outputs\aime24\Qwen3-8B\sample1\triattention\budget_384_h100_gate_qwen_aime24_6run_20260412\merged\merged.jsonl",
    "C:\Users\AIUSER\Downloads\CASK\experiments\outputs\aime24\DeepSeek-R1-Distill-Llama-8B\sample1\triattention\budget_384_dsg\merged\merged.jsonl"
)

$CutTags = @(
    "h100_gate_qwen_aime25_6run_20260412",
    "h100_gate_qwen_aime24_6run_20260412",
    "dsg"
)

$RelaunchTag = "q384cmp"

function Write-WatchLog {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $Message"
    Add-Content -LiteralPath $WatchLog -Value $line
}

function Test-CutoverReady {
    foreach ($path in $RequiredMerged) {
        if (-not (Test-Path -LiteralPath $path)) {
            return $false
        }
    }
    return $true
}

function Stop-TaggedProcesses {
    $targets = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" | Where-Object {
        $cmd = $_.CommandLine
        foreach ($tag in $CutTags) {
            if ($cmd -like "*$tag*") {
                return $true
            }
        }
        return $false
    }
    foreach ($proc in $targets) {
        try {
            Write-WatchLog ("Stopping PID {0}: {1}" -f $proc.ProcessId, $proc.CommandLine)
            Stop-Process -Id $proc.ProcessId -Force
        } catch {
            Write-WatchLog ("Failed to stop PID {0}: {1}" -f $proc.ProcessId, $_.Exception.Message)
        }
    }
}

function Start-ReducedComparison {
    $existing = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" | Where-Object {
        $_.CommandLine -like "*$RelaunchTag*"
    }
    if ($existing) {
        Write-WatchLog "Reduced comparison already running; skipping relaunch."
        return
    }

    $frontierDir = Join-Path $RepoRoot "experiments\frontier\Qwen3-8B\$RelaunchTag"
    New-Item -ItemType Directory -Force -Path $frontierDir | Out-Null
    $stdout = Join-Path $frontierDir "launcher.stdout.log"
    $stderr = Join-Path $frontierDir "launcher.stderr.log"
    $args = @(
        "scripts/run_cask_frontier.py",
        "--model", "Qwen3-8B",
        "--datasets", "aime24", "aime25",
        "--methods", "cask", "snapkv",
        "--budgets", "384",
        "--gpus", "0",
        "--num-shards", "1",
        "--num-samples", "1",
        "--max-examples", "10",
        "--job-parallel", "1",
        "--attn-implementation", "sdpa",
        "--load-dtype", "bfloat16",
        "--frontier-tag", $RelaunchTag
    )

    $proc = Start-Process -FilePath $PythonExe -ArgumentList $args -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    Write-WatchLog ("Started reduced comparison PID {0} with tag {1}" -f $proc.Id, $RelaunchTag)
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $WatchLog) | Out-Null
Write-WatchLog "Cutover watcher started."

while ($true) {
    if (Test-CutoverReady) {
        Write-WatchLog "All required triattention@384 merged outputs detected."
        Stop-TaggedProcesses
        Start-Sleep -Seconds 3
        Start-ReducedComparison
        Write-WatchLog "Cutover completed. Exiting watcher."
        break
    }

    $status = $RequiredMerged | ForEach-Object {
        "{0}={1}" -f $_, (Test-Path -LiteralPath $_)
    }
    Write-WatchLog ("Waiting for merged outputs: " + ($status -join "; "))
    Start-Sleep -Seconds 60
}
