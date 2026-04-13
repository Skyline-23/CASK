param(
    [Parameter(Mandatory = $true)]
    [string]$ConfigPath,
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonExe = "python",
    [string]$ModelPath = "experiments\models\Qwen3-8B",
    [string]$StatsFile = "cask\calibration\for_aime25_experiment\qwen3_8b.pt",
    [string]$OutputDir = "",
    [int]$WaitPid = 0,
    [int]$DefaultMaxRecords = 6
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Missing config file: $ConfigPath"
}

$resolvedRepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$resolvedConfigPath = (Resolve-Path -LiteralPath $ConfigPath).Path
if (-not $OutputDir) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputDir = Join-Path $resolvedRepoRoot "experiments\reports\replay_queue_$timestamp"
}
$resolvedOutputDir = [System.IO.Path]::GetFullPath($OutputDir)
$queueLog = Join-Path $resolvedOutputDir "queue.log"

New-Item -ItemType Directory -Force -Path $resolvedOutputDir | Out-Null

function Write-QueueLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $queueLog -Value $line
}

function Resolve-RepoPath {
    param([string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }
    return [System.IO.Path]::GetFullPath((Join-Path $resolvedRepoRoot $PathValue))
}

function Invoke-ReplayRun {
    param(
        [hashtable]$RunSpec
    )

    $label = [string]$RunSpec.label
    if (-not $label) {
        throw "Each run spec must include a non-empty label"
    }

    $referencePath = Resolve-RepoPath ([string]$RunSpec.reference)
    $method = [string]$RunSpec.method
    $budget = [int]$RunSpec.budget
    $maxRecords = if ($RunSpec.ContainsKey("maxRecords")) { [int]$RunSpec.maxRecords } else { $DefaultMaxRecords }
    $stdoutPath = Join-Path $resolvedOutputDir "$label.out.log"
    $stderrPath = Join-Path $resolvedOutputDir "$label.err.log"
    $jsonPath = Join-Path $resolvedOutputDir "$label.json"
    $csvPath = Join-Path $resolvedOutputDir "$label.csv"

    if ((Test-Path -LiteralPath $jsonPath) -and (Test-Path -LiteralPath $csvPath)) {
        Write-QueueLog "SKIP $label existing_outputs=true"
        return
    }

    $args = @(
        "scripts/replay_reference_fidelity.py",
        "--reference", $referencePath,
        "--model-path", $ModelPath,
        "--method", $method,
        "--budget", "$budget",
        "--triattention-stats-file", $StatsFile,
        "--max-records", "$maxRecords",
        "--attn-implementation", "sdpa",
        "--load-dtype", "bfloat16",
        "--json-output", $jsonPath,
        "--csv-output", $csvPath
    )

    if ($RunSpec.ContainsKey("extraArgs")) {
        foreach ($arg in $RunSpec.extraArgs) {
            $args += [string]$arg
        }
    }

    Write-QueueLog "START $label"
    $proc = Start-Process -FilePath $PythonExe `
        -ArgumentList $args `
        -WorkingDirectory $resolvedRepoRoot `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru
    $proc.WaitForExit()
    $proc.Refresh()
    $exitCode = $proc.ExitCode
    $jsonExists = Test-Path -LiteralPath $jsonPath
    $csvExists = Test-Path -LiteralPath $csvPath
    Write-QueueLog "END $label exit=$exitCode json=$jsonExists csv=$csvExists"
    if (-not ($jsonExists -and $csvExists)) {
        throw "Replay run failed for $label because expected outputs were not produced"
    }
}

if ($WaitPid -gt 0) {
    Write-QueueLog "WAIT pid=$WaitPid"
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 10
    }
    Write-QueueLog "WAIT_DONE pid=$WaitPid"
}

$configJson = Get-Content -LiteralPath $resolvedConfigPath -Raw | ConvertFrom-Json -AsHashtable
if ($configJson -isnot [System.Collections.IEnumerable]) {
    throw "Config must be a JSON array of replay run specs"
}

foreach ($runSpec in $configJson) {
    Invoke-ReplayRun -RunSpec $runSpec
}

Write-QueueLog "QUEUE_DONE"
