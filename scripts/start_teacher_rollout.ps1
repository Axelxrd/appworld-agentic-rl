param(
    [string]$Model = "deepseek-v4-pro",
    [string]$OutputName = "sft_full_234x3_opencode_harness_v1",
    [switch]$DisableParallelToolCalls
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$outputDir = Join-Path $projectRoot "artifacts\teacher_rollouts\$OutputName"
$logDir = Join-Path $projectRoot ".analysis\teacher_rollout_logs"
$stdoutLog = Join-Path $logDir "$OutputName.stdout.log"
$stderrLog = Join-Path $logDir "$OutputName.stderr.log"
$processFile = Join-Path $logDir "$OutputName.process.json"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project Python was not found: $python"
}
if ([string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) {
    throw "OPENAI_API_KEY is not set in this PowerShell session."
}
if ([string]::IsNullOrWhiteSpace($env:OPENAI_BASE_URL)) {
    throw "OPENAI_BASE_URL is not set in this PowerShell session."
}

$active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq "python.exe" -and
    $_.CommandLine -match "appworld_agentic_rl\.cli collect-teacher"
}
if ($active) {
    $activeIds = ($active.ProcessId | Sort-Object) -join ", "
    throw "A teacher collector is already active (PID: $activeIds)."
}

$resumeArguments = @()
if (Test-Path -LiteralPath $outputDir) {
    $configPath = Join-Path $outputDir "run_config.json"
    if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
        throw "Output exists without run_config.json; refusing unsafe resume: $outputDir"
    }
    $resumeArguments = @("--resume")
}

New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$env:NO_COLOR = "1"

$arguments = @(
    "-m", "appworld_agentic_rl.cli", "collect-teacher",
    "--model", $Model,
    "--appworld-root", (Join-Path $projectRoot "artifacts\appworld_rl_v1_root"),
    "--split", "appworld_rl_v1_teacher_sft",
    "--attempts-per-task", "3",
    "--max-curated-per-task", "2",
    "--temperature", "0.5",
    "--max-tokens", "2048",
    "--max-model-turns", "40",
    "--max-steps", "50",
    "--bc-guard-mode", "enforce",
    "--max-action-attempts", "80",
    "--invalid-action-limit", "6",
    "--max-identical-action-repeats", "1",
    "--max-context-chars", "131072",
    "--max-observation-chars", "12000",
    "--context-compaction-enable",
    "--context-compaction-trigger-chars", "131072",
    "--context-compaction-target-chars", "98304",
    "--context-preserve-recent-groups", "4",
    "--output-dir", $outputDir
) + $resumeArguments
if ($DisableParallelToolCalls) {
    $arguments += "--provider-disable-parallel-tool-calls"
}

$process = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru

$metadata = [ordered]@{
    pid = $process.Id
    started_at = [DateTimeOffset]::Now.ToString("o")
    model = $Model
    output_dir = $outputDir
    expected_tasks = 234
    attempts_per_task = 3
    expected_rollouts = 702
    parallel_tool_calls_requested = -not $DisableParallelToolCalls
    stdout_log = $stdoutLog
    stderr_log = $stderrLog
}
$metadata | ConvertTo-Json | Set-Content -LiteralPath $processFile -Encoding UTF8
$metadata | ConvertTo-Json
