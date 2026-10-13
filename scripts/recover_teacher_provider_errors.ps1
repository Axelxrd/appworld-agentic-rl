param(
    [string]$SourceName = "sft_full_234x3_gpt56terra_harness_v1",
    [string]$OutputName = "sft_full_234x3_gpt56terra_harness_v1_provider_recovery_v1"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$rolloutRoot = Join-Path $projectRoot "artifacts\teacher_rollouts"
$sourceDir = Join-Path $rolloutRoot $SourceName
$outputDir = Join-Path $rolloutRoot $OutputName
$logDir = Join-Path $projectRoot ".analysis\teacher_rollout_logs"
$stdoutLog = Join-Path $logDir "$OutputName.stdout.log"
$stderrLog = Join-Path $logDir "$OutputName.stderr.log"
$processFile = Join-Path $logDir "$OutputName.process.json"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project Python was not found: $python"
}
if (-not (Test-Path -LiteralPath (Join-Path $sourceDir "raw.jsonl") -PathType Leaf)) {
    throw "Source raw.jsonl was not found: $sourceDir"
}
if ([string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) {
    throw "OPENAI_API_KEY is not set in this PowerShell session."
}

$active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq "python.exe" -and (
        $_.CommandLine -match "appworld_agentic_rl\.cli collect-teacher" -or
        $_.CommandLine -match "appworld_agentic_rl\.cli recover-teacher-provider-errors"
    )
}
if ($active) {
    $activeIds = ($active.ProcessId | Sort-Object) -join ", "
    throw "A teacher collection/recovery process is already active (PID: $activeIds)."
}

$resumeArguments = @()
if (Test-Path -LiteralPath $outputDir) {
    $configPath = Join-Path $outputDir "run_config.json"
    if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
        throw "Recovery output exists without run_config.json; refusing unsafe resume: $outputDir"
    }
    $resumeArguments = @("--resume")
}

New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$env:NO_COLOR = "1"

$arguments = @(
    "-m", "appworld_agentic_rl.cli", "recover-teacher-provider-errors",
    "--source-dir", $sourceDir,
    "--output-dir", $outputDir,
    "--appworld-root", (Join-Path $projectRoot "artifacts\appworld_rl_v1_root"),
    "--experiment-name", "appworld_teacher_provider_recovery_v1",
    "--max-steps", "50"
) + $resumeArguments

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
    source_dir = $sourceDir
    output_dir = $outputDir
    selection = "rollout.end_reason == provider_error"
    stdout_log = $stdoutLog
    stderr_log = $stderrLog
}
$metadata | ConvertTo-Json | Set-Content -LiteralPath $processFile -Encoding UTF8
$metadata | ConvertTo-Json
