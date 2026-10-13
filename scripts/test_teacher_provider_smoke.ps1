param(
    [Parameter(Mandatory = $true)]
    [string]$Model,
    [string]$TaskId = "21abae1_1",
    [string]$OutputName = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project Python was not found: $python"
}
if ([string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) {
    throw "OPENAI_API_KEY is not set in this PowerShell session."
}
if ([string]::IsNullOrWhiteSpace($env:OPENAI_BASE_URL)) {
    throw "OPENAI_BASE_URL is not set in this PowerShell session."
}
if ([string]::IsNullOrWhiteSpace($OutputName)) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputName = "provider_smoke_${timestamp}"
}
if ($OutputName -match '[\\/]' -or $OutputName -in @(".", "..")) {
    throw "OutputName must be a directory name, not a path: $OutputName"
}

$outputDir = Join-Path $projectRoot "artifacts\teacher_rollouts\$OutputName"
if (Test-Path -LiteralPath $outputDir) {
    throw "Smoke output already exists; choose another OutputName: $outputDir"
}

$active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq "python.exe" -and
    $_.CommandLine -match "appworld_agentic_rl\.cli collect-teacher"
}
if ($active) {
    $activeIds = ($active.ProcessId | Sort-Object) -join ", "
    throw "A teacher collector is already active (PID: $activeIds)."
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$env:NO_COLOR = "1"

$arguments = @(
    "-m", "appworld_agentic_rl.cli", "collect-teacher",
    "--model", $Model,
    "--appworld-root", (Join-Path $projectRoot "artifacts\appworld_rl_v1_root"),
    "--split", "appworld_rl_v1_teacher_sft",
    "--task-id", $TaskId,
    "--attempts-per-task", "1",
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
    "--request-retries", "0",
    "--provider-disable-parallel-tool-calls",
    "--output-dir", $outputDir
)

Write-Host "Running one-task Teacher provider smoke..."
Write-Host "Model:       $Model"
Write-Host "Task:        $TaskId"
Write-Host "Output:      $outputDir"
Write-Host "Parallel:    disabled in provider request and enforced locally"

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Teacher provider smoke exited with code $LASTEXITCODE"
}

$summaryPath = Join-Path $outputDir "summary.json"
$rawPath = Join-Path $outputDir "raw.jsonl"
$summary = Get-Content -Raw -LiteralPath $summaryPath | ConvertFrom-Json
$record = if (Test-Path -LiteralPath $rawPath) {
    Get-Content -LiteralPath $rawPath -First 1 | ConvertFrom-Json
} else {
    $null
}
$providerRequestCompleted = (
    $null -ne $record -and
    $record.rollout.end_reason -notin @("provider_error", "model_protocol_error")
)

[ordered]@{
    provider_request_completed = $providerRequestCompleted
    parallel_tool_calls_requested = $false
    task_id = $TaskId
    model = $Model
    rollout_id = if ($null -ne $record) { $record.rollout_id } else { $null }
    collection_status = if ($null -ne $record) { $record.collection_status } else { $null }
    end_reason = if ($null -ne $record) { $record.rollout.end_reason } else { $null }
    model_turns = if ($null -ne $record) { $record.rollout.model_turns } else { 0 }
    environment_steps = if ($null -ne $record) { $record.rollout.environment_steps } else { 0 }
    stopped_early = $summary.stopped_early
    stop_reason = $summary.stop_reason
    output_dir = $outputDir
} | ConvertTo-Json
