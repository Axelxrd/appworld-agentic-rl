param(
    [string]$OutputName = "sft_full_234x3_opencode_harness_v1"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$outputDir = Join-Path $projectRoot "artifacts\teacher_rollouts\$OutputName"
$logDir = Join-Path $projectRoot ".analysis\teacher_rollout_logs"
$processFile = Join-Path $logDir "$OutputName.process.json"
$rawPath = Join-Path $outputDir "raw.jsonl"
$summaryPath = Join-Path $outputDir "summary.json"

$metadata = if (Test-Path -LiteralPath $processFile) {
    Get-Content -Raw -LiteralPath $processFile | ConvertFrom-Json
} else {
    $null
}
$process = if ($null -ne $metadata) {
    Get-Process -Id $metadata.pid -ErrorAction SilentlyContinue
} else {
    $null
}
$completed = if (Test-Path -LiteralPath $rawPath) {
    (Get-Content -LiteralPath $rawPath | Measure-Object -Line).Lines
} else {
    0
}
$lastRecord = if ($completed -gt 0) {
    Get-Content -LiteralPath $rawPath -Tail 1 | ConvertFrom-Json
} else {
    $null
}
$summary = if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
    Get-Content -Raw -LiteralPath $summaryPath | ConvertFrom-Json
} else {
    $null
}

[ordered]@{
    running = $null -ne $process
    pid = if ($null -ne $metadata) { $metadata.pid } else { $null }
    completed_rollouts = $completed
    expected_rollouts = 702
    progress_percent = [math]::Round(100 * $completed / 702, 2)
    last_rollout_id = if ($null -ne $lastRecord) { $lastRecord.rollout_id } else { $null }
    last_status = if ($null -ne $lastRecord) { $lastRecord.collection_status } else { $null }
    last_end_reason = if ($null -ne $lastRecord) { $lastRecord.rollout.end_reason } else { $null }
    summary_available = $null -ne $summary
    stopped_early = if ($null -ne $summary) { $summary.stopped_early } else { $null }
    stop_reason = if ($null -ne $summary) { $summary.stop_reason } else { $null }
    pending_task_id = if ($null -ne $summary) { $summary.pending_task_id } else { $null }
    pending_attempt_index = if ($null -ne $summary) { $summary.pending_attempt_index } else { $null }
} | ConvertTo-Json
