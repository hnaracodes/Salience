# Threadmind end-to-end pipeline — dot-source for Invoke-ThreadmindPipeline
# Usage:
#   . .\scripts\run_threadmind_pipeline.ps1
#   Invoke-ThreadmindPipeline
#   Invoke-ThreadmindPipeline -CreateBaseline
#   Invoke-ThreadmindPipeline -Explore -CreateBaseline

$ErrorActionPreference = "Stop"

function Invoke-ThreadmindPipeline {
    [CmdletBinding()]
    param(
        [string]$SessionId,
        [switch]$CreateBaseline,
        [switch]$Explore,
        [switch]$NoServe,
        [int]$Port = 8780,
        [string]$NormId = "synthetic_bootstrap_v1",
        [ValidateSet("all", "capture", "tribe", "dual_track", "heatmaps", "analyze", "narrative", "export_viewer")]
        [string]$FromStage = "all",
        [switch]$UniformHeatmap,
        [switch]$SkipNarrative,
        [string]$Provider
    )

    $Root = Split-Path $PSScriptRoot -Parent
    $Py = Join-Path $Root ".venv\Scripts\python.exe"
    if (-not (Test-Path $Py)) {
        $Py = "python"
    }

    $args = @(
        (Join-Path $Root "scripts\run_threadmind_pipeline.py"),
        "--from-stage", $FromStage,
        "--norm-id", $NormId,
        "--port", $Port
    )
    if ($SessionId) { $args += @("--session-id", $SessionId) }
    if ($CreateBaseline) { $args += "--create-baseline" }
    if ($Explore) { $args += "--explore" }
    if ($NoServe) { $args += "--no-serve" }
    if ($UniformHeatmap) { $args += "--uniform-heatmap" }
    if ($SkipNarrative) { $args += "--skip-narrative" }
    if ($Provider) { $args += @("--provider", $Provider) }

    Push-Location $Root
    try {
        & $Py @args
    } finally {
        Pop-Location
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    Invoke-ThreadmindPipeline @args
}
