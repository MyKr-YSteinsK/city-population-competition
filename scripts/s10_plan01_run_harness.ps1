param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,
    [Parameter(Mandatory = $true)]
    [string]$ModulePath,
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [Parameter(Mandatory = $true)]
    [string]$StdoutPath,
    [Parameter(Mandatory = $true)]
    [string]$StderrPath,
    [Parameter(Mandatory = $true)]
    [string]$StatusPath
)

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$python = (Get-Command python).Source
$exitCode = $null
$errorMessage = $null
try {
    Push-Location -LiteralPath $RepoRoot
    & $python "scripts\s10_plan01_forensic_recompute.py" --module-path $ModulePath --repo-root $RepoRoot --output-dir $OutputDir 1> $StdoutPath 2> $StderrPath
    $exitCode = $LASTEXITCODE
}
catch {
    $errorMessage = $_.Exception.ToString()
    $exitCode = 1
}
finally {
    Pop-Location
    $stopwatch.Stop()
}

$status = [ordered]@{
    command = "python scripts\\s10_plan01_forensic_recompute.py --module-path <runtime>/forecast_population.py --repo-root <repo> --output-dir <repo>/reports/S10_HANDOFFS/S10-Plan01/forensics"
    repo_root = $RepoRoot
    module_path = $ModulePath
    python_executable = $python
    exit_code = $exitCode
    elapsed_seconds = [math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
    stdout_path = $StdoutPath
    stderr_path = $StderrPath
    stdout_bytes = if (Test-Path -LiteralPath $StdoutPath) { (Get-Item -LiteralPath $StdoutPath).Length } else { 0 }
    stderr_bytes = if (Test-Path -LiteralPath $StderrPath) { (Get-Item -LiteralPath $StderrPath).Length } else { 0 }
    error = $errorMessage
}
$status | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $StatusPath -Encoding UTF8
