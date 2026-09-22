param(
    [Parameter(Mandatory = $true)]
    [string]$RuntimeRoot,
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
    Push-Location -LiteralPath $RuntimeRoot
    & $python forecast_population.py 1> $StdoutPath 2> $StderrPath
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
    command = "python forecast_population.py"
    runtime_root = $RuntimeRoot
    python_executable = $python
    exit_code = $exitCode
    elapsed_seconds = [math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
    stdout_path = $StdoutPath
    stderr_path = $StderrPath
    stdout_bytes = if (Test-Path -LiteralPath $StdoutPath) { (Get-Item -LiteralPath $StdoutPath).Length } else { 0 }
    stderr_bytes = if (Test-Path -LiteralPath $StderrPath) { (Get-Item -LiteralPath $StderrPath).Length } else { 0 }
    resident_output = Test-Path -LiteralPath (Join-Path $RuntimeRoot "submission_resident.csv")
    registered_output = Test-Path -LiteralPath (Join-Path $RuntimeRoot "submission_registered.csv")
    error = $errorMessage
}
$status | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $StatusPath -Encoding UTF8
