Set-StrictMode -Version Latest
# Run the app inside the project's venv and capture stdout/stderr to app_start.log
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location -LiteralPath "$scriptRoot\cesar_assistant"

$pythonPath = "$scriptRoot\.venv\Scripts\python.exe"
if (-Not (Test-Path $pythonPath)) {
    Write-Error "Python executable not found at $pythonPath. Activate your venv or create .venv first."
    exit 1
}

$Env:PORT = '7861'
Write-Host "Starting app.py with: $pythonPath (PORT=$Env:PORT) — logs -> app_start.log"
& $pythonPath -u app.py *>&1 | Tee-Object -FilePath "app_start.log"