$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = if (Test-Path ".venv-ml\Scripts\python.exe") { ".venv-ml\Scripts\python.exe" } else { ".venv\Scripts\python.exe" }
if (-not (Test-Path $python)) { throw "No virtualenv found. Follow the install steps in README.md first." }
if (-not (Test-Path "web\dist\index.html")) { Push-Location web; npm install; npm run build; Pop-Location }
if (-not (Test-Path ".env")) { throw "Create .env from .env.example and add your keys." }

$port = if ($env:PORT) { $env:PORT } else { "8000" }
$bytes = New-Object byte[] 24
[Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
$env:ACCESS_TOKEN = [Convert]::ToBase64String($bytes).Replace("+", "A").Replace("/", "B").Replace("=", "")
$env:PORT = $port

$log = Join-Path $env:TEMP "cloudflared-$PID.log"
Remove-Item $log -ErrorAction SilentlyContinue
$tunnel = Start-Process cloudflared -ArgumentList "tunnel", "--url", "http://localhost:$port" -RedirectStandardError $log -WindowStyle Hidden -PassThru

try {
    $origin = $null
    for ($i = 0; $i -lt 60 -and -not $origin; $i++) {
        Start-Sleep -Seconds 1
        if (Test-Path $log) {
            $hit = Select-String -Path $log -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -First 1
            if ($hit) { $origin = $hit.Matches[0].Value }
        }
    }
    if (-not $origin) { throw "cloudflared did not publish a tunnel URL. Is it installed (winget install Cloudflare.cloudflared)?" }
    $env:PUBLIC_ORIGIN = $origin
    Write-Host ""
    Write-Host "Open this on the iPhone (Safari):"
    Write-Host "  $origin/?t=$($env:ACCESS_TOKEN)"
    Write-Host ""
    & $python -m libclaim.server
}
finally {
    if ($tunnel -and -not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id -Force }
}
