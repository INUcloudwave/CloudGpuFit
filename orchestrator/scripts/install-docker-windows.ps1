<#
.SYNOPSIS
  Installs the prerequisites for running the xmem-cpu container on Windows:
  WSL2 + Docker Desktop.

.NOTES
  Must be run from an ELEVATED (Administrator) PowerShell. Enabling WSL2
  requires a restart if it wasn't installed before, so this is a two-step
  process — run it, restart when it tells you to, then run it again.

  This script only automates what winget/wsl can do unattended. Docker
  Desktop's own first-run wizard (accepting its license, picking the WSL2
  backend) still needs one interactive click-through the first time you
  launch it.
#>

# Force UTF-8 for console output. Without this, Windows PowerShell renders
# this script's Korean text as mojibake on a non-UTF-8 system codepage (e.g.
# Korean Windows' default CP949) even though the source file itself is valid
# UTF-8 — that's a console rendering setting, separate from file encoding.
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch {}

# Deliberately NOT $ErrorActionPreference = "Stop": under Windows PowerShell
# 5.1, a native command's stderr output gets wrapped into a NativeCommandError
# and "Stop" promotes that to a terminating exception even for expected,
# non-fatal stderr chatter (e.g. `wsl --status` prints its "not installed"
# notice to stderr). That's what crashed the first version of this script.

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).
    IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Error "관리자 권한으로 다시 실행하세요: PowerShell을 '관리자 권한으로 실행'한 뒤 이 스크립트를 실행합니다."
    exit 1
}

function Test-Command($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

Write-Host "[1/2] WSL2 상태 확인 중..."
wsl --status *> $null
$wslExitCode = $LASTEXITCODE

if ($wslExitCode -ne 0) {
    Write-Host "WSL2가 설치되어 있지 않습니다. 설치를 진행합니다 (재부팅이 필요합니다)..."
    wsl --install --no-distribution
    Write-Host ""
    Write-Host "재부팅한 뒤 이 스크립트를 다시 실행해 Docker Desktop 설치를 이어가세요."
    exit 0
}
Write-Host "[1/2] WSL2 확인됨."

if (Test-Command "docker") {
    Write-Host "[2/2] Docker가 이미 설치되어 있습니다."
    docker --version
    exit 0
}

Write-Host "[2/2] Docker Desktop을 설치합니다..."
try {
    winget install --id Docker.DockerDesktop -e --accept-package-agreements --accept-source-agreements
} catch {
    Write-Error "Docker Desktop 설치 실패: $_"
    exit 1
}

Write-Host ""
Write-Host "설치 완료. Docker Desktop을 한 번 실행해 최초 설정(라이선스 동의, WSL2 backend 선택)을 마친 뒤,"
Write-Host "새 터미널에서 'docker run hello-world'로 정상 동작을 확인하세요."

