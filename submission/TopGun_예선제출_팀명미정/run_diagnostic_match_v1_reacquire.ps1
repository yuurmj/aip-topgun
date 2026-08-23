param(
    [int]$ServerPort = 9999
)

$ErrorActionPreference = "Stop"
$SubmissionRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = "C:\Users\Ran\anaconda3\envs\aip\python.exe"
$ViewerExe = "C:\Users\Ran\Desktop\2026\aip-topgun\01_Work\AIP_LIB\Windows\DogFightViewer.exe"
$CutoffExe = Join-Path $SubmissionRoot "unreal_bt_client.exe"
$RunId = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = Join-Path $SubmissionRoot ("diagnostic_logs\" + $RunId)
$PacketLog = Join-Path $LogDir "sac_packets.jsonl"
$SacConsoleLog = Join-Path $LogDir "sac_console.log"
$CutoffConsoleLog = Join-Path $LogDir "cutoff_console.log"
$CutoffErrorLog = Join-Path $LogDir "cutoff_error.log"

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "AIP Python not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $CutoffExe)) {
    throw "Cutoff client not found: $CutoffExe"
}

$Viewer = Get-Process DogFightViewer -ErrorAction SilentlyContinue
if (-not $Viewer) {
    Start-Process -FilePath $ViewerExe -WorkingDirectory (Split-Path -Parent $ViewerExe)
    Write-Host "DogFightViewer를 실행했습니다."
}

Write-Host ""
Write-Host "Viewer에서 포트 $ServerPort 확인 -> OpenServer -> 아래 두 AI 등록 확인 -> 시나리오 Start"
Write-Host "교전이 끝나면 이 창에서 Ctrl+C를 누르세요."
Write-Host "로그 폴더: $LogDir"
Write-Host ""

Write-Host "UDP 서버 127.0.0.1:$ServerPort 가 열리기를 기다리는 중..."
while (-not (netstat -ano -p udp | Select-String -Quiet "^\s*UDP\s+\S+:$ServerPort\s")) {
    Start-Sleep -Seconds 1
}
Write-Host "서버가 열렸습니다. SAC와 컷오프를 시작합니다."

$CutoffArgs = @(
    "--server-ip", "127.0.0.1",
    "--server-port", "$ServerPort",
    "--team-name", "EasyModeCutoff",
    "--action-repeat", "6"
)
$Cutoff = Start-Process -FilePath $CutoffExe -ArgumentList $CutoffArgs `
    -WorkingDirectory $SubmissionRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $CutoffConsoleLog -RedirectStandardError $CutoffErrorLog

$SacArgs = @(
    "run_unreal_inference.py",
    "--mode", "rl",
    "--bundle-dir", "model_variants\V1_REACQUIRE",
    "--team-name-file", "team_name.txt",
    "--server-ip", "127.0.0.1",
    "--server-port", "$ServerPort",
    "--action-repeat", "1",
    "--observation-repeat", "6",
    "--rebase-pair-positions",
    "--initial-pair-range-m", "762.0",
    "--plane-info-velocity-frame", "body",
    "--packet-monitor",
    "--packet-log", $PacketLog
)

try {
    Push-Location $SubmissionRoot
    # The inference client writes informational startup lines to stderr.
    # Windows PowerShell converts those lines into error records; with the
    # script-wide Stop preference that used to abort both clients immediately.
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PythonExe @SacArgs 2>&1 | Tee-Object -FilePath $SacConsoleLog
    $SacExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorActionPreference
    if ($SacExitCode -ne 0) {
        throw "SAC client exited with code $SacExitCode"
    }
}
finally {
    $ErrorActionPreference = "Continue"
    Pop-Location
    if ($Cutoff -and -not $Cutoff.HasExited) {
        Stop-Process -Id $Cutoff.Id -ErrorAction SilentlyContinue
    }
    Write-Host "진단 로그 저장 완료: $LogDir"
}
