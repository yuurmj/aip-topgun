param(
    [int]$ReplayCount = 10
)

$ErrorActionPreference = "Stop"

$releaseRoot = "C:\dev\aip-topgun\01_Work\AIP_LIB\DogFightEnv\Release_260708"
$aipPython = "C:\Users\shwsr\anaconda3\envs\aip\python.exe"
$bundleDir = "artifacts\models\team01\ppo_mlp_v1"

Set-Location -LiteralPath $releaseRoot

Write-Output ("PPO_REPLAY_BATCH_START=" + (Get-Date).ToString("o"))

& $aipPython run_local_dogfight.py `
    --ownship-backend rl `
    --ownship-bundle-dir $bundleDir `
    --target-backend bt `
    --target-bt-dll AIP_BASE_target.dll `
    --observation-mode tactical16 `
    --max-engage-time 60 `
    --episode-step-limit 3600 `
    --min-altitude 300 `
    --episodes $ReplayCount `
    --save-log

if ($LASTEXITCODE -ne 0) {
    throw "Replay batch failed with exit code $LASTEXITCODE"
}

Write-Output ("PPO_REPLAY_BATCH_END=" + (Get-Date).ToString("o"))
