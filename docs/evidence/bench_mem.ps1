param([string[]]$Fix = @('15s_en', '30s_zh', '90s_en', '90s_ja', '180s_en', '180s_zh'))
# Per-fixture peak RAM / devVRAM for the Rust GPU path, via run_mon.
# usage: powershell -ExecutionPolicy Bypass -File bench_mem.ps1 [-Fix 15s_en,90s_en]
$ErrorActionPreference = 'Stop'
$exe = 'D:\SenseVoiceSmall\SenseVoiceSmall-wgpu\target\release\sensevoice.exe'
$root = 'D:\SenseVoiceSmall'
foreach ($w in $Fix) {
    $log = Join-Path $env:TEMP "mem_$w.log"
    & "$root\_scratch\run_mon.ps1" -Exe $exe `
        -CliArgs @('--wav', "$root\fixtures\$w.wav", '--device', 'P104', '--check-gold', "$root\SenseVoiceSmall-wgpu\gold\$w.json") `
        -Label "mem_$w" -RamKillGB 14 -VramKillMiB 7600 -TimeoutSec 600 -PollMs 100 *>&1 | Out-File -Encoding utf8 $log
    $t = Get-Content $log -Encoding utf8
    $peakRam = ([string]($t | Select-String 'PEAK RAM' | Select-Object -First 1)).Trim()
    $peakVram = ([string]($t | Select-String 'PEAK devVRAM' | Select-Object -First 1)).Trim()
    $killed = ([string]($t | Select-String 'killed' | Select-Object -First 1)).Trim()
    $rtfx = ([string]($t | Select-String 'rtfx:' | Select-Object -Last 1)).Trim()
    Write-Output ("$w | $peakRam | $peakVram | $killed | $rtfx")
}
