# Median-of-N RTFx per fixture for the Rust GPU path + gold exit check.
# usage: powershell -ExecutionPolicy Bypass -File bench_median.ps1 [-N 3] [-Dev P104]
param([int]$N = 3, [string]$Dev = 'P104')
$ErrorActionPreference = 'Stop'
$exe = 'D:\SenseVoiceSmall\SenseVoiceSmall-wgpu\target\release\sensevoice.exe'
$root = 'D:\SenseVoiceSmall'
$fix = @('15s_en', '30s_zh', '90s_en', '90s_ja', '180s_en', '180s_zh')
$target = @{ '15s_en'=193.1; '30s_zh'=239.8; '90s_en'=192.8; '90s_ja'=213.4; '180s_en'=146.1; '180s_zh'=145.6 }
$audio = @{ '15s_en'=15.00; '30s_zh'=30.09; '90s_en'=90.00; '90s_ja'=89.33; '180s_en'=176.31; '180s_zh'=180.00 }
foreach ($w in $fix) {
    $totals = @()
    $ok = $true
    for ($i = 0; $i -lt $N; $i++) {
        $err = Join-Path $env:TEMP "med_$w.txt"
        $p = Start-Process -FilePath $exe `
            -ArgumentList @('--wav', "$root\fixtures\$w.wav", '--device', $Dev, '--check-gold', "$root\SenseVoiceSmall-wgpu\gold\$w.json") `
            -NoNewWindow -PassThru -RedirectStandardError $err -RedirectStandardOutput (Join-Path $env:TEMP "med_$w.out.txt")
        $p.WaitForExit()
        if ($p.ExitCode -ne 0) { $ok = $false }
        $t = Get-Content $err -Encoding utf8
        $fe = [double](([string]($t | Select-String 'frontend_ms:' | Select-Object -Last 1)) -replace '.*frontend_ms:\s*([0-9.]+).*','$1')
        $en = [double](([string]($t | Select-String 'encode_ms:' | Select-Object -Last 1)) -replace '.*encode_ms:\s*([0-9.]+).*','$1')
        $totals += ($fe + $en) / 1000.0
    }
    $sorted = $totals | Sort-Object
    $med = $sorted[[int]($sorted.Count / 2)]
    $rtfx = $audio[$w] / $med
    $mark = if ($rtfx -ge $target[$w]) { 'OK ' } else { 'BELOW' }
    Write-Output ("{0,-8} median {1,7:F1} ms  RTFx {2,6:F1}  target {3,6:F1}  {4}  gold={5}" -f $w, ($med * 1000), $rtfx, $target[$w], $mark, $ok)
}
