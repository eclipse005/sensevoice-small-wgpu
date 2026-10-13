# Peak process RSS for one fixture. Samples WorkingSet64 every 10 ms while the
# CLI runs; Refresh() is mandatory because .NET Process objects cache property
# values (a cached read reports the startup footprint).
param([string]$Fix = "30s_zh", [string]$Exe = "D:\SenseVoiceSmall\SenseVoiceSmall-wgpu\target\release\sensevoice.exe")
$p = Start-Process -FilePath $Exe -ArgumentList "--wav","D:/SenseVoiceSmall/fixtures/$Fix.wav","--device","P104" -NoNewWindow -PassThru -RedirectStandardOutput "$env:TEMP\sv_o.txt" -RedirectStandardError "$env:TEMP\sv_e.txt"
$max = 0
while (-not $p.HasExited) {
  try { $p.Refresh(); $ws = $p.WorkingSet64 } catch { $ws = 0 }
  if ($ws -gt $max) { $max = $ws }
  Start-Sleep -Milliseconds 10
}
$p.WaitForExit()
"{0} peakRSS={1:N0} MiB" -f $Fix, ($max / 1MB)
