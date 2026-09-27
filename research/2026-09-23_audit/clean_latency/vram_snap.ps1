param([string]$Tag = "snap")
# Снимок видеопамяти: nvidia-smi + счётчики WDDM по процессам (dedicated/shared) и по адаптеру.
$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"=== $Tag $ts"
$smi = & nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,pstate --format=csv,noheader
"nvidia-smi: $smi"
$ded = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples
$sh = (Get-Counter '\GPU Process Memory(*)\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples
$rows = @{}
foreach ($s in $ded) {
  if ($s.InstanceName -match 'pid_(\d+)_') { $p = [int]$Matches[1]; if (-not $rows.ContainsKey($p)) { $rows[$p] = @{d=0.0; s=0.0} }; $rows[$p].d += $s.CookedValue }
}
foreach ($s in $sh) {
  if ($s.InstanceName -match 'pid_(\d+)_') { $p = [int]$Matches[1]; if (-not $rows.ContainsKey($p)) { $rows[$p] = @{d=0.0; s=0.0} }; $rows[$p].s += $s.CookedValue }
}
$totD = 0.0; $totS = 0.0
$out = foreach ($p in $rows.Keys) {
  $n = (Get-Process -Id $p -ErrorAction SilentlyContinue).ProcessName
  $totD += $rows[$p].d; $totS += $rows[$p].s
  [pscustomobject]@{ pid = $p; name = $n; dedicated_MiB = [math]::Round($rows[$p].d / 1MB); shared_MiB = [math]::Round($rows[$p].s / 1MB) }
}
$out | Where-Object { $_.dedicated_MiB -ge 50 -or $_.shared_MiB -ge 20 -or $_.name -match 'python|llama|ollama' } | Sort-Object dedicated_MiB -Descending | Format-Table -AutoSize | Out-String -Width 200
"sum over processes: dedicated {0} MiB, shared {1} MiB" -f [math]::Round($totD / 1MB), [math]::Round($totS / 1MB)
$ad = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples
$as = (Get-Counter '\GPU Adapter Memory(*)\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples
foreach ($s in $ad) { "adapter {0} dedicated {1} MiB" -f $s.InstanceName, [math]::Round($s.CookedValue / 1MB) }
foreach ($s in $as) { "adapter {0} shared {1} MiB" -f $s.InstanceName, [math]::Round($s.CookedValue / 1MB) }
