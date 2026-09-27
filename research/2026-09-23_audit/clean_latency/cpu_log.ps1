param([string]$Out, [int]$Seconds = 900)
# Загрузка процессора раз в секунду (время<TAB>проценты). Через WMI-класс: имена счётчиков
# Get-Counter на русской Windows локализованы, и '\Processor(_Total)\...' в дочернем powershell не находится.
$inv = [System.Globalization.CultureInfo]::InvariantCulture
$end = (Get-Date).AddSeconds($Seconds)
while ((Get-Date) -lt $end) {
  $v = (Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'").PercentProcessorTime
  $line = [string]::Format($inv, "{0:HH:mm:ss.fff}`t{1}", (Get-Date), $v)
  Add-Content -Path $Out -Value $line -Encoding ascii
  Start-Sleep -Milliseconds 1000
}
