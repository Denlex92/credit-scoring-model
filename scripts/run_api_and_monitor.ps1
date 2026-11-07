# Локальная проверка API без Docker + мониторинг дрифта через API.
# Результаты: reports/api_check.txt, reports/monitoring_log.txt, reports/drift_report*.json, reports/figures/drift_report*.png
$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:NO_PROXY = "127.0.0.1,localhost"   # запросы к локальному API не через VPN/прокси
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path (Split-Path -Parent $Root) ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }
$Url = "http://127.0.0.1:8000"
New-Item -ItemType Directory -Force -Path reports\figures | Out-Null

function Save([string]$file, [switch]$Append) {
    # запись вывода в файл в UTF-8 с одновременным выводом на экран
    begin { if (-not $Append) { Set-Content -Path $file -Value $null -Encoding UTF8 } }
    process { $s = "$_"; Write-Host $s; Add-Content -Path $file -Value $s -Encoding UTF8 }
}

if (-not (Test-Path "models/credit_default_model.pkl")) {
    Write-Host "Модель не найдена. Сначала выполните: dvc repro" -ForegroundColor Red; exit 1
}

Write-Host "==> Запуск API (uvicorn) на $Url" -ForegroundColor Cyan
$server = Start-Process -FilePath $Py -ArgumentList "-m", "uvicorn", "src.api.app:app", "--host", "127.0.0.1", "--port", "8000" `
    -PassThru -WindowStyle Hidden -RedirectStandardOutput reports\api_server.log -RedirectStandardError reports\api_server_err.log
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
    try { Invoke-RestMethod "$Url/health" | Out-Null; $ok = $true; break } catch { Start-Sleep -Seconds 1 }
}
if (-not $ok) {
    Write-Host "API не запустился, см. reports\api_server_err.log" -ForegroundColor Red
    Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue; exit 1
}

try {
    Write-Host "==> Проверка /health и /predict" -ForegroundColor Cyan
    & $Py scripts/check_api.py --url $Url 2>&1 | Save reports\api_check.txt

    Write-Host "==> Мониторинг через API: обычные данные" -ForegroundColor Cyan
    & $Py -m src.monitoring.monitor --url $Url 2>&1 | Save reports\monitoring_log.txt
    Copy-Item -Force reports\drift_report.json reports\drift_report_baseline.json
    Copy-Item -Force reports\figures\drift_report.png reports\figures\drift_report_baseline.png

    Write-Host "==> Мониторинг через API: имитация дрифта (--shift 0.6)" -ForegroundColor Cyan
    Add-Content -Path reports\monitoring_log.txt -Value "`n===== shift 0.6 =====" -Encoding UTF8
    & $Py -m src.monitoring.monitor --url $Url --shift 0.6 2>&1 | Save reports\monitoring_log.txt -Append
}
finally {
    Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
    Write-Host "API остановлен. Результаты в папке reports" -ForegroundColor Green
}
