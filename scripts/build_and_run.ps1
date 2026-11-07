# Сборка Docker-образа с API и локальный запуск контейнера (Windows PowerShell).
# Перед сборкой модель должна быть обучена: dvc repro
$ErrorActionPreference = "Continue"
$Image = "credit-default-api:latest"
$Port = 8000
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path (Split-Path -Parent $Root) ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

if (-not (Test-Path "models/credit_default_model.pkl")) {
    Write-Host "Модель не найдена. Сначала выполните: dvc repro" -ForegroundColor Red; exit 1
}

# Docker Desktop должен быть запущен (иконка кита в трее, статус Engine running)
$info = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    # Docker Desktop на WSL2 слушает отдельный pipe — пробуем его контекст
    $info2 = docker --context desktop-linux info 2>&1
    if ($LASTEXITCODE -eq 0) {
        $env:DOCKER_CONTEXT = "desktop-linux"
        Write-Host "Используется контекст Docker: desktop-linux" -ForegroundColor Yellow
    } else {
        Write-Host "Docker недоступен. Ответ docker info:" -ForegroundColor Red
        $info | Select-Object -Last 8 | ForEach-Object { Write-Host "  $_" }
        $info2 | Select-Object -Last 4 | ForEach-Object { Write-Host "  $_" }
        Write-Host "Если Docker Desktop показывает 'Engine running', а здесь 'access denied' — добавьте пользователя в группу docker-users и перезайдите в Windows." -ForegroundColor Yellow
        exit 1
    }
}

docker build -t $Image .
if ($LASTEXITCODE -ne 0) { Write-Host "Ошибка docker build" -ForegroundColor Red; exit 1 }

docker rm -f credit-default-api *> $null
docker run -d --name credit-default-api -p "${Port}:8000" $Image
if ($LASTEXITCODE -ne 0) { Write-Host "Ошибка docker run" -ForegroundColor Red; exit 1 }

Write-Host "Ожидание запуска API..."
for ($i = 0; $i -lt 30; $i++) {
    try { Invoke-RestMethod "http://localhost:$Port/health" | Out-Null; break } catch { Start-Sleep -Seconds 1 }
}
& $Py scripts/check_api.py --url "http://localhost:$Port"
Write-Host "Swagger UI: http://localhost:$Port/docs  |  остановить: docker rm -f credit-default-api" -ForegroundColor Green
