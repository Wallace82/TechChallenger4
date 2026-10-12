# Script de inicializacao da Guardia AI
param(
    [string]$Mode = "docker",
    [int]$Port = 7860,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "         Guardia AI -- Assistente Clinico Hospitalar      " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Modo selecionado: $Mode" -ForegroundColor Yellow
Write-Host "Porta: $Port" -ForegroundColor Yellow
Write-Host ""

$AppUrl = "http://localhost:$Port"

if ($Mode -eq 'docker') {
    # 1. Verifica se o Docker CLI esta instalado
    $DockerCmd = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $DockerCmd) {
        Write-Error "Docker nao foi encontrado no PATH do sistema. Instale o Docker Desktop ou utilize '.\start.ps1 -Mode local'."
        exit 1
    }

    # 2. Verifica se o Docker daemon esta ativo
    Write-Host "Verificando se o Docker esta em execucao..." -ForegroundColor Cyan
    & docker info >$null 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "O Docker esta instalado, mas o daemon/Docker Desktop nao esta em execucao. Inicie o Docker Desktop e tente novamente."
        exit 1
    }

    Write-Host "Iniciando conteiner via Docker Compose..." -ForegroundColor Green
    Write-Host "Acesse a aplicacao em: $AppUrl" -ForegroundColor White
    Write-Host "Pressione Ctrl+C para encerrar o conteiner." -ForegroundColor Gray
    Write-Host ""

    # Dispara a abertura do browser em background apos delay para subida do conteiner
    if (-not $NoBrowser) {
        Start-Job -ScriptBlock {
            param($url)
            Start-Sleep -Seconds 4
            Start-Process $url
        } -ArgumentList $AppUrl | Out-Null
    }

    # Executa docker compose
    & docker compose up --build
}
elseif ($Mode -eq 'local') {
    # Verifica Python
    $PythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCmd) {
        Write-Error "Interpretador Python nao encontrado no PATH."
        exit 1
    }

    Write-Host "Configurando variaveis de ambiente locais..." -ForegroundColor Cyan
    $env:PERFIL_EXECUCAO = "demo-cpu"
    $env:ML_RISCO_HABILITADO = "1"
    $env:PORT = "$Port"
    $env:HOSPITAL_DB_PATH = Join-Path $ScriptDir "artifacts\demo\hospital_ui.db"
    $env:ARTIFACTS_DIR = Join-Path $ScriptDir "artifacts"

    Write-Host "Iniciando aplicacao local (scripts/app.py)..." -ForegroundColor Green
    Write-Host "Acesse a aplicacao em: $AppUrl" -ForegroundColor White
    Write-Host "Pressione Ctrl+C para parar a aplicacao." -ForegroundColor Gray
    Write-Host ""

    if (-not $NoBrowser) {
        Start-Job -ScriptBlock {
            param($url)
            Start-Sleep -Seconds 3
            Start-Process $url
        } -ArgumentList $AppUrl | Out-Null
    }

    & python scripts/app.py
}
