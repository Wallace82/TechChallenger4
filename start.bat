@echo off
setlocal
cd /d "%~dp0"

echo ==========================================================
echo          Guardia AI -- Assistente Clinico Hospitalar       
echo ==========================================================
echo.

where docker >nul 2>nul
if %ERRORLEVEL% equ 0 (
    docker info >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        echo [INFO] Iniciando via Docker Compose...
        echo [INFO] Aplicacao disponivel em: http://localhost:7860
        echo [INFO] Pressione Ctrl+C para encerrar.
        echo.
        start http://localhost:7860
        docker compose up --build
        goto :eof
    ) else (
        echo [AVISO] Docker esta instalado mas o Docker Desktop/daemon nao esta ativo.
        echo [INFO] Alternando para execucao local via Python...
        echo.
    )
) else (
    echo [AVISO] Docker nao encontrado.
    echo [INFO] Alternando para execucao local via Python...
    echo.
)

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERRO] Nem o Docker ativo nem o Python foram encontrados no sistema.
    pause
    exit /b 1
)

echo [INFO] Configurando variaveis de ambiente e iniciando aplicacao local...
set PERFIL_EXECUCAO=demo-cpu
set ML_RISCO_HABILITADO=1
set PORT=7860
set HOSPITAL_DB_PATH=%~dp0artifacts\demo\hospital_ui.db
set ARTIFACTS_DIR=%~dp0artifacts

start http://localhost:7860
python scripts\app.py
