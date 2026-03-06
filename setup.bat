@echo off
REM =============================================================================
REM  setup.bat — One-command setup for Arbitrage Bots (Windows)
REM =============================================================================
REM
REM  Usage (in Command Prompt or PowerShell):
REM    setup.bat
REM
REM  What it does:
REM    1. Checks Python 3.9+
REM    2. Creates a virtual environment in python\.venv
REM    3. Installs all Python dependencies
REM    4. Copies config template -> python\config\config.yaml (if not present)
REM    5. Prints next steps
REM =============================================================================
setlocal enabledelayedexpansion

echo.
echo  =============================================
echo   Arbitrage Bots -- Quick Setup (Windows)
echo  =============================================
echo.

REM ── 1. Check Python ─────────────────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found.
    echo          Install Python 3.9+ from https://python.org
    echo          Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PY_VER=%%v
echo  [OK] Python %PY_VER% found

REM ── 2. Create virtual environment ───────────────────────────────────────────
set VENV_DIR=python\.venv
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo  [SKIP] Virtual environment already exists
) else (
    echo  [..] Creating virtual environment at python\.venv ...
    python -m venv %VENV_DIR%
    echo  [OK] Virtual environment created
)

REM ── 3. Install dependencies ─────────────────────────────────────────────────
echo  [..] Installing Python dependencies...
call %VENV_DIR%\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
python -m pip install -r python\requirements.txt --quiet
echo  [OK] Dependencies installed

REM ── 4. Copy config template ─────────────────────────────────────────────────
if exist "python\config\config.yaml" (
    echo  [SKIP] python\config\config.yaml already exists
) else (
    copy "python\config\config.example.yaml" "python\config\config.yaml" >nul
    echo  [OK] Created python\config\config.yaml from example template
)

REM ── 5. Copy .env template ────────────────────────────────────────────────────
if exist ".env" (
    echo  [SKIP] .env already exists
) else if exist ".env.example" (
    copy ".env.example" ".env" >nul
    echo  [OK] Created .env from .env.example
)

REM ── 6. Done ──────────────────────────────────────────────────────────────────
echo.
echo  Setup complete!
echo.
echo  Next steps:
echo.
echo    Option A -- Try the simulator (no API keys needed):
echo.
echo      python\.venv\Scripts\activate.bat
echo      python arb.py
echo.
echo    Option B -- Configure API keys and run a bot:
echo.
echo      python\.venv\Scripts\activate.bat
echo      python arb.py      ^(choose "Setup Wizard" from the menu^)
echo.
echo    Tip: All bots start in dry_run mode -- no real orders are placed.
echo.
pause
