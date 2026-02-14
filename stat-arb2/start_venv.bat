@echo off
set "VENV_DIR=..\.venv"

if not exist "%VENV_DIR%" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

if exist "requirements.txt" (
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    echo requirements.txt not found.
)

echo Setup complete. Virtual environment is active.
cmd /k
