@echo off
echo 🚀 Starting Yoga Persian Assistant Backend Deployment...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo ✅ Python is installed.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Set environment variables
set DEEPSEEK_API_KEY=sk-be84c4ea0d24481aab5940fe43e43449
set DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions
set HOST=0.0.0.0
set PORT=8000

REM Create uploads directory
if not exist "uploads" mkdir uploads

REM Start the server
echo 🌟 Starting Yoga Persian Assistant Backend...
echo 📍 Server will be available at: http://localhost:8000
echo 📚 API Documentation: http://localhost:8000/docs
echo 🔍 Health Check: http://localhost:8000/health
echo.
echo Press Ctrl+C to stop the server
echo.

uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
