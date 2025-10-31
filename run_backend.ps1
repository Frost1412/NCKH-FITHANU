# Backend startup script
# Usage: .\run_backend.ps1

Write-Host "🚀 Starting FastAPI backend..." -ForegroundColor Cyan

# Activate virtual environment
$venvPath = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    & $venvPath
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "⚠️ Virtual environment not found. Creating..." -ForegroundColor Yellow
    python -m venv .venv
    & $venvPath
    Write-Host "✅ Virtual environment created and activated" -ForegroundColor Green
}

# Check dependencies
Write-Host "📦 Checking dependencies..." -ForegroundColor Cyan
$pipList = pip list
if (-not ($pipList -match "fastapi")) {
    Write-Host "⚠️ Dependencies not installed. Installing..." -ForegroundColor Yellow
    pip install -r backend/requirements.txt
    Write-Host "✅ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "✅ Dependencies ready" -ForegroundColor Green
}

# Run server
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "🌐 Backend running at: http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "📚 API Docs: http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "❤️  Health Check: http://127.0.0.1:8000/health" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
Write-Host "⚠️  First run will download models from Hugging Face (3-10 min)" -ForegroundColor Yellow
Write-Host "🛑 Press Ctrl+C to stop server" -ForegroundColor Red
Write-Host ""

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --app-dir backend
