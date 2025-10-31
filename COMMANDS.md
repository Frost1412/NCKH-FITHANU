# Các lệnh hữu ích - Quick Reference

## Setup lần đầu

```powershell
# Tạo virtual environment
python -m venv .venv

# Kích hoạt venv
.\.venv\Scripts\Activate.ps1

# Cài đặt dependencies
pip install -r backend/requirements.txt

# Upgrade pip (optional)
python -m pip install --upgrade pip
```

## Chạy backend

```powershell
# Cách 1: Script tự động
.\run_backend.ps1

# Cách 2: Thủ công
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --app-dir backend

# Cách 3: Production mode (no reload)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir backend --workers 2
```

## Kiểm tra

```powershell
# Health check
python check_backend.py

# Hoặc dùng curl
curl http://127.0.0.1:8000/health

# Xem API docs
start http://127.0.0.1:8000/docs
```

## Testing

```powershell
# Cài pytest
pip install pytest

# Chạy tất cả tests
pytest -v backend/tests/

# Chạy 1 file test
pytest -v backend/tests/test_audio_processing.py

# Xem coverage
pip install pytest-cov
pytest --cov=backend/app --cov-report=html backend/tests/
```

## Python environment

```powershell
# Kiểm tra Python version
python --version

# Xem installed packages
pip list

# Xem package info
pip show transformers

# Update package
pip install --upgrade transformers

# Freeze dependencies
pip freeze > requirements.txt

# Deactivate venv
deactivate
```

## Tạo sample data

```powershell
# Tạo file audio test
python create_sample_audio.py

# Tạo transcript mẫu
echo "I am so happy today!" > samples/transcript.txt
```

## Git commands

```powershell
# Init repo (nếu chưa có)
git init

# Add all files
git add .

# Commit
git commit -m "feat: initial backend + web demo"

# Check status
git status

# View diff
git diff
```

## Debugging

```powershell
# Chạy Python REPL với venv
python

# Test import
python -c "import fastapi; print(fastapi.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import torch; print(torch.__version__)"

# Kiểm tra CUDA (GPU)
python -c "import torch; print(torch.cuda.is_available())"

# Check RAM usage
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().percent}%')"
```

## API testing với curl

```powershell
# Health check
curl http://127.0.0.1:8000/health

# Predict với audio file
curl -X POST http://127.0.0.1:8000/predict `
  -F "file=@samples/test_audio.wav" `
  -F "audio_weight=0.7"

# Predict với audio + transcript
curl -X POST http://127.0.0.1:8000/predict `
  -F "file=@samples/test_audio.wav" `
  -F "transcript=I am so happy today" `
  -F "audio_weight=0.7"
```

## Clean up

```powershell
# Xóa cache Python
Get-ChildItem -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force

# Xóa venv (cẩn thận!)
Remove-Item -Recurse -Force .venv

# Xóa Hugging Face cache (giải phóng dung lượng)
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface"
```

## Performance monitoring

```powershell
# Cài monitoring tools
pip install psutil

# Monitor script
python -c "
import psutil
import time
while True:
    print(f'CPU: {psutil.cpu_percent()}% | RAM: {psutil.virtual_memory().percent}%')
    time.sleep(1)
"
```

## Docker (tương lai)

```powershell
# Build image
docker build -t ser-backend -f backend/Dockerfile .

# Run container
docker run -p 8000:8000 ser-backend

# Docker compose
docker-compose up -d
```

## Useful VS Code commands

```
Ctrl+Shift+P -> Python: Select Interpreter
Ctrl+Shift+P -> Python: Create Terminal
Ctrl+` -> Toggle Terminal
F5 -> Start Debugging
```

## Troubleshooting one-liners

```powershell
# Tìm process chiếm port 8000
netstat -ano | findstr :8000

# Kill process theo PID
taskkill /PID <PID> /F

# Xem biến môi trường
$env:PATH

# Set proxy (nếu cần)
$env:HTTP_PROXY="http://proxy:port"
$env:HTTPS_PROXY="http://proxy:port"

# Clear proxy
Remove-Item Env:\HTTP_PROXY
Remove-Item Env:\HTTPS_PROXY
```

## Backup & Export

```powershell
# Export environment
pip freeze > requirements.txt

# Tạo archive (cần 7zip hoặc tar)
tar -czf nckh2025-backup.tar.gz --exclude=.venv --exclude=__pycache__ .

# Hoặc dùng Compress-Archive
Compress-Archive -Path . -DestinationPath nckh2025-backup.zip -Force
```

---

**Lưu ý:** 
- Luôn activate venv trước khi chạy Python commands
- Kiểm tra backend health trước khi test web
- Xem logs terminal để debug lỗi
