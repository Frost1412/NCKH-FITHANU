# 🚀 HƯỚNG DẪN CHẠY NHANH

## Bước 1: Cài đặt môi trường (chỉ làm 1 lần)

Mở PowerShell trong VS Code hoặc terminal Windows tại thư mục gốc dự án:

```powershell
# Tạo môi trường ảo Python
python -m venv .venv

# Kích hoạt môi trường
.\.venv\Scripts\Activate.ps1

# Cài đặt tất cả dependencies (FastAPI, transformers, torch, librosa...)
pip install -r backend/requirements.txt
```

**Lưu ý:** Lần cài đầu tiên sẽ tải khoảng 2-4GB (PyTorch + các thư viện ML). Hãy kiên nhẫn!

---

## Bước 2: Chạy Backend API

```powershell
# Đảm bảo môi trường đã được kích hoạt (có (.venv) ở đầu dòng prompt)
.\.venv\Scripts\Activate.ps1

# Chạy backend FastAPI
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --app-dir backend
```

Khi thấy:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

Backend đã sẵn sàng! ✅

**Lần chạy đầu tiên:** Hugging Face sẽ tự động tải 2 mô hình:
- `superb/wav2vec2-base-superb-er` (Wav2Vec2 cho emotion recognition)
- `j-hartmann/emotion-english-distilroberta-base` (BERT cho text emotion)

Có thể mất 3-10 phút tùy tốc độ mạng.

---

## Bước 3: Mở Web Demo

**Cách 1 (khuyến nghị):** Dùng Live Server của VS Code
1. Cài extension "Live Server" trong VS Code
2. Chuột phải vào `web/index.html` → chọn "Open with Live Server"
3. Web sẽ tự động mở tại `http://127.0.0.1:5500/web/`

**Cách 2:** Mở trực tiếp
1. Mở File Explorer
2. Nhấp đúp vào file `web/index.html`
3. Web sẽ mở trong trình duyệt mặc định

---

## Bước 4: Test thử!

1. **Chọn file âm thanh** (.wav hoặc .mp3)
   - Có thể dùng file ghi âm từ điện thoại
   - Hoặc tạo file test bằng Windows Voice Recorder
   
2. **(Tùy chọn)** Nhập transcript tiếng Anh
   - Ví dụ: "I am so happy today!"
   - Giúp cải thiện độ chính xác

3. **Điều chỉnh trọng số Audio** (mặc định 0.7)
   - 1.0 = chỉ dùng âm thanh
   - 0.5 = cân bằng audio + text
   - 0.0 = chỉ dùng text (cần có transcript)

4. **Bấm "Dự đoán cảm xúc"**

5. **Xem kết quả:**
   - Final label và score
   - Top-k emotions từ audio
   - Top-k emotions từ text (nếu có)
   - Fusion top-k (kết hợp)

---

## Kiểm tra Backend hoạt động

Mở trình duyệt, truy cập: http://127.0.0.1:8000/health

Nếu thấy: `{"status":"ok"}` → Backend OK! ✅

Xem API docs tự động: http://127.0.0.1:8000/docs

---

## Xử lý lỗi thường gặp

### ❌ "Import 'fastapi' could not be resolved"
→ Bạn chưa kích hoạt môi trường ảo hoặc chưa cài dependencies
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

### ❌ Web báo "Có lỗi khi dự đoán"
→ Backend chưa chạy hoặc chạy sai port
1. Kiểm tra backend: http://127.0.0.1:8000/health
2. Xem log terminal backend để debug

### ❌ "ModuleNotFoundError: No module named 'uvicorn'"
→ Chưa cài dependencies hoặc kích hoạt sai venv
```powershell
.\.venv\Scripts\Activate.ps1
pip install uvicorn[standard]
```

### ❌ Model download bị lỗi
→ Vấn đề mạng hoặc Hugging Face Hub
1. Kiểm tra kết nối internet
2. Thử lại sau vài phút
3. Nếu ở công ty: cấu hình proxy
   ```powershell
   $env:HTTP_PROXY="http://proxy:port"
   $env:HTTPS_PROXY="http://proxy:port"
   ```

---

## Tắt ứng dụng

1. **Tắt backend:** Nhấn `Ctrl+C` trong terminal đang chạy uvicorn
2. **Tắt web:** Đóng tab trình duyệt

---

## Cấu trúc API

### POST `/predict`
**Input (multipart/form-data):**
- `file`: audio file (.wav, .mp3, .flac...)
- `transcript` (optional): text transcript (English)
- `audio_weight` (optional): float 0-1, default 0.7

**Output (JSON):**
```json
{
  "final_label": "joy",
  "final_score": 0.834,
  "top_k": [
    {"label": "joy", "score": 0.834},
    {"label": "neutral", "score": 0.112},
    ...
  ],
  "audio_top_k": [...],
  "text_top_k": [...],
  "fusion_weights": {"audio": 0.7, "text": 0.3}
}
```

**Cảm xúc được nhận diện:**
- `anger` (tức giận)
- `disgust` (ghê tởm)
- `fear` (sợ hãi)
- `joy` (vui vẻ/hạnh phúc)
- `sadness` (buồn bã)
- `surprise` (ngạc nhiên)
- `neutral` (trung tính)

---

## Tips

- Mô hình hoạt động tốt nhất với:
  ✅ Audio rõ ràng, ít nhiễu
  ✅ Giọng nói tiếng Anh
  ✅ Đoạn âm thanh 2-10 giây
  
- File âm thanh quá dài (>30s) sẽ chậm hơn
- Transcript giúp cải thiện độ chính xác 5-15%
- Lần đầu predict mỗi model sẽ chậm (load vào RAM), lần sau nhanh hơn

---

## Hỗ trợ

Nếu gặp vấn đề:
1. Kiểm tra logs trong terminal backend
2. Mở DevTools trình duyệt (F12) → tab Console để xem lỗi JS
3. Kiểm tra `backend/README.md` để biết thêm chi tiết

Chúc bạn thành công! 🎉
