# Web Frontend - Hướng dẫn sử dụng

Web demo đơn giản để test API nhận diện cảm xúc.

## Cách mở web

### Cách 1: Live Server (khuyến nghị)

1. Cài extension "Live Server" trong VS Code
2. Chuột phải vào `web/index.html`
3. Chọn "Open with Live Server"
4. Web tự động mở tại `http://127.0.0.1:5500/web/`

**Ưu điểm:** Auto-reload khi sửa code, không có CORS issues

### Cách 2: Mở trực tiếp

1. Mở File Explorer
2. Navigate đến `NCKH2025\web\`
3. Nhấp đúp `index.html`
4. Web mở trong trình duyệt mặc định

**Lưu ý:** Có thể gặp CORS nếu backend chưa config đúng (đã fix sẵn)

## Sử dụng

### Bước 1: Chọn file âm thanh

- Click "Chọn tệp" hoặc drag & drop
- Hỗ trợ: .wav, .mp3, .flac, .ogg, .m4a
- Khuyến nghị: WAV 16kHz, mono, 2-10 giây

### Bước 2: (Tùy chọn) Nhập transcript

- Nếu bạn biết nội dung giọng nói (tiếng Anh)
- Ví dụ: "I am so happy today!"
- Giúp tăng độ chính xác 5-15%

### Bước 3: Điều chỉnh trọng số

- Slider "Trọng số Audio" (0-1)
- Mặc định: 0.7 (70% audio, 30% text)
- Nếu không nhập transcript, text weight = 0

### Bước 4: Bấm "Dự đoán cảm xúc"

- Hệ thống sẽ gửi request đến backend
- Đợi vài giây (lần đầu chậm hơn do load model)
- Kết quả hiển thị dưới dạng bảng

## Kết quả hiển thị

### Cảm xúc chính
- Label: anger, disgust, fear, joy, sadness, surprise, neutral
- Score: độ tin cậy 0-1 (càng cao càng chắc chắn)

### Fusion Top-k
Top 5 cảm xúc sau khi kết hợp audio + text

### Audio Top-k
Top 5 cảm xúc từ model Wav2Vec2 (chỉ dùng âm thanh)

### Text Top-k
Top 5 cảm xúc từ model BERT (chỉ dùng transcript)
- Chỉ hiển thị nếu có nhập transcript

## Troubleshooting

### "⚠️ Hãy chọn file âm thanh trước"
→ Bạn chưa chọn file audio

### "❌ Có lỗi khi dự đoán"
1. Kiểm tra backend đã chạy chưa: http://127.0.0.1:8000/health
2. Xem Console (F12) để debug
3. Kiểm tra file audio có hợp lệ không

### Kết quả không chính xác
- File audio quá nhiễu → làm sạch âm thanh
- Giọng không phải tiếng Anh → model train trên tiếng Anh
- Âm thanh quá ngắn (<1s) → khó nhận diện
- Thử điều chỉnh trọng số audio/text

### Web load chậm
- Lần đầu chạy backend sẽ tải model (3-10 phút)
- Request đầu tiên mỗi model chậm (load vào RAM)
- Request tiếp theo sẽ nhanh hơn nhiều

## Tính năng

✅ Upload audio file  
✅ Preview audio trước khi predict  
✅ Nhập transcript tùy chọn  
✅ Điều chỉnh fusion weights  
✅ Hiển thị top-k từ 3 nguồn  
✅ Dark theme responsive  
✅ Error handling & user feedback  

## Tech Stack

- **HTML5** - semantic markup
- **CSS3** - modern dark theme, grid layout
- **Vanilla JS** - no dependencies
- **Fetch API** - async requests
- **FormData** - multipart file upload

## Files

```
web/
├── index.html    # UI markup
├── styles.css    # Dark theme + responsive
├── script.js     # Logic + API calls
└── README.md     # This file
```

## Customization

### Thay đổi backend URL

Edit `script.js`, dòng ~38:
```javascript
const resp = await fetch('http://127.0.0.1:8000/predict', ...)
```

### Thay đổi màu sắc

Edit `styles.css`, dòng 1:
```css
:root{
  --bg:#0b1220;       /* Background */
  --card:#151c2f;     /* Card background */
  --text:#e8ecf1;     /* Text color */
  --muted:#9fb0c3;    /* Muted text */
  --accent:#5ac8fa;   /* Accent (blue) */
}
```

### Thêm tính năng

- Real-time audio recording: MediaRecorder API
- Waveform visualization: WaveSurfer.js
- Batch upload: File API + Promise.all
- Export results: Blob + download

## Browser Support

✅ Chrome/Edge 90+  
✅ Firefox 88+  
✅ Safari 14+  
✅ Opera 76+  

Cần hỗ trợ: Fetch API, FormData, ES6+

---

Chúc bạn thành công! 🎉
