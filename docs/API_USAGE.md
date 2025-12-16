# API Usage Guide - Hướng dẫn Sử dụng API

## 🚀 Khởi động Server

### Cài đặt Dependencies

```bash
pip install fastapi uvicorn pydantic
```

Hoặc cài đặt tất cả từ requirements.txt:

```bash
pip install -r requirements.txt
```

### Chạy Server

```bash
# Cách 1: Chạy trực tiếp
python src/api_server.py

# Cách 2: Sử dụng uvicorn
uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload
```

Server sẽ chạy tại: **http://localhost:8000**

---

## 📖 API Documentation

Khi server đang chạy, truy cập:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🔌 API Endpoints

### 1. Health Check

```bash
GET /
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "services_ready": true
}
```

---

### 2. Predict Comments (Main Endpoint)

```bash
POST /predict
Content-Type: application/json
```

**Request Body:**

```json
{
  "comments": {
    "1": "Sản phẩm rất tốt, tôi rất hài lòng!",
    "2": "Đồ rác, lừa đảo, đừng mua!",
    "3": "Giao hàng nhanh, đóng gói cẩn thận"
  }
}
```

**Response:**

```json
{
  "results": {
    "1": {
      "is_violation": false,
      "violation_probability": 0.15,
      "comment": "Sản phẩm rất tốt, tôi rất hài lòng!"
    },
    "2": {
      "is_violation": true,
      "violation_probability": 0.92,
      "comment": "Đồ rác, lừa đảo, đừng mua!"
    },
    "3": {
      "is_violation": false,
      "violation_probability": 0.08,
      "comment": "Giao hàng nhanh, đóng gói cẩn thận"
    }
  },
  "total_comments": 3,
  "violation_count": 1
}
```

---

### 3. Simple Predict (Alternative Endpoint)

```bash
POST /predict/simple
Content-Type: application/json
```

**Request Body:**

```json
["Sản phẩm tốt", "Đồ rác", "Giao hàng nhanh"]
```

**Response:**

```json
{
  "predictions": [
    {
      "comment": "Sản phẩm tốt",
      "is_violation": false,
      "violation_probability": 0.12
    },
    {
      "comment": "Đồ rác",
      "is_violation": true,
      "violation_probability": 0.85
    },
    {
      "comment": "Giao hàng nhanh",
      "is_violation": false,
      "violation_probability": 0.05
    }
  ]
}
```

---

## 💻 Code Examples

### Python (requests)

```python
import requests

# API endpoint
url = "http://localhost:8000/predict"

# Request data
data = {
    "comments": {
        "1": "Sản phẩm rất tốt!",
        "2": "Đồ rác, lừa đảo!",
        "3": "Giao hàng nhanh"
    }
}

# Gửi request
response = requests.post(url, json=data)

# Xử lý response
if response.status_code == 200:
    result = response.json()

    print(f"Tổng số bình luận: {result['total_comments']}")
    print(f"Số bình luận vi phạm: {result['violation_count']}")
    print()

    for comment_id, prediction in result['results'].items():
        status = "❌ VI PHẠM" if prediction['is_violation'] else "✅ HỢP LỆ"
        prob = prediction['violation_probability']
        comment = prediction['comment']

        print(f"ID {comment_id}: {status}")
        print(f"  Bình luận: {comment}")
        print(f"  Xác suất vi phạm: {prob:.2%}")
        print()
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

### Python (httpx - async)

```python
import httpx
import asyncio

async def predict_comments(comments_dict):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/predict",
            json={"comments": comments_dict},
            timeout=30.0
        )
        return response.json()

# Sử dụng
comments = {
    "1": "Sản phẩm tốt",
    "2": "Đồ rác"
}

result = asyncio.run(predict_comments(comments))
print(result)
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "comments": {
      "1": "Sản phẩm rất tốt!",
      "2": "Đồ rác, lừa đảo!"
    }
  }'
```

### JavaScript (fetch)

```javascript
const url = "http://localhost:8000/predict";

const data = {
  comments: {
    1: "Sản phẩm rất tốt!",
    2: "Đồ rác, lừa đảo!",
  },
};

fetch(url, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify(data),
})
  .then((response) => response.json())
  .then((result) => {
    console.log("Total comments:", result.total_comments);
    console.log("Violations:", result.violation_count);

    for (const [id, prediction] of Object.entries(result.results)) {
      console.log(`\nID ${id}:`);
      console.log(`  Comment: ${prediction.comment}`);
      console.log(`  Is violation: ${prediction.is_violation}`);
      console.log(
        `  Probability: ${(prediction.violation_probability * 100).toFixed(2)}%`
      );
    }
  })
  .catch((error) => console.error("Error:", error));
```

### Node.js (axios)

```javascript
const axios = require("axios");

const data = {
  comments: {
    1: "Sản phẩm rất tốt!",
    2: "Đồ rác, lừa đảo!",
  },
};

axios
  .post("http://localhost:8000/predict", data)
  .then((response) => {
    const result = response.data;
    console.log("Results:", result);
  })
  .catch((error) => {
    console.error("Error:", error.response?.data || error.message);
  });
```

---

## 🔧 Production Deployment

### Docker

Tạo `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build và chạy:

```bash
docker build -t comment-classification-api .
docker run -p 8000:8000 comment-classification-api
```

### Systemd Service (Linux)

Tạo file `/etc/systemd/system/comment-api.service`:

```ini
[Unit]
Description=Comment Classification API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/CommentClassification
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/uvicorn src.api_server:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Khởi động:

```bash
sudo systemctl enable comment-api
sudo systemctl start comment-api
sudo systemctl status comment-api
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

---

## ⚡ Performance Tips

### 1. Workers (Production)

```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

### 2. GPU Acceleration

Chỉnh sửa `config/model/embedding.py`:

```python
sbert_config = SBERTEmbeddingConfig(
    device='cuda'  # Sử dụng GPU
)
```

### 3. Batch Processing

API tự động xử lý batch. Gửi nhiều comments cùng lúc để tối ưu:

```python
# ✅ GOOD: Batch request
comments = {str(i): text for i, text in enumerate(large_list)}
response = requests.post(url, json={"comments": comments})

# ❌ BAD: Multiple single requests
for text in large_list:
    requests.post(url, json={"comments": {"1": text}})
```

---

## 🐛 Troubleshooting

### Lỗi: "Model not found"

```bash
# Train model trước khi start server
python src/train_main.py
```

### Lỗi: "Address already in use"

```bash
# Đổi port
uvicorn src.api_server:app --port 8001
```

### Lỗi: CUDA out of memory

```python
# Chuyển về CPU trong config/model/embedding.py
sbert_config = SBERTEmbeddingConfig(device='cpu')
```

---

## 📊 Monitoring

### Health Check

```bash
# Kiểm tra server
curl http://localhost:8000/health
```

### Logs

Server logs được lưu trong `logs/` folder.

```bash
# Xem logs realtime
tail -f logs/training_*.log
```

---

## 🔒 Security (Production)

### 1. API Key Authentication

Thêm vào `api_server.py`:

```python
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != "your-secret-key":
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict_comments_endpoint(request: CommentRequest):
    # ...
```

### 2. Rate Limiting

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("100/minute")
async def predict_comments_endpoint(request: Request, ...):
    # ...
```

---

## 📈 Metrics & Analytics

### Prometheus Integration

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

Metrics available at: `http://localhost:8000/metrics`

---

## ❓ FAQ

**Q: API có hỗ trợ HTTPS không?**  
A: Sử dụng nginx hoặc reverse proxy với SSL certificate.

**Q: Có thể xử lý bao nhiêu requests/giây?**  
A: Phụ thuộc vào hardware. CPU: ~10 req/s, GPU: ~50 req/s.

**Q: API có cache predictions không?**  
A: Model tự động cache features, nhưng predictions không cache (real-time).

**Q: Có thể chạy multiple workers không?**  
A: Có, dùng `--workers N` với uvicorn.

---

**Happy API Usage! 🚀**
