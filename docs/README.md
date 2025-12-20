# Comment Classification - Hệ thống Phân loại Bình luận

## Tổng quan

`Comment Classification` là một hệ thống phân loại bình luận tự động sử dụng Machine Learning, được tổ chức theo kiến trúc Clean Architecture với Dependency Injection. Hệ thống cung cấp các pipeline huấn luyện và dự đoán cho nhiều loại classifier (ví dụ: LightGBM, Logistic Regression) và hỗ trợ caching feature để tăng hiệu năng.

---

## Cấu trúc Dự án (tóm tắt)

```
CommentClassification/
├── config/                      # Cấu hình toàn bộ ứng dụng
├── src/
│   ├── train_lightgbm_main.py   # Huấn luyện LightGBM
│   ├── train_logreg_main.py     # Huấn luyện Logistic Regression
│   ├── predict_lightgbm_main.py # Dự đoán LightGBM (demo)
│   ├── predict_logreg_main.py   # Dự đoán Logistic Regression (demo)
│   └── api_server.py            # API server (entry point)
├── api_examples/
│   └── api_client_example.py    # Ví dụ client gọi API
├── data/
├── ml_model_storage/
├── docs/
└── requirements.txt
```

---

## Cài đặt

1. Clone repository và vào thư mục dự án:

```bash
git clone <repository-url>
cd CommentClassification
```

2. (Tuỳ chọn) Tạo virtual environment và kích hoạt:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Cài đặt dependencies từ `requirements.txt`:

## Hướng dẫn Sử dụng (chỉ các file `*__main__.py`, `api_server.py`, `api_examples/api_client_example.py`)

1. Huấn luyện (CLI):

```bash
# Huấn luyện LightGBM (mặc định dùng data/raw/train_full.csv)
python -m src.train_lightgbm_main

# Sử dụng file raw khác
python -m src.train_lightgbm_main --data data/raw/train.csv

# Sử dụng file đã processed
python -m src.train_lightgbm_main --processed data/processed/processed_lightgbm_YYYYMMDD_train_xxx.csv

# Tương tự cho Logistic Regression
python -m src.train_logreg_main
```

Tuỳ chọn chung: `--no-split` để dùng toàn bộ dữ liệu làm train (không chia test/val).

2. Dự đoán (demo scripts):

```bash
# Chạy demo dự đoán in kết quả ra stdout
python -m src.predict_lightgbm_main
python -m src.predict_logreg_main
```

3. Chạy API server:

```bash
# Chạy server (entry point)
python src/api_server.py
```

4. Ví dụ client gọi API:

```bash
# Ví dụ client mẫu (gọi server ở localhost)
python api_examples/api_client_example.py
```

Ghi chú: các script demo `predict_*_main.py` hiện in kết quả ra stdout. Nếu cần tích hợp programmatic, hãy sử dụng trực tiếp các pipeline trong `src/pipelines/` hoặc xem `api_examples/api_client_example.py` để gọi API.
. Ví dụ client (api_examples)

```bash
# Ví dụ client mẫu gọi API
python api_examples/api_client_example.py
```

---

## Cấu hình

Các cấu hình chính nằm trong thư mục `config/` (ví dụ: `config/core/paths.py`, `config/model/`, `config/training/`). Thay đổi cấu hình bằng cách chỉnh các file tương ứng trong `config/`.

---

## Kiến trúc

Hệ thống tổ chức theo Clean Architecture với các tầng: Interfaces, Domain (models), Application (services, pipelines) và Infrastructure (repositories, classifiers). Dependency Injection được cấu hình trong `src/app_setup.py` và container trong `src/containers/`.

Chi tiết kiến trúc: [ARCHITECTURE.md](../docs/ARCHITECTURE.md)

---

## Troubleshooting (những vấn đề phổ biến)

- "No module named 'src'": chạy từ thư mục gốc hoặc dùng `-m`:

```bash
cd CommentClassification
python -m src.train_lightgbm_main
```

- "Model not found": chưa có model được huấn luyện — chạy một trong các script `train_*_main.py` trước:

```bash
python -m src.train_lightgbm_main
```

- CUDA out of memory: cấu hình `config/model/embedding.py` để dùng CPU (`device='cpu'`).

- Cache không hoạt động: kiểm tra `config/training/cache.py` và đường dẫn cache (`ml_model_storage/cache`).

---
