# Comment Classification - Hệ thống Phân loại Bình luận

## Tổng quan

**Comment Classification** là một hệ thống phân loại bình luận tự động sử dụng Machine Learning, được xây dựng với kiến trúc **Clean Architecture**, tuân thủ các nguyên tắc **SOLID**, **DRY** và **Dependency Injection**.

Hệ thống hỗ trợ:

- ✅ Phân loại bình luận vi phạm/không vi phạm
- ✅ Kết hợp nhiều phương pháp embedding (SBERT + FastText)
- ✅ Pipeline linh hoạt, dễ mở rộng
- ✅ Cache thông minh để tối ưu hiệu suất
- ✅ Cấu hình tập trung, dễ quản lý

---

## Cấu trúc Dự án

```
CommentClassification/
├── config/                      # Cấu hình toàn bộ ứng dụng
│   ├── core/                    # Cấu hình cốt lõi (paths, settings)
│   ├── model/                   # Cấu hình mô hình (embedding, classifier)
│   └── training/                # Cấu hình huấn luyện
├── src/
│   ├── app_setup.py             # Đăng ký DI Container
│   ├── classifiers/             # Các classifier (LightGBM)
│   ├── containers/              # DI Container
│   ├── interfaces/              # Interfaces/Abstract classes
│   ├── models/                  # Context models
│   ├── pipelines/               # Training & Prediction Pipeline
│   ├── repositories/            # Lưu/load model (Joblib, LightGBM)
│   ├── services/                # Services (extractor, cache, loader)
│   ├── utils/                   # Utilities
│   ├── validators/              # Validation logic
│   ├── train_main.py            # Entry point huấn luyện
│   └── predict_main.py          # Entry point dự đoán
├── data/                        # Dữ liệu huấn luyện
├── ml_model_storage/            # Lưu trữ model đã train
├── docs/                        # Tài liệu dự án
├── test/                        # Unit tests
└── requirements.txt             # Dependencies
```

---

## Cài đặt

### 1. Clone Repository

```bash
git clone <repository-url>
cd CommentClassification
```

### 2. Tạo Virtual Environment (khuyến nghị)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

**Lưu ý:** Dự án yêu cầu Python 3.9+

---

## Hướng dẫn Sử dụng

### 1. Huấn luyện Model

#### Cách 1: Sử dụng Python

```python
from src.train_main import main

# Huấn luyện với cấu hình mặc định
main()
```

#### Cách 2: Chạy trực tiếp

```bash
python src/train_main.py
```

**Quá trình huấn luyện:**

1. Load dữ liệu từ file CSV/Excel
2. Preprocessing (lowercase, loại bỏ ký tự đặc biệt)
3. Trích xuất features (SBERT + FastText embeddings)
4. Chia tập train/validation (stratified split)
5. Huấn luyện LightGBM classifier
6. Đánh giá và lưu model

**Output:**

- Model được lưu tại: `ml_model_storage/models/`
- Feature cache: `ml_model_storage/cache/`
- Logs: `logs/`

### 2. Dự đoán

#### Dự đoán với DataFrame

```python
import pandas as pd
from src.predict_main import main

# Tạo dữ liệu test
test_data = pd.DataFrame({
    'comment': [
        'Sản phẩm rất tốt, tôi rất hài lòng!',
        'Chất lượng tệ, đừng mua!',
        'Giao hàng nhanh, đóng gói cẩn thận'
    ]
})

# Dự đoán
predictions = main(test_data, 'comment')
print(predictions)
```

#### Dự đoán với file

```python
from src.predict_main import main

# Dự đoán từ file CSV
predictions = main('data/test_data.csv', 'comment')

# Dự đoán từ file Excel
predictions = main('data/test_data.xlsx', 'comment')
```

**Output format:**

```
[0, 1, 0]  # 0: không vi phạm, 1: vi phạm
```

---

## Cấu hình

### Cấu trúc Config

Tất cả cấu hình được quản lý tập trung tại thư mục `config/`:

```
config/
├── core/
│   ├── paths.py         # Đường dẫn file/folder
│   └── settings.py      # Tổng hợp tất cả config
├── model/
│   ├── embedding.py     # Cấu hình SBERT & FastText
│   └── classifier.py    # Cấu hình LightGBM
└── training/
    ├── data.py          # Cột dữ liệu, file path
    ├── preprocessing.py # Preprocessing settings
    ├── cache.py         # Cache settings
    ├── evaluation.py    # Metrics, validation split
    ├── logging_config.py # Logging configuration
    └── trainer.py       # Training parameters
```

### Thay đổi Cấu hình

#### Ví dụ 1: Thay đổi model SBERT

Chỉnh sửa `config/model/embedding.py`:

```python
sbert_config = SBERTEmbeddingConfig(
    model_name='keepitreal/vietnamese-sbert',  # Đổi model
    max_seq_length=256,
    device='cuda'  # Sử dụng GPU
)
```

#### Ví dụ 2: Thay đổi tham số LightGBM

Chỉnh sửa `config/model/classifier.py`:

```python
lightgbm_config = LightGBMClassifierConfig(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=63
)
```

#### Ví dụ 3: Thay đổi preprocessing

Chỉnh sửa `config/training/preprocessing.py`:

```python
preprocessing_config = PreprocessingConfig(
    lowercase=True,
    remove_special_chars=True,
    remove_numbers=False,  # Giữ lại số
    remove_urls=True
)
```

---

## Kiến trúc

Hệ thống được xây dựng theo **Clean Architecture** với **Dependency Injection**:

### Các tầng chính:

1. **Interfaces Layer** (`src/interfaces/`)

   - Định nghĩa contracts cho tất cả các component
   - Đảm bảo loose coupling

2. **Domain Layer** (`src/models/`)

   - Business logic và domain models
   - Context objects cho pipeline

3. **Application Layer** (`src/services/`, `src/pipelines/`)

   - Use cases và orchestration
   - Pipeline steps và services

4. **Infrastructure Layer** (`src/repositories/`, `src/classifiers/`)

   - Concrete implementations
   - External dependencies (LightGBM, SBERT, FastText)

5. **DI Container** (`src/containers/`)
   - Quản lý dependencies
   - Singleton và Factory registrations

Chi tiết xem tại: [ARCHITECTURE.md](./ARCHITECTURE.md)

---

## Testing

### Chạy Unit Tests

```bash
# Chạy tất cả tests
pytest test/

# Chạy với coverage
pytest test/ --cov=src --cov-report=html

# Chạy test cụ thể
pytest test/test_preprocessor.py -v
```

### Test Structure

```
test/
├── unit/
│   ├── test_preprocessor.py
│   ├── test_extractor.py
│   ├── test_classifier.py
│   └── test_pipeline.py
└── integration/
    └── test_full_pipeline.py
```

---

## API Documentation

Xem chi tiết tại: [API.md](./API.md)

### Các API chính:

- **TrainingPipeline**: Pipeline huấn luyện model
- **PredictionPipeline**: Pipeline dự đoán
- **FeatureExtractorService**: Trích xuất features
- **DataLoaderService**: Load dữ liệu
- **PreprocessingService**: Tiền xử lý văn bản
- **FeatureCacheService**: Quản lý cache

---

## Performance Tips

### 1. Sử dụng Cache

Cache được tự động kích hoạt. Features đã trích xuất sẽ được lưu lại:

```python
# Lần chạy đầu tiên: chậm (trích xuất features)
predictions = main(test_data, 'comment')

# Lần chạy sau: nhanh (load từ cache)
predictions = main(test_data, 'comment')
```

### 2. GPU Acceleration

Để sử dụng GPU cho SBERT, chỉnh sửa `config/model/embedding.py`:

```python
sbert_config = SBERTEmbeddingConfig(
    device='cuda'  # hoặc 'cuda:0'
)
```

### 3. Batch Processing

Xử lý nhiều dữ liệu cùng lúc:

```python
# Hiệu quả hơn
predictions = main(large_dataframe, 'comment')

# Thay vì
for row in large_dataframe.iterrows():
    prediction = main(pd.DataFrame([row]), 'comment')
```

---

## Troubleshooting

### Lỗi: "No module named 'src'"

**Giải pháp:** Chạy từ thư mục gốc của project:

```bash
cd CommentClassification
python src/train_main.py
```

### Lỗi: "Model not found"

**Nguyên nhân:** Chưa huấn luyện model

**Giải pháp:** Chạy training trước:

```bash
python src/train_main.py
```

### Lỗi: CUDA out of memory

**Giải pháp:** Chuyển về CPU:

```python
# config/model/embedding.py
sbert_config = SBERTEmbeddingConfig(device='cpu')
```

### Cache không hoạt động

**Giải pháp:** Kiểm tra cấu hình cache:

```python
# config/training/cache.py
cache_config = CacheConfig(
    use_cache=True,
    cache_dir='ml_model_storage/cache'
)
```

---

## Contributing

### Quy tắc Code Style

1. **Formatting:** Sử dụng `black`

   ```bash
   black src/ test/
   ```

2. **Linting:** Sử dụng `flake8`

   ```bash
   flake8 src/ test/
   ```

3. **Type Checking:** Sử dụng `mypy`
   ```bash
   mypy src/
   ```

### Pull Request Process

1. Tạo branch mới từ `main`
2. Thực hiện thay đổi và test
3. Chạy linting & formatting
4. Tạo Pull Request với mô tả rõ ràng

---

## License

[Chọn license phù hợp]

---

## Contact

- **Project Lead:** [Tên]
- **Email:** [Email]
- **GitHub:** [Link]

---

## Changelog

### Version 1.0.0 (2024-01-XX)

- ✅ Initial release
- ✅ LightGBM classifier
- ✅ SBERT + FastText embeddings
- ✅ Clean Architecture với DI
- ✅ Feature caching
- ✅ Complete documentation

---

**Happy Coding! 🚀**
