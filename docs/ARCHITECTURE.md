# ARCHITECTURE - Kiến trúc Hệ thống

## Tổng quan Kiến trúc

Comment Classification được xây dựng dựa trên **Clean Architecture** kết hợp với **Dependency Injection (DI)**, tuân thủ các nguyên tắc **SOLID** để đảm bảo:

- ✅ **Testability**: Dễ dàng kiểm thử từng component riêng biệt
- ✅ **Maintainability**: Dễ bảo trì và mở rộng
- ✅ **Flexibility**: Dễ dàng thay đổi implementation mà không ảnh hưởng toàn hệ thống
- ✅ **Scalability**: Có thể mở rộng với các mô hình/pipeline mới

---

## Kiến trúc Tổng thể

```
┌─────────────────────────────────────────────────────────┐
│                   Entry Points                          │
│              (train_main.py, predict_main.py)           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              DI Container (app_setup.py)                │
│         Quản lý dependencies & lifecycle                │
└────────────────────┬────────────────────────────────────┘
                     │
      ┌──────────────┼──────────────┐
      │              │               │
      ▼              ▼               ▼
┌──────────┐  ┌──────────┐   ┌──────────┐
│Interfaces│  │Services  │   │Pipelines │
│   Layer  │  │  Layer   │   │  Layer   │
└─────┬────┘  └────┬─────┘   └────┬─────┘
      │            │              │
      └────────────┼──────────────┘
                   ▼
      ┌────────────────────────┐
      │  Infrastructure Layer  │
      │ (Repositories,         │
      │  Classifiers, Models)  │
      └────────────────────────┘
                   │
                   ▼
      ┌────────────────────────┐
      │   External Libraries   │
      │ (LightGBM, SBERT,      │
      │  FastText, Pandas)     │
      └────────────────────────┘
```

---

## Chi tiết Các Layer

### 1. Interfaces Layer (`src/interfaces/`)

**Mục đích:** Định nghĩa contracts (abstract base classes) cho tất cả các component

**Cấu trúc:**

```
interfaces/
├── classifier/
│   ├── i_classifier.py                  # Interface cho classifier
│   └── i_classifier_context.py          # Context cho classifier
├── data/
│   ├── i_data_loader.py                 # Interface load dữ liệu
│   └── i_data_splitter.py               # Interface chia tập train/val
├── embedding/
│   ├── i_embedding_extractor.py         # Base interface embedding
│   ├── i_sbert_extractor.py             # Interface SBERT
│   └── i_fasttext_extractor.py          # Interface FastText
├── pipeline/
│   ├── i_pipeline.py                    # Base pipeline interface
│   ├── i_pipeline_step.py               # Interface cho mỗi step
│   └── i_pipeline_config.py             # Interface config pipeline
├── repositories/
│   ├── i_classifier_repository.py       # Interface lưu/load classifier
│   └── i_classifier_repository_provider.py  # Factory pattern
└── services/
    ├── i_cache_service.py               # Interface cache service
    ├── i_extractor_service.py           # Interface feature extractor
    └── i_preprocessor_service.py        # Interface preprocessing
```

**Ví dụ Interface:**

```python
from abc import ABC, abstractmethod
from typing import Any

class IClassifier(ABC):
    """Interface cho tất cả các classifier"""

    @abstractmethod
    def train(self, X: Any, y: Any, config: dict = None) -> None:
        """Huấn luyện model"""
        pass

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Dự đoán"""
        pass

    @abstractmethod
    def save(self, repository: 'IClassifierRepository') -> None:
        """Lưu model"""
        pass
```

**Lợi ích:**

- Loose coupling giữa các component
- Dễ dàng mock cho unit testing
- Có thể swap implementation mà không ảnh hưởng code khác

---

### 2. Domain Layer (`src/models/`)

**Mục đích:** Chứa business logic và domain models

**Cấu trúc:**

```
models/
├── classifier/
│   ├── base_classifier_context.py       # Base context cho classifier
│   └── lightgbm_classifier_context.py   # Context cho LightGBM
├── embedding/
│   ├── embedding_context.py             # Base embedding context
│   ├── sbert_context.py                 # Context SBERT
│   └── fasttext_context.py              # Context FastText
└── pipeline/
    ├── training_context.py              # Context cho training pipeline
    └── prediction_context.py            # Context cho prediction pipeline
```

**Context Pattern:**

Context objects đóng gói dữ liệu và state của pipeline:

```python
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class TrainingContext:
    """Context object cho Training Pipeline"""

    # Input data
    raw_data: pd.DataFrame = None
    text_column: str = None
    label_columns: list[str] = None

    # Processed data
    preprocessed_texts: list[str] = None
    features: np.ndarray = None
    labels: np.ndarray = None

    # Split data
    X_train: np.ndarray = None
    X_val: np.ndarray = None
    y_train: np.ndarray = None
    y_val: np.ndarray = None

    # Trained model
    trained_classifier: 'IClassifier' = None

    # Evaluation results
    evaluation_metrics: dict = None
```

**Lợi ích:**

- Tránh parameter explosion
- Dễ dàng truyền state giữa các pipeline steps
- Type-safe với dataclass

---

### 3. Application Layer

#### 3.1. Services Layer (`src/services/`)

**Mục đích:** Implement business use cases

**Cấu trúc:**

```
services/
├── cache/
│   └── feature_cache_service.py         # Quản lý cache features
├── data/
│   ├── loader.py                        # Load dữ liệu CSV/Excel
│   └── splitter.py                      # Chia train/validation
├── embedding/
│   ├── extractor_service.py             # Service trích xuất features
│   ├── sbert_extractor.py               # SBERT embedding
│   └── fasttext_extractor.py            # FastText embedding
├── preprocessing/
│   └── preprocessor_service.py          # Tiền xử lý text
└── logging/
    └── logger_service.py                # Logging service
```

**Ví dụ Service:**

```python
class FeatureExtractorService(IFeatureExtractorService):
    """Service kết hợp nhiều embedding methods"""

    def __init__(
        self,
        sbert_extractor: ISBERTExtractor,
        fasttext_extractor: IFastTextExtractor,
        cache_service: IFeatureCacheService
    ):
        self.sbert = sbert_extractor
        self.fasttext = fasttext_extractor
        self.cache = cache_service

    def extract_features(self, texts: list[str]) -> np.ndarray:
        """Trích xuất features kết hợp SBERT + FastText"""

        # Kiểm tra cache trước
        cached = self.cache.load_features(texts)
        if cached is not None:
            return cached

        # Extract features
        sbert_features = self.sbert.extract(texts)
        fasttext_features = self.fasttext.extract(texts)

        # Concatenate
        combined = np.hstack([sbert_features, fasttext_features])

        # Save to cache
        self.cache.save_features(texts, combined)

        return combined
```

**Lợi ích:**

- Single Responsibility: Mỗi service có một nhiệm vụ rõ ràng
- Dependencies được inject qua constructor
- Dễ dàng test với mock dependencies

#### 3.2. Pipelines Layer (`src/pipelines/`)

**Mục đích:** Orchestrate workflow của training và prediction

**Cấu trúc:**

```
pipelines/
├── training_pipeline.py                 # Training pipeline
├── prediction_pipeline.py               # Prediction pipeline
├── pipeline_config/
│   └── lightgbm_pipeline_config.py      # Config cho pipeline
└── steps/
    ├── training/
    │   ├── data_loading_step.py         # Step 1: Load data
    │   ├── preprocessing_step.py        # Step 2: Preprocessing
    │   ├── feature_extraction_step.py   # Step 3: Extract features
    │   ├── data_splitting_step.py       # Step 4: Split train/val
    │   ├── training_step.py             # Step 5: Train model
    │   ├── evaluation_step.py           # Step 6: Evaluate
    │   └── model_saving_step.py         # Step 7: Save model
    └── prediction/
        ├── data_loading_step.py
        ├── preprocessing_step.py
        ├── feature_extraction_step.py
        ├── model_loading_step.py
        └── prediction_step.py
```

**Pipeline Pattern:**

```python
class TrainingPipeline(IPipeline):
    """Pipeline cho training workflow"""

    def __init__(self, steps: list[IPipelineStep]):
        self.steps = steps

    def execute(self, context: TrainingContext) -> TrainingContext:
        """Execute tất cả steps theo thứ tự"""

        for step in self.steps:
            logger.info(f"Đang thực thi: {step.name}")
            context = step.execute(context)
            logger.info(f"Hoàn thành: {step.name}")

        return context
```

**Step Pattern:**

```python
class PreprocessingStep(IPipelineStep):
    """Step tiền xử lý dữ liệu"""

    def __init__(self, preprocessor: IPreprocessorService):
        self.preprocessor = preprocessor

    @property
    def name(self) -> str:
        return "Preprocessing"

    def execute(self, context: TrainingContext) -> TrainingContext:
        """Thực thi preprocessing"""

        texts = context.raw_data[context.text_column].tolist()
        context.preprocessed_texts = self.preprocessor.process(texts)

        return context
```

**Lợi ích:**

- Workflow rõ ràng, dễ hiểu
- Mỗi step độc lập, dễ test
- Dễ thêm/xóa/sửa steps mà không ảnh hưởng toàn pipeline

---

### 4. Infrastructure Layer

#### 4.1. Repositories (`src/repositories/`)

**Mục đích:** Quản lý persistence (lưu/load models)

**Cấu trúc:**

```
repositories/
├── classifier/
│   ├── joblib_provider.py               # Provider dùng Joblib
│   ├── boost_provider.py                # Provider dùng LightGBM native
│   └── repository_factory.py            # Factory tạo repository
└── cache/
    └── feature_cache_repository.py      # Repository cho cache
```

**Repository Pattern:**

```python
class JoblibRepositoryProvider(IClassifierRepositoryProvider):
    """Provider sử dụng Joblib để lưu/load model"""

    def save(self, model: Any, filepath: str) -> None:
        """Lưu model bằng Joblib"""
        joblib.dump(model, filepath)

    def load(self, filepath: str) -> Any:
        """Load model bằng Joblib"""
        return joblib.load(filepath)
```

**Factory Pattern:**

```python
class ClassifierRepositoryFactory:
    """Factory tạo repository dựa trên loại model"""

    @staticmethod
    def create(model_type: str) -> IClassifierRepositoryProvider:
        if model_type == 'lightgbm':
            return BoostRepositoryProvider()
        elif model_type == 'generic':
            return JoblibRepositoryProvider()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
```

#### 4.2. Classifiers (`src/classifiers/`)

**Mục đích:** Concrete implementations của ML models

```python
class LightGBMClassifier(IClassifier):
    """LightGBM implementation"""

    def __init__(self):
        self.model = None

    def train(self, X: Any, y: Any, config: dict = None) -> None:
        config = config or {}
        self.model = lgb.LGBMClassifier(**config)
        self.model.fit(X, y.ravel())

    def predict(self, X: Any) -> Any:
        return self.model.predict(X)

    def save(self, repository: IClassifierRepository) -> None:
        repository.save(self.model)
```

---

### 5. DI Container (`src/containers/`)

**Mục đích:** Quản lý lifecycle và dependencies của tất cả components

**Cấu trúc:**

```
containers/
└── service_container.py                 # Singleton DI Container
```

**Implementation:**

```python
class ServiceContainer:
    """Singleton DI Container"""

    _instance = None
    _services = {}
    _factories = {}

    @classmethod
    def register_singleton(cls, name: str, service: Any):
        """Đăng ký service dạng singleton"""
        cls._services[name] = service

    @classmethod
    def register_factory(cls, name: str, factory: Callable):
        """Đăng ký factory để tạo instance mới mỗi lần resolve"""
        cls._factories[name] = factory

    @classmethod
    def resolve(cls, name: str) -> Any:
        """Lấy service đã đăng ký"""

        # Kiểm tra singleton trước
        if name in cls._services:
            return cls._services[name]

        # Tạo mới từ factory
        if name in cls._factories:
            return cls._factories[name]()

        raise ValueError(f"Service '{name}' not registered")
```

**Registration trong `app_setup.py`:**

```python
def setup_di_container() -> None:
    """Đăng ký tất cả services vào DI Container"""

    # 1. Logger (Singleton)
    logger_service = LoggerService()
    ServiceContainer.register_singleton('logger_service', logger_service)

    # 2. SBERT Extractor (Singleton - vì model nặng)
    sbert_extractor = SBERTExtractor(sbert_config)
    ServiceContainer.register_singleton('sbert_extractor', sbert_extractor)

    # 3. FastText Extractor (Singleton)
    fasttext_extractor = FastTextExtractor(fasttext_config)
    ServiceContainer.register_singleton('fasttext_extractor', fasttext_extractor)

    # 4. Cache Service (Singleton)
    cache_service = FeatureCacheService(cache_config)
    ServiceContainer.register_singleton('cache_service', cache_service)

    # 5. Extractor Service (Singleton)
    extractor = FeatureExtractorService(
        sbert_extractor,
        fasttext_extractor,
        cache_service
    )
    ServiceContainer.register_singleton('extractor_service', extractor)

    # 6. Classifier (Factory - mỗi pipeline cần instance mới)
    ServiceContainer.register_factory(
        'classifier',
        lambda: LightGBMClassifier()
    )
```

---

## Design Patterns Áp dụng

### 1. Dependency Injection (DI)

**Mục đích:** Giảm coupling, tăng testability

**Ví dụ:**

```python
# BAD: Hard-coded dependency
class FeatureExtractorService:
    def __init__(self):
        self.sbert = SBERTExtractor()  # ❌ Tight coupling
        self.fasttext = FastTextExtractor()  # ❌ Tight coupling

# GOOD: Dependency Injection
class FeatureExtractorService:
    def __init__(
        self,
        sbert: ISBERTExtractor,  # ✅ Inject interface
        fasttext: IFastTextExtractor  # ✅ Inject interface
    ):
        self.sbert = sbert
        self.fasttext = fasttext
```

### 2. Repository Pattern

**Mục đích:** Tách biệt data access logic

**Ví dụ:**

```python
# Classifier không quan tâm cách lưu/load
class LightGBMClassifier:
    def save(self, repository: IClassifierRepository):
        repository.save(self.model)  # Repository handle persistence

    def load(self, repository: IClassifierRepository):
        self.model = repository.load()
```

### 3. Factory Pattern

**Mục đích:** Tạo objects mà không expose creation logic

**Ví dụ:**

```python
# Factory tạo repository phù hợp
repository = ClassifierRepositoryFactory.create('lightgbm')
classifier.save(repository)
```

### 4. Pipeline Pattern

**Mục đích:** Tổ chức workflow thành sequence of steps

**Ví dụ:**

```python
pipeline = TrainingPipeline([
    DataLoadingStep(),
    PreprocessingStep(),
    FeatureExtractionStep(),
    DataSplittingStep(),
    TrainingStep(),
    EvaluationStep(),
    ModelSavingStep()
])

context = pipeline.execute(TrainingContext())
```

### 5. Context Object Pattern

**Mục đích:** Pass state giữa các steps mà không parameter explosion

**Ví dụ:**

```python
# Thay vì:
def step1(data): ...
def step2(preprocessed, features): ...
def step3(features, labels, split_ratio): ...  # ❌ Too many params

# Dùng Context:
def step1(context: Context) -> Context: ...
def step2(context: Context) -> Context: ...
def step3(context: Context) -> Context: ...  # ✅ Clean
```

---

## SOLID Principles

### 1. Single Responsibility Principle (SRP)

Mỗi class chỉ có **một lý do để thay đổi**.

**Ví dụ:**

```python
# ✅ GOOD: Mỗi service có trách nhiệm riêng
class DataLoaderService:
    """Chỉ lo load dữ liệu"""
    def load_csv(self, filepath: str) -> pd.DataFrame: ...

class PreprocessorService:
    """Chỉ lo preprocessing"""
    def process(self, texts: list[str]) -> list[str]: ...

class FeatureExtractorService:
    """Chỉ lo extract features"""
    def extract(self, texts: list[str]) -> np.ndarray: ...
```

### 2. Open/Closed Principle (OCP)

Open for extension, closed for modification.

**Ví dụ:**

```python
# Thêm embedding method mới mà KHÔNG sửa code cũ
class NewEmbeddingExtractor(IEmbeddingExtractor):
    """Embedding method mới"""
    def extract(self, texts: list[str]) -> np.ndarray:
        # Implementation mới
        pass

# Chỉ cần đăng ký vào DI Container
ServiceContainer.register_singleton('new_embedding', NewEmbeddingExtractor())
```

### 3. Liskov Substitution Principle (LSP)

Subclasses phải thay thế được base class mà không làm sai logic.

**Ví dụ:**

```python
# Mọi classifier đều implement IClassifier
def train_model(classifier: IClassifier, X, y):
    classifier.train(X, y)  # Works với bất kỳ classifier nào

# Có thể thay thế bất kỳ implementation nào
train_model(LightGBMClassifier(), X, y)  # ✅ Works
train_model(RandomForestClassifier(), X, y)  # ✅ Works
```

### 4. Interface Segregation Principle (ISP)

Không ép client implement interface không cần thiết.

**Ví dụ:**

```python
# ✅ GOOD: Interfaces nhỏ, focused
class IEmbeddingExtractor(ABC):
    @abstractmethod
    def extract(self, texts: list[str]) -> np.ndarray: ...

class IModelPersistence(ABC):
    @abstractmethod
    def save(self, model: Any, path: str) -> None: ...
    @abstractmethod
    def load(self, path: str) -> Any: ...

# ❌ BAD: Interface quá lớn
class IHugeInterface(ABC):
    def extract(self, texts): ...
    def train(self, X, y): ...
    def save(self, path): ...
    def load(self, path): ...  # Không phải ai cũng cần tất cả
```

### 5. Dependency Inversion Principle (DIP)

High-level modules không phụ thuộc vào low-level modules. Cả hai phụ thuộc vào abstractions.

**Ví dụ:**

```python
# ✅ GOOD: Pipeline phụ thuộc vào interface, không phụ thuộc concrete class
class TrainingPipeline:
    def __init__(
        self,
        loader: IDataLoader,        # ← Interface
        preprocessor: IPreprocessor, # ← Interface
        extractor: IExtractor        # ← Interface
    ):
        self.loader = loader
        self.preprocessor = preprocessor
        self.extractor = extractor
```

---

## Data Flow

### Training Flow

```
1. DataLoadingStep
   ├── Input: filepath, text_column, label_columns
   └── Output: context.raw_data

2. PreprocessingStep
   ├── Input: context.raw_data
   └── Output: context.preprocessed_texts

3. FeatureExtractionStep
   ├── Input: context.preprocessed_texts
   ├── Check cache first
   └── Output: context.features

4. DataSplittingStep
   ├── Input: context.features, context.labels
   └── Output: context.X_train, X_val, y_train, y_val

5. TrainingStep
   ├── Input: context.X_train, y_train
   └── Output: context.trained_classifier

6. EvaluationStep
   ├── Input: context.trained_classifier, X_val, y_val
   └── Output: context.evaluation_metrics

7. ModelSavingStep
   ├── Input: context.trained_classifier
   └── Output: Model saved to disk
```

### Prediction Flow

```
1. DataLoadingStep
   ├── Input: data (DataFrame hoặc filepath)
   └── Output: context.raw_data

2. PreprocessingStep
   ├── Input: context.raw_data
   └── Output: context.preprocessed_texts

3. FeatureExtractionStep
   ├── Input: context.preprocessed_texts
   ├── Check cache first
   └── Output: context.features

4. ModelLoadingStep
   ├── Input: model_path
   └── Output: context.loaded_classifier

5. PredictionStep
   ├── Input: context.loaded_classifier, context.features
   └── Output: context.predictions
```

---

## Configuration Management

### Centralized Config

Tất cả cấu hình được tập trung tại `config/`:

```python
# config/core/settings.py - Tổng hợp tất cả config
from config.core.paths import paths_config
from config.model.embedding import sbert_config, fasttext_config
from config.model.classifier import lightgbm_config
from config.training.data import data_config
from config.training.preprocessing import preprocessing_config
from config.training.cache import cache_config
from config.training.evaluation import evaluation_config
from config.training.trainer import trainer_config

class Settings:
    """Namespace tổng hợp tất cả config"""
    paths = paths_config
    sbert = sbert_config
    fasttext = fasttext_config
    classifier = lightgbm_config
    data = data_config
    preprocessing = preprocessing_config
    cache = cache_config
    evaluation = evaluation_config
    trainer = trainer_config

settings = Settings()
```

### Config Validation

Mỗi config class validate dữ liệu:

```python
@dataclass
class SBERTEmbeddingConfig:
    model_name: str
    max_seq_length: int
    device: str = 'cpu'

    def __post_init__(self):
        """Validate config sau khi khởi tạo"""
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length phải > 0")

        if self.device not in ['cpu', 'cuda', 'mps']:
            raise ValueError(f"Device không hợp lệ: {self.device}")
```

---

## Extensibility

### Thêm Embedding Method Mới

**Bước 1:** Tạo interface (nếu cần)

```python
# src/interfaces/embedding/i_new_embedding.py
class INewEmbedding(IEmbeddingExtractor):
    @abstractmethod
    def extract(self, texts: list[str]) -> np.ndarray:
        pass
```

**Bước 2:** Implement concrete class

```python
# src/services/embedding/new_embedding_extractor.py
class NewEmbeddingExtractor(INewEmbedding):
    def extract(self, texts: list[str]) -> np.ndarray:
        # Your implementation
        return embeddings
```

**Bước 3:** Đăng ký vào DI Container

```python
# src/app_setup.py
new_embedding = NewEmbeddingExtractor()
ServiceContainer.register_singleton('new_embedding', new_embedding)
```

**Bước 4:** Sử dụng trong service

```python
# src/services/embedding/extractor_service.py
class FeatureExtractorService:
    def __init__(
        self,
        sbert: ISBERTExtractor,
        fasttext: IFastTextExtractor,
        new_embedding: INewEmbedding  # ← Thêm dependency
    ):
        self.sbert = sbert
        self.fasttext = fasttext
        self.new_embedding = new_embedding
```

### Thêm Classifier Mới

Tương tự như trên, implement `IClassifier` interface.

### Thêm Pipeline Step Mới

```python
class CustomStep(IPipelineStep):
    @property
    def name(self) -> str:
        return "Custom Step"

    def execute(self, context: Context) -> Context:
        # Your logic
        return context

# Thêm vào pipeline
pipeline = TrainingPipeline([
    ...,
    CustomStep(),  # ← Thêm step mới
    ...
])
```

---

## Error Handling Strategy

### Validation Layer

```python
# src/validators/
class DataValidator:
    @staticmethod
    def validate_data(df: pd.DataFrame, text_col: str, label_cols: list):
        """Validate dữ liệu đầu vào"""
        if df.empty:
            raise ValueError("DataFrame rỗng")

        if text_col not in df.columns:
            raise ValueError(f"Không tìm thấy cột '{text_col}'")

        for col in label_cols:
            if col not in df.columns:
                raise ValueError(f"Không tìm thấy cột label '{col}'")
```

### Logging Strategy

```python
# src/services/logging/logger_service.py
class LoggerService:
    def __init__(self):
        self.logger = logging.getLogger('CommentClassification')

    def log_step(self, step_name: str, status: str, details: dict = None):
        """Log mỗi step trong pipeline"""
        message = f"[{step_name}] {status}"
        if details:
            message += f" | {details}"

        if status == 'SUCCESS':
            self.logger.info(message)
        elif status == 'ERROR':
            self.logger.error(message)
        else:
            self.logger.warning(message)
```

---

## Testing Architecture

### Unit Tests

Test từng component riêng biệt:

```python
# test/unit/test_preprocessor.py
def test_preprocessor_lowercase():
    preprocessor = PreprocessorService(config)

    result = preprocessor.process(['HELLO WORLD'])

    assert result == ['hello world']
```

### Integration Tests

Test interaction giữa các components:

```python
# test/integration/test_pipeline.py
def test_full_training_pipeline():
    # Setup DI Container
    setup_di_container()

    # Create pipeline
    pipeline = TrainingPipeline([...])

    # Execute
    context = pipeline.execute(TrainingContext(...))

    # Verify
    assert context.trained_classifier is not None
    assert context.evaluation_metrics['accuracy'] > 0.7
```

### Mocking Dependencies

```python
from unittest.mock import Mock

def test_extractor_with_mock():
    # Mock dependencies
    mock_sbert = Mock(spec=ISBERTExtractor)
    mock_sbert.extract.return_value = np.array([[1, 2, 3]])

    mock_fasttext = Mock(spec=IFastTextExtractor)
    mock_fasttext.extract.return_value = np.array([[4, 5, 6]])

    # Test với mock
    extractor = FeatureExtractorService(mock_sbert, mock_fasttext, None)
    result = extractor.extract_features(['test'])

    assert result.shape == (1, 6)
```

---

## Performance Considerations

### 1. Singleton cho Heavy Objects

Embedding models nặng → Singleton để tránh load nhiều lần:

```python
# ✅ GOOD: Singleton
ServiceContainer.register_singleton('sbert_extractor', SBERTExtractor())

# ❌ BAD: Factory (load lại mỗi lần)
ServiceContainer.register_factory('sbert_extractor', lambda: SBERTExtractor())
```

### 2. Feature Caching

Cache features để tránh trích xuất lại:

```python
class FeatureCacheService:
    def load_features(self, texts: list[str]) -> Optional[np.ndarray]:
        cache_key = self._generate_cache_key(texts)
        return self._load_from_disk(cache_key)
```

### 3. Lazy Loading

Load model chỉ khi cần:

```python
class LazyClassifier:
    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model
```

---

## Best Practices Summary

✅ **DO:**

- Luôn code to interfaces, không code to implementations
- Inject dependencies qua constructor
- Validate inputs ở mỗi layer boundary
- Log tất cả operations quan trọng
- Sử dụng type hints cho tất cả public methods
- Viết docstrings cho classes và methods
- Test với mock dependencies

❌ **DON'T:**

- Hard-code dependencies trong class
- Truy cập global state trực tiếp
- Bỏ qua error handling
- Tạo God classes (classes làm quá nhiều việc)
- Skip validation
- Forget to log errors

---

## Kết luận

Kiến trúc Comment Classification được thiết kế để:

1. **Dễ hiểu:** Clear separation of concerns, mỗi layer có trách nhiệm rõ ràng
2. **Dễ test:** Dependencies được inject, dễ mock
3. **Dễ mở rộng:** Thêm features mới không ảnh hưởng code cũ
4. **Dễ bảo trì:** SOLID principles đảm bảo code clean và maintainable
5. **Performance:** Singleton cho heavy objects, feature caching

---

**Tài liệu liên quan:**

- [README.md](./README.md) - Hướng dẫn sử dụng
- [API.md](./API.md) - API Documentation
