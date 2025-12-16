from typing import Dict, Type, TypeVar, Generic

T = TypeVar("T")

class GenericFactory(Generic[T]):
    _registry: Dict[str, Type[T]] = {}

    @classmethod
    def register(cls, name: str, impl_cls: Type[T]):
        cls._registry[name] = impl_cls

    @classmethod
    def get(cls, name: str) -> Type[T]:
        if name not in cls._registry:
            valid = list(cls._registry.keys())
            raise ValueError(f"'{name}' does not exist. Valid: {valid}")
        return cls._registry[name]

    @classmethod
    def create(cls, name: str) -> T:
        impl_cls = cls.get(name)
        return impl_cls() 