import re
import unicodedata

from src.services.preprocessors.preprocess_step_base import (
    PreprocessStepBase,
    StepPriority
)

class UnicodeNormalizeStep(PreprocessStepBase):
    priority = StepPriority.LOWEST
    
    def __init__(self, form: str = "NFC", conflicts: list[str] = None, priority: StepPriority = None):
        super().__init__(conflicts, priority)
        self.form = form
    
    def apply(self, text: str) -> str:
        return unicodedata.normalize(self.form, text)


class LowercaseStep(PreprocessStepBase):
    priority = StepPriority.HIGH
    
    def __init__(self, conflicts: list[str] = None, priority: StepPriority = None):
        super().__init__(conflicts, priority)
    
    def apply(self, text: str) -> str:
        return text.lower()


class RemoveUrlStep(PreprocessStepBase):
    priority = StepPriority.MEDIUM
    
    def __init__(self, conflicts: list[str] = None, priority: StepPriority = None):
        super().__init__(conflicts, priority)
        self.pattern = re.compile(r"https?://\S+")
    
    def apply(self, text: str) -> str:
        return self.pattern.sub(" ", text)


class RemoveEmailStep(PreprocessStepBase):
    priority = StepPriority.MEDIUM
    
    def __init__(self, conflicts: list[str] = None, priority: StepPriority = None):
        super().__init__(conflicts, priority)
        self.pattern = re.compile(r"\S+@\S+")
    
    def apply(self, text: str) -> str:
        return self.pattern.sub(" ", text)

class RemoveNonVietnameseStep(PreprocessStepBase):
    priority = StepPriority.HIGHEST
    
    def apply(self, text: str) -> str:
        # Giữ lại chữ cái Latin + tiếng Việt, bỏ ký tự ngoài
        pattern = re.compile(r"[^a-zA-ZÀ-ỹ\s]")
        return pattern.sub(" ", text)

class RemovePhoneStep(PreprocessStepBase):
    priority = StepPriority.MEDIUM
    
    def __init__(self, conflicts: list[str] = None, priority: StepPriority = None):
        super().__init__(conflicts, priority)
        self.pattern = re.compile(r"(\+84|0)[0-9]{9,10}")
    
    def apply(self, text: str) -> str:
        return self.pattern.sub(" ", text)

class RemoveEmojiStep(PreprocessStepBase):
    priority = StepPriority.HIGH
    
    def __init__(self, conflicts: list[str] = None, priority: StepPriority = None):
        super().__init__(conflicts, priority)

    def apply(self, text: str) -> str:
        import emoji
        # Loại bỏ toàn bộ emoji bằng thư viện emoji
        text = emoji.replace_emoji(text, "")
        # Loại bỏ ký tự điều khiển
        result = []
        for c in text:
            code = ord(c)
            if code >= 0x20 or code in (0x09, 0x0A):
                result.append(c)
        return "".join(result)

class RemoveSpecialCharsStep(PreprocessStepBase):
    priority = StepPriority.LOW
    conflicts = ["KeepVietnameseCharsStep"]
    
    def __init__(self, conflicts: list[str] = None, priority: StepPriority = None):
        super().__init__(conflicts, priority)
        self.pattern = re.compile(r"[^\w\s]")
    
    def apply(self, text: str) -> str:
        return self.pattern.sub(" ", text)


class KeepVietnameseCharsStep(PreprocessStepBase):
    priority = StepPriority.LOW
    conflicts = ["RemoveSpecialCharsStep"]
    
    def __init__(self, pattern: str, conflicts: list[str] = None, priority: StepPriority = None):
        super().__init__(conflicts, priority)
        self.pattern = re.compile(pattern)
    
    def apply(self, text: str) -> str:
        return self.pattern.sub(" ", text)


class NormalizeWhitespaceStep(PreprocessStepBase):
    priority = StepPriority.LOWEST
    
    def __init__(self, conflicts: list[str] = None, priority: StepPriority = None):
        super().__init__(conflicts, priority)
        self.pattern = re.compile(r"\s+")
    
    def apply(self, text: str) -> str:
        return self.pattern.sub(" ", text).strip()
