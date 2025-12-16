import re
import unicodedata

from src.services.preprocessors.preprocess_step_base import (
    PreprocessStepBase,
    StepPriority
)

class UnicodeNormalizeStep(PreprocessStepBase):
    priority = StepPriority.HIGHEST
    
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
        self.pattern = re.compile(r"http[s]?://\S+")
    
    def apply(self, text: str) -> str:
        return self.pattern.sub(" ", text)


class RemoveEmailStep(PreprocessStepBase):
    priority = StepPriority.MEDIUM
    
    def __init__(self, conflicts: list[str] = None, priority: StepPriority = None):
        super().__init__(conflicts, priority)
        self.pattern = re.compile(r"\S+@\S+")
    
    def apply(self, text: str) -> str:
        return self.pattern.sub(" ", text)


class RemovePhoneStep(PreprocessStepBase):
    priority = StepPriority.MEDIUM
    
    def __init__(self, conflicts: list[str] = None, priority: StepPriority = None):
        super().__init__(conflicts, priority)
        self.pattern = re.compile(r"(\+84|0)[0-9]{9,10}")
    
    def apply(self, text: str) -> str:
        return self.pattern.sub(" ", text)


class RemoveEmojiStep(PreprocessStepBase):
    priority = StepPriority.MEDIUM
    
    def __init__(self, conflicts: list[str] = None, priority: StepPriority = None):
        super().__init__(conflicts, priority)
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002700-\U000027BF"
            "\U0001F900-\U0001F9FF"
            "\U00002600-\U000026FF"
            "\U00002B50"
            "\U0000231A"
            "\U0001FA70-\U0001FAFF"
            "\U000025A0-\U000025FF"
            "\U0001F700-\U0001F77F"
            "\U0001F780-\U0001F7FF"
            "\U0001F800-\U0001F8FF"
            "\U0001F000-\U0001F02F"
            "\U0001F0A0-\U0001F0FF"
            "\u200d"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"
            "\u3030"
            "]+",
            flags=re.UNICODE
        )

    def apply(self, text: str) -> str:
        # Loại bỏ emoji/icon bằng regex mạnh
        text = self.emoji_pattern.sub("", text)
        # Loại bỏ ký tự điều khiển (0x00-0x1F trừ \t, \n)
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
