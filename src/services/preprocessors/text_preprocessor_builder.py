from src.services.preprocessors.preprocess_step_base import PreprocessStepBase
from src.services.preprocessors.text_preprocessor_service import TextPreprocessorService
from src.services.preprocessors.preprocessing_steps import (
    UnicodeNormalizeStep,
    LowercaseStep,
    RemoveUrlStep,
    RemoveEmailStep,
    RemovePhoneStep,
    RemoveEmojiStep,
    RemoveSpecialCharsStep,
    KeepVietnameseCharsStep,
    NormalizeWhitespaceStep
)

class TextPreprocessorBuilder:
    def __init__(self):
        self._config = {}
        self._steps: list[PreprocessStepBase] = []
    
    def _add_step(self, step: PreprocessStepBase, config_key: str, value=True):
        self._steps.append(step)
        self._config[config_key] = value
        return self
    
    def with_unicode_normalize(self, enabled: bool = True, form: str = "NFC"):
        if enabled:
            return self._add_step(UnicodeNormalizeStep(form), "unicode_normalize", True)
        return self
    
    def with_lowercase(self, enabled: bool = True):
        if enabled:
            return self._add_step(LowercaseStep(), "lowercase", True)
        return self
    
    def with_url_removal(self, enabled: bool = True):
        if enabled:
            return self._add_step(RemoveUrlStep(), "remove_urls", True)
        return self
    
    def with_email_removal(self, enabled: bool = True):
        if enabled:
            return self._add_step(RemoveEmailStep(), "remove_emails", True)
        return self
    
    def with_phone_removal(self, enabled: bool = True):
        if enabled:
            return self._add_step(RemovePhoneStep(), "remove_phone_numbers", True)
        return self
    
    def with_emoji_removal(self, enabled: bool = True):
        if enabled:
            return self._add_step(RemoveEmojiStep(), "remove_emojis", True)
        return self
    
    def with_special_chars_removal(self, enabled: bool = True):
        if enabled:
            return self._add_step(RemoveSpecialCharsStep(), "remove_special_chars", True)
        return self
    
    def with_whitespace_normalization(self, enabled: bool = True):
        if enabled:
            return self._add_step(NormalizeWhitespaceStep(), "remove_extra_whitespace", True)
        return self
    
    def with_vietnamese_chars(self, enabled: bool = True, pattern: str = None):
        if enabled and pattern:
            return self._add_step(KeepVietnameseCharsStep(pattern), "keep_vietnamese_chars", True)
        return self
    
    def build(self, config: dict = None) -> TextPreprocessorService:
        if config:
            self._config.update(config)
        return TextPreprocessorService(self._steps, self._config.copy())
