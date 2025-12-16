class PreprocessingConfig:
    """Cấu hình cho text preprocessing."""
    
    PREPROCESSOR_CONFIG = {
        "remove_emojis": {"enabled": True},
        "lowercase": {"enabled": True},
        "remove_urls": {"enabled": True},
        "remove_emails": {"enabled": True},
        "remove_phone_numbers": {"enabled": True},
        "remove_special_chars": {"enabled": True, "pattern": r"[^\w\s]"},
        "remove_extra_whitespace": {"enabled": True},
        "unicode_normalize": {"enabled": True, "form": "NFC"}
    }

# Export singleton instance
preprocessing_config = PreprocessingConfig()
