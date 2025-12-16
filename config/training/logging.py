class LoggingConfig:
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Export singleton instance
logging_config = LoggingConfig()
