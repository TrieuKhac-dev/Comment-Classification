from typing import List
import logging

from src.interfaces.logging import IListener, ILoggerService


class LoggerService(ILoggerService):
    def __init__(self, name: str = "CommentClassification", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self._listeners: List[IListener] = []

    def attach(self, listener: IListener):
        if listener not in self._listeners:
            self._listeners.append(listener)

    def detach(self, listener: IListener):
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify(self, level: str, message: str, *args, **kwargs):
        for listener in self._listeners:
            try:
                listener.update(level, message, *args, **kwargs)
            except Exception as e:
                self.logger.error(f"Lỗi khi notify listener: {e}")

    def log(self, level: str, message: str, **kwargs):
        log_method = getattr(self.logger, level, None)
        if callable(log_method):
            log_method(message, **kwargs)
        self._notify(level, message, **kwargs)

    def info(self, message: str): self.log("info", message)
    def warning(self, message: str): self.log("warning", message)
    def error(self, message: str, exc_info: bool = False): 
        self.log("error", message, exc_info=exc_info)
    def debug(self, message: str): self.log("debug", message)
    def exception(self, message: str): 
        self.log("exception", message, exc_info=True)