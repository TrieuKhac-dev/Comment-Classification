from src.interfaces.logging.i_listener import IListener

class FileLoggerListener(IListener):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def update(self, level: str, message: str, *args, **kwargs):
        try:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(f"[{level.upper()}] {message}\n")
        except Exception as e:
            # Avoid infinite log loop, just print to console
            print(f"[FileLoggerListener] Error writing log to file: {e}")