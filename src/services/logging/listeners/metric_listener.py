from datetime import datetime, timezone

from src.interfaces.logging.i_listener import IListener

class MetricLoggerListener(IListener):
    def __init__(self, metric_client):
        self.metric_client = metric_client
        
    def update(self, level: str, message: str, *args, **kwargs):
        try:
            # Ví dụ: tăng số lượng log theo level
            self.metric_client.inc_log_count(level)
            # Gửi thêm thông tin chi tiết về log cho metric client (nếu cần)
            self.metric_client.log_event(
                level=level,
                message=message,
                timestamp=datetime.now(timezone.utc).isoformat(),
                extra=kwargs.get("extra", {})
            )
        except Exception as e:
            print(f"[MetricLoggerListener] Lỗi ghi metric: {e}")