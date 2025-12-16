import requests
from typing import Dict, List

class CommentClassificationClient:
    """Client để tương tác với Comment Classification API"""
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')

    def health_check(self) -> Dict:
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def predict(self, comments: Dict[str, str]) -> Dict:
        response = requests.post(
            f"{self.base_url}/predict",
            json={"comments": comments},
            timeout=60
        )
        response.raise_for_status()
        return response.json()

    def predict_simple(self, comments: List[str]) -> Dict:
        response = requests.post(
            f"{self.base_url}/predict/simple",
            json=comments,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
