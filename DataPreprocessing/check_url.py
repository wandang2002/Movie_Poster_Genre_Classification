import requests
import time

# ---------------------------
# URL Validation Function
# ---------------------------
def is_url_active(url, max_retries=3):
    """Check if a URL is accessible (status code 200)"""
    for _ in range(max_retries):
        try:
            response = requests.head(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException as e:
            print(f"URL check failed for {url}: {e}")
            time.sleep(1)
    return False