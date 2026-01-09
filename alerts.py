# alerts.py
import json
from typing import Tuple

import requests


def _clean_webhook(url: str) -> str:
    if not url:
        return ""
    url = url.strip()
    # remove accidental surrounding quotes
    if (url.startswith('"') and url.endswith('"')) or (url.startswith("'") and url.endswith("'")):
        url = url[1:-1].strip()
    return url


def send_discord_alert(
    webhook_url: str,
    content: str,
    username: str = "Crypto Market Engine",
    timeout: int = 10,
) -> Tuple[bool, str]:
    """
    Send a Discord webhook message.
    Returns: (ok, message). Never raises.
    """
    webhook_url = _clean_webhook(webhook_url)

    if not webhook_url:
        return False, "DISCORD_WEBHOOK_URL is missing/empty."

    payload = {
        "content": content[:1900],  # Discord limit is 2000; leave headroom
        "username": username,
    }

    try:
        r = requests.post(webhook_url, json=payload, timeout=timeout)
        if 200 <= r.status_code < 300:
            return True, f"Sent OK (HTTP {r.status_code})."

        extra = ""
        try:
            extra = json.dumps(r.json())
        except Exception:
            extra = (r.text or "").strip()

        return False, f"Discord rejected (HTTP {r.status_code}): {extra}"
    except requests.RequestException as e:
        return False, f"Request failed: {type(e).__name__}: {e}"
