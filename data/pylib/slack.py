import os
from typing import Any

import requests


def send_alert(message: dict[str, Any]) -> None:
    url = os.getenv("SLACK_BATCH_LOGGING_URL", None)
    if url is not None:
        requests.post(url=url, json=message)  # pylint: disable=missing-timeout


def send_report(message: dict[str, Any]) -> None:
    url = os.getenv("SLACK_DATA_REPORT_URL", None)
    if url is not None:
        requests.post(url=url, json=message)  # pylint: disable=missing-timeout
