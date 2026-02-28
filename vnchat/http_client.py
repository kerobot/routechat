"""HTTPでJSONを送受信する小さなヘルパ。

llama-server などHTTP APIを叩く処理を、標準ライブラリだけで扱えるようにする。
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, cast


def request_json(
    url: str,
    timeout_sec: float,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """HTTPリクエストを実行し、レスポンスJSONを辞書として返す。

    Raises:
        ValueError: HTTPステータスが200以外、またはJSON形式が想定外の場合。
    """
    data: bytes | None = None
    headers: dict[str, str] = {}

    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )

    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        if response.status != 200:
            raise ValueError(f"unexpected status: {response.status}")

        body = response.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise ValueError("response is not a JSON object")
        return cast(dict[str, Any], parsed)


def get_json_safe(url: str, timeout_sec: float = 5.0) -> dict[str, Any] | None:
    """GETでJSONを取得する（失敗時はNone）。"""
    try:
        return request_json(url=url, timeout_sec=timeout_sec, method="GET")
    except Exception:
        return None


def post_json_safe(
    url: str, payload: dict[str, Any], timeout_sec: float = 15.0
) -> dict[str, Any] | None:
    """POSTでJSONを送信し、JSONを受け取る（失敗時はNone）。"""
    try:
        return request_json(
            url=url,
            timeout_sec=timeout_sec,
            method="POST",
            payload=payload,
        )
    except Exception:
        return None
