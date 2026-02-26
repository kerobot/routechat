from __future__ import annotations

import argparse
import urllib.error
from dataclasses import dataclass

from vnchat.http_client import request_json


@dataclass
class ChatConfig:
    server_url: str
    model: str = ""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    timeout_sec: float = 120.0


def _post_json(url: str, payload: dict, timeout_sec: float) -> dict:
    return request_json(
        url=url,
        timeout_sec=timeout_sec,
        method="POST",
        payload=payload,
    )


def chat_completion_openai(
    config: ChatConfig, system_prompt: str, user_text: str
) -> str | None:
    url = f"{config.server_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": config.model or "local-model",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "stream": False,
    }

    try:
        result = _post_json(url, payload, timeout_sec=config.timeout_sec)
        choices = result.get("choices", [])
        if not choices:
            return None
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        return None
    except urllib.error.HTTPError as exc:
        print(f"[WARN] /v1/chat/completions HTTPError: {exc.code}")
        return None
    except Exception as exc:
        print(f"[WARN] /v1/chat/completions 失敗: {type(exc).__name__}: {exc}")
        return None


def completion_fallback(config: ChatConfig, prompt: str) -> str | None:
    url = f"{config.server_url.rstrip('/')}/completion"
    payload = {
        "prompt": prompt,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "n_predict": config.max_tokens,
    }

    try:
        result = _post_json(url, payload, timeout_sec=config.timeout_sec)
        content = result.get("content")
        if isinstance(content, str):
            return content.strip()
        return None
    except Exception as exc:
        print(f"[WARN] /completion 失敗: {type(exc).__name__}: {exc}")
        return None


def run_chat(config: ChatConfig) -> None:
    system_prompt = "あなたは有能な会話アシスタント。日本語で簡潔かつ自然に返答する。"

    print("[INFO] llama-server API チャットを開始")
    print(f"[INFO] endpoint: {config.server_url}")
    print("[INFO] 終了するには 'quit' を入力")

    while True:
        user_text = input("\nYou> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit"}:
            print("[INFO] 終了する")
            break

        answer = chat_completion_openai(config, system_prompt, user_text)

        if answer is None:
            fallback_prompt = (
                f"<|system|>\n{system_prompt}\n"
                f"<|user|>\n{user_text}\n"
                f"<|assistant|>\n"
            )
            answer = completion_fallback(config, fallback_prompt)

        if not answer:
            print("Assistant> [ERROR] 応答を取得できなかった")
            continue

        print(f"Assistant> {answer}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="llama-server API 経由チャット")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ChatConfig(
        server_url=args.server_url,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout_sec=args.timeout_sec,
    )
    run_chat(config)


if __name__ == "__main__":
    main()
