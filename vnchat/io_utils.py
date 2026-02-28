"""入出力の薄いユーティリティ。

標準入力が中断されたケース（Ctrl+C / EOF）を安全に扱うための関数を提供する。
"""

from __future__ import annotations


def safe_input(prompt: str) -> str | None:
    """安全に `input()` を呼び出す。

    Args:
        prompt (str): 表示するプロンプト。

    Returns:
        str | None: 入力文字列。割り込みやEOFの場合はNone。
    """
    try:
        return input(prompt)
    except (KeyboardInterrupt, EOFError):
        return None
