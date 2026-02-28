"""表示（プレゼンテーション）層。

LLMの出力を、ビジュアルノベルっぽく読みやすい形に整形して端末に表示する。
"""

from __future__ import annotations

import re

from colorama import Fore, Style


class VNDisplayRenderer:
    """ビジュアルノベル風にテキストを色分けしながら表示するレンダラ。"""

    def __init__(self, user_name: str) -> None:
        """レンダラを初期化する。

        Args:
            user_name (str): ユーザー名（会話ログの発話者判定に使う）。
        """
        self.user_name = user_name

    def print_vn_display(self, text: str) -> None:
        """テキストをノベル風に整形して表示する。

        - 地の文（情景描写）とセリフを色分け
        - 混在している行は分割して読みやすくする
        - 不要な見出しマーカーは表示前に除去
        """
        normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")

        # 不要なマーカーを削除
        for marker in ("【竜胆の動作・情景描写】", "【状況の変化】"):
            normalized = normalized.replace(marker, "")

        normalized = re.sub(r"([^\n])(竜胆[:：])", r"\1\n\2", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip("\n")

        narrative_style = f"{Fore.YELLOW}{Style.DIM}"
        dialogue_style = f"{Fore.LIGHTYELLOW_EX}{Style.NORMAL}"

        def is_dialogue_line(line: str) -> bool:
            stripped = (line or "").lstrip()
            if not stripped:
                return False
            if stripped.startswith(("竜胆：", "竜胆:")):
                return True
            if stripped.startswith((f"{self.user_name}：", f"{self.user_name}:")):
                return True
            if stripped.startswith(("「", "『")):
                return True
            return False

        def add_rindou_prefix_if_needed(line: str) -> str:
            if not line:
                return line
            lstripped = line.lstrip()
            if not lstripped.startswith(("「", "『")):
                return line
            if lstripped.startswith(("竜胆：", "竜胆:")):
                return line
            if lstripped.startswith((f"{self.user_name}：", f"{self.user_name}:")):
                return line
            leading_ws = line[: len(line) - len(lstripped)]
            return f"{leading_ws}竜胆：{lstripped}"

        for raw_line in normalized.split("\n"):
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            if not stripped:
                continue

            mix_idx = -1
            for token in ("竜胆：", "竜胆:"):
                idx = line.find(token)
                if idx != -1:
                    mix_idx = idx
                    break

            if mix_idx > 0:
                pre = line[:mix_idx].strip()
                dlg = line[mix_idx:].strip()
                if pre:
                    print(f"{narrative_style}{pre}{Style.RESET_ALL}")
                if dlg:
                    print(f"{dialogue_style}{dlg}{Style.RESET_ALL}")
                continue

            if is_dialogue_line(line):
                print(
                    f"{dialogue_style}{add_rindou_prefix_if_needed(line)}{Style.RESET_ALL}"
                )
            else:
                print(f"{narrative_style}{line}{Style.RESET_ALL}")
