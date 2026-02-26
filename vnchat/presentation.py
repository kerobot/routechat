from __future__ import annotations

import re

from colorama import Fore, Style


class VNDisplayRenderer:
    def __init__(self, user_name: str) -> None:
        self.user_name = user_name

    def print_vn_display(self, text: str) -> None:
        scene_marker = "【竜胆の動作・情景描写】"
        change_marker = "【状況の変化】"
        normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")

        normalized = normalized.replace(scene_marker, f"\n{scene_marker}\n")
        normalized = normalized.replace(change_marker, f"\n{change_marker}\n")
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
            if stripped in (scene_marker, change_marker):
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
