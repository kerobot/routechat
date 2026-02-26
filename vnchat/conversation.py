from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from colorama import Fore, Style

from vnchat.config import RuntimeTuning
from vnchat.models import CharacterState, Message


class ConversationManager:
    def __init__(
        self,
        user_name: str,
        llm: Any,
        tuning: RuntimeTuning,
    ):
        self.messages: list[Message] = []
        self.summary_history: list[str] = []
        self.character_state = CharacterState()
        self.user_name = user_name
        self.llm = llm
        self.tuning = tuning

    def add_message(self, role: str, content: str) -> None:
        msg = Message(role=role, content=content)
        self.messages.append(msg)
        if role == "assistant":
            self.character_state.scene_count += 1

        if role == "assistant" and len(self.messages) > self.tuning.summary_threshold:
            self._trigger_summary()

    def _trigger_summary(self) -> None:
        print(f"\n{Fore.YELLOW}[システム] 会話履歴を要約中...{Style.RESET_ALL}")

        system_msg = [m for m in self.messages if m.role == "system"]
        recent_messages = self.messages[-(self.tuning.max_history // 2) :]
        to_summarize = self.messages[len(system_msg) : -len(recent_messages)]

        if to_summarize:
            summary_text = self._create_summary(to_summarize)
            self.summary_history.append(summary_text)
            self.messages = system_msg + recent_messages

            print(
                f"{Fore.GREEN}[システム] 要約完了（{len(to_summarize)}件を要約）{Style.RESET_ALL}\n"
            )

    def _create_summary(self, messages: list[Message]) -> str:
        if self.llm is not None:
            try:
                summary = self._create_summary_with_llm(messages)
                if summary:
                    return summary
            except Exception:
                pass

        summary_parts: list[str] = []
        for msg in messages:
            if msg.role == "user":
                summary_parts.append(f"{self.user_name}: {msg.content[:50]}...")
            elif msg.role == "assistant":
                summary_parts.append(f"竜胆: {msg.content[:50]}...")

        return "\n".join(summary_parts)

    def _create_summary_with_llm(self, messages: list[Message]) -> str:
        if self.llm is None:
            return ""

        def _format_messages_for_summary(max_chars_per_message: int = 400) -> str:
            lines: list[str] = []
            for msg in messages:
                if msg.role not in ("user", "assistant"):
                    continue
                speaker = self.user_name if msg.role == "user" else "竜胆"
                content = (
                    (msg.content or "")
                    .strip()
                    .replace("\r\n", "\n")
                    .replace("\r", "\n")
                )
                if len(content) > max_chars_per_message:
                    content = content[:max_chars_per_message] + "..."
                lines.append(f"{speaker}: {content}")
            return "\n".join(lines) if lines else "(なし)"

        dialogue = _format_messages_for_summary()
        system = (
            "あなたは会話要約器。入力の会話ログを、後続の会話生成に使える形で日本語で要約する。\n"
            "制約: 出力は要約本文のみ。前置き・説明・コードブロック禁止。\n"
            "要約方針:\n"
            "- 重要な出来事/話題/ユーザーの意図/決定事項を残す\n"
            "- 竜胆の反応（温度感、心境の変化）が分かる一言を入れる\n"
            "- 未解決の宿題/次にやること（次の一手）を1つ残す\n"
            "- 箇条書き 5〜8 行程度で簡潔に\n"
        )
        user = "以下の会話ログを要約して。\n\n" "## 会話ログ\n" f"{dialogue}\n"

        prompt = (
            f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        output = self.llm(
            prompt,
            max_tokens=300,
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.05,
            stop=["<|eot_id|>"],
            stream=False,
        )

        text = str(output.get("choices", [{}])[0].get("text", "")).strip()
        return re.sub(r"^要約[:：]\s*", "", text).strip()

    def get_context_for_llm(
        self, system_prompt: str, messages_override: list[Message] | None = None
    ) -> list[dict[str, str]]:
        system_parts = [system_prompt]

        if self.summary_history:
            recent_summaries = self.summary_history[
                -self.tuning.max_summaries_in_context :
            ]
            system_parts.append(
                "\n\n## これまでの会話要約\n" + "\n---\n".join(recent_summaries)
            )

        system_parts.append(
            f"\n\n## 現在の状態\n{self.character_state.to_string()}\n"
            f"シーン数: {self.character_state.scene_count}"
        )

        context: list[dict[str, str]] = [
            {"role": "system", "content": "".join(system_parts)}
        ]

        messages = messages_override if messages_override is not None else self.messages
        for msg in messages:
            if msg.role != "system":
                context.append({"role": msg.role, "content": msg.content})

        return context

    def save_to_file(self, filename: str) -> None:
        data = {
            "messages": [asdict(m) for m in self.messages],
            "summary_history": self.summary_history,
            "character_state": asdict(self.character_state),
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"{Fore.GREEN}会話履歴を保存しました: {filename}{Style.RESET_ALL}")
