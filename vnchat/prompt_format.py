"""モデル別チャットテンプレートの判定とプロンプト整形。"""

from __future__ import annotations

from typing import Literal


ChatTemplateName = Literal["llama3", "chatml"]
ChatTemplateArg = Literal["auto", "llama3", "chatml"]


def resolve_chat_template(
    model_path: str,
    template_arg: ChatTemplateArg = "auto",
) -> ChatTemplateName:
    """モデル名とCLI指定値から利用テンプレートを決定する。"""
    if template_arg == "llama3":
        return "llama3"
    if template_arg == "chatml":
        return "chatml"

    normalized = (model_path or "").replace("\\", "/").lower()
    filename = normalized.split("/")[-1]

    if any(keyword in filename for keyword in ("qwen", "deepseek-r1-distill-qwen")):
        return "chatml"

    return "llama3"


def default_stop_tokens(template: ChatTemplateName) -> tuple[str, ...]:
    """テンプレートごとの既定 stop トークン群を返す。"""
    if template == "chatml":
        return ("<|im_end|>", "<|endoftext|>", "<|eot_id|>")
    return ("<|eot_id|>", "<|end_of_text|>")


def build_chat_prompt(
    messages: list[dict[str, str]],
    template: ChatTemplateName,
    add_generation_prompt: bool = True,
) -> str:
    """会話配列をテンプレートに従って文字列プロンプトへ直列化する。"""
    prompt_parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role not in ("system", "user", "assistant"):
            continue

        if template == "chatml":
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            continue

        prompt_parts.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        )

    if add_generation_prompt:
        if template == "chatml":
            prompt_parts.append("<|im_start|>assistant\n")
        else:
            prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

    return "".join(prompt_parts)
