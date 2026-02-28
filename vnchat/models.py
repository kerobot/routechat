"""会話データ・状態管理で使うデータモデル定義。

- `Message`: 会話の1メッセージ
- `CharacterState`: キャラクターの状態（好感度/温度感/心境など）
- `StateAnalysis`: 状態分析の結果（LLM出力）
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TypedDict


@dataclass
class Message:
    """会話履歴の1要素（roleとcontentを持つ）。"""

    role: str
    content: str
    timestamp: Optional[str] = None

    def __post_init__(self) -> None:
        """timestamp未指定の場合に現在時刻を補完する。"""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class CharacterState:
    """キャラクターの内面状態を表す（会話ごとに変化する）。"""

    affection: float = 3.0
    temperature: float = 0.35
    mood: str = "通常"
    scene_count: int = 0

    def to_string(self) -> str:
        """状態を表示用の短い文字列に整形する。"""
        return (
            f"好感度: {self.affection}/10 | "
            f"温度感: {self.temperature:.2f} | 現在の心境: {self.mood}"
        )


class StateAnalysis(TypedDict, total=False):
    """LLMなどの分析器が返す状態分析結果の型。

    全キーが常に揃うとは限らないため `total=False`。
    """

    mood: str
    affection_delta: float
    temperature: float
    confidence: float
