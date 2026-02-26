from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TypedDict


@dataclass
class Message:
    role: str
    content: str
    timestamp: Optional[str] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class CharacterState:
    affection: float = 3.0
    temperature: float = 0.35
    mood: str = "通常"
    scene_count: int = 0

    def to_string(self) -> str:
        return (
            f"好感度: {self.affection}/10 | "
            f"温度感: {self.temperature:.2f} | 現在の心境: {self.mood}"
        )


class StateAnalysis(TypedDict, total=False):
    mood: str
    affection_delta: float
    temperature: float
    confidence: float
