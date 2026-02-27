from __future__ import annotations

import json
import re
from typing import Any

from vnchat.character import CharacterProfile
from vnchat.config import RuntimeTuning
from vnchat.conversation import ConversationManager
from vnchat.models import CharacterState, StateAnalysis


class CharacterStateUpdater:
    _POSITIVE_WORDS = (
        "ありがとう",
        "嬉しい",
        "嬉しかった",
        "助かる",
        "助かった",
        "助かりました",
        "最高",
        "大好き",
        "素晴らしい",
        "すごい",
    )
    _NEGATIVE_WORDS = ("ダメ", "辛い", "無理")

    def __init__(
        self,
        conversation: ConversationManager,
        profile: CharacterProfile,
        tuning: RuntimeTuning,
        llm: Any,
        user_name: str,
    ) -> None:
        self.conversation = conversation
        self.profile = profile
        self.tuning = tuning
        self.llm = llm
        self.user_name = user_name

    def update_character_state(self, turn_count: int, user_input: str) -> None:
        state = self.conversation.character_state

        force_eval = any(
            k in (user_input or "") for k in self.tuning.force_eval_keywords
        )
        do_eval = force_eval or (
            self.tuning.state_eval_every_n_turns > 0
            and (
                turn_count == 1
                or (turn_count % self.tuning.state_eval_every_n_turns == 0)
            )
        )

        analysis: StateAnalysis | None = None
        if do_eval:
            analysis = self._analyze_state_with_llm(user_input=user_input)
        else:
            self._update_state_lightweight(state=state, user_input=user_input)
            return

        if analysis is not None:
            self._apply_analysis_result(
                state=state, analysis=analysis, user_input=user_input
            )
            return

        self._update_state_fallback(state=state, user_input=user_input)

    def _update_state_lightweight(self, state: CharacterState, user_input: str) -> None:
        if self._contains_any(user_input, self._POSITIVE_WORDS):
            state.affection = self._clamp(state.affection + 0.15, 0.0, 10.0)
            if state.mood not in ("苛立ち", "警戒"):
                state.mood = "満更でもない"
            state.temperature = self._clamp(state.temperature + 0.02, 0.0, 1.0)
        elif self._contains_any(user_input, self._NEGATIVE_WORDS):
            if state.mood not in ("苛立ち", "警戒"):
                state.mood = "心配"
            state.temperature = self._clamp(state.temperature + 0.02, 0.0, 1.0)
        else:
            state.temperature = self._clamp(state.temperature * 0.99, 0.0, 1.0)

        self._finalize_state_update(
            state=state,
            affection_delta=None,
            user_input=user_input,
        )

    def _apply_analysis_result(
        self, state: CharacterState, analysis: StateAnalysis, user_input: str
    ) -> None:
        affection_delta_value: float | None = None
        affection_delta = analysis.get("affection_delta")
        if isinstance(affection_delta, (int, float)):
            affection_delta_value = self._clamp(float(affection_delta), -1.0, 1.0)
            state.affection = self._clamp(
                state.affection + affection_delta_value, 0.0, 10.0
            )

        temperature = analysis.get("temperature")
        if isinstance(temperature, (int, float)):
            temperature = self._clamp(float(temperature), 0.0, 1.0)
            state.temperature = (state.temperature * 0.7) + (temperature * 0.3)

        proposed_mood = analysis.get("mood")
        confidence = analysis.get("confidence")
        conf_value = float(confidence) if isinstance(confidence, (int, float)) else None

        next_mood = self._resolve_proposed_mood(
            state=state,
            proposed_mood=proposed_mood,
            conf_value=conf_value,
        )
        if next_mood is not None:
            state.mood = next_mood

        self._finalize_state_update(
            state=state,
            affection_delta=affection_delta_value,
            user_input=user_input,
        )

    def _update_state_fallback(self, state: CharacterState, user_input: str) -> None:
        if self._contains_any(user_input, self._POSITIVE_WORDS):
            state.affection = min(10, state.affection + 0.5)
            state.mood = "満更でもない"
            state.temperature = self._clamp(state.temperature + 0.05, 0.0, 1.0)
        elif self._contains_any(user_input, self._NEGATIVE_WORDS):
            state.mood = "心配"
            state.temperature = self._clamp(state.temperature + 0.03, 0.0, 1.0)
        else:
            state.mood = "通常"
            state.temperature = self._clamp(state.temperature * 0.98, 0.0, 1.0)

        self._finalize_state_update(
            state=state,
            affection_delta=None,
            user_input=user_input,
        )

    @staticmethod
    def _contains_any(text: str, words: tuple[str, ...]) -> bool:
        return any(word in (text or "") for word in words)

    def _finalize_state_update(
        self,
        state: CharacterState,
        affection_delta: float | None,
        user_input: str,
    ) -> None:
        self._apply_affection_drift(
            state=state,
            mood=state.mood,
            affection_delta=affection_delta,
            user_input=user_input,
        )
        self._apply_affection_overrides(state)

    def _resolve_proposed_mood(
        self,
        state: CharacterState,
        proposed_mood: Any,
        conf_value: float | None,
    ) -> str | None:
        if not (
            isinstance(proposed_mood, str)
            and proposed_mood in self.profile.allowed_moods
        ):
            return None

        if (
            proposed_mood in ("苛立ち", "警戒")
            and conf_value is not None
            and conf_value < 0.55
        ):
            return None

        if (
            proposed_mood in ("好意的", "クーデレ")
            and conf_value is not None
            and conf_value < 0.35
        ):
            return None

        if proposed_mood == "クーデレ" and state.affection < (
            self.tuning.affection_kuudere_threshold - 0.30
        ):
            return "好意的"

        return proposed_mood

    def _apply_affection_drift(
        self,
        state: CharacterState,
        mood: str,
        affection_delta: float | None,
        user_input: str,
    ) -> None:
        if isinstance(affection_delta, (int, float)) and affection_delta > 0.05:
            return

        bump = 0.0
        if mood in ("クーデレ", "好意的"):
            bump += 0.12
        elif mood in ("満更でもない", "安心", "興味"):
            bump += 0.06

        if self._contains_any(user_input, self._POSITIVE_WORDS):
            bump += 0.08

        if bump > 0.0:
            state.affection = self._clamp(state.affection + bump, 0.0, 10.0)

    def _apply_affection_overrides(self, state: CharacterState) -> None:
        hysteresis = 0.35
        soft_moods = ("通常", "満更でもない", "安心", "興味", "好意的", "クーデレ")

        if state.mood == "クーデレ" and state.affection >= (
            self.tuning.affection_kuudere_threshold - hysteresis
        ):
            return
        if state.mood == "好意的" and state.affection >= (
            self.tuning.affection_favorable_threshold - hysteresis
        ):
            return

        if state.affection >= self.tuning.affection_kuudere_threshold:
            if state.mood in soft_moods and state.mood not in ("苛立ち", "警戒"):
                state.mood = "クーデレ"
            return

        if state.affection >= self.tuning.affection_favorable_threshold:
            if state.mood in ("通常", "満更でもない", "安心", "興味"):
                state.mood = "好意的"

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))

    def _format_recent_dialogue(self, max_messages: int = 6) -> str:
        items: list[str] = []
        for msg in self.conversation.messages[-max_messages:]:
            if msg.role == "user":
                content = (msg.content or "").strip()
                if len(content) > 220:
                    content = content[:220] + "..."
                items.append(f"{self.user_name}: {content}")
            elif msg.role == "assistant":
                content = (msg.content or "").strip()
                if len(content) > 260:
                    content = content[:260] + "..."
                items.append(f"竜胆: {content}")

        return "\n".join(items) if items else "(なし)"

    def _extract_first_json_object(self, text: str) -> dict[str, Any] | None:
        if not text:
            return None
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        candidate = cleaned[start : end + 1]
        try:
            obj = json.loads(candidate)
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        return obj

    def _analyze_state_with_llm(self, user_input: str) -> StateAnalysis | None:
        state = self.conversation.character_state
        recent_dialogue = self._format_recent_dialogue(max_messages=8)
        current_state = (
            f"affection={state.affection:.2f}, temperature={state.temperature:.2f}, "
            f"mood={state.mood}, scene_count={state.scene_count}"
        )

        analyzer_prompt = self.profile.state_analyzer_prompt_template.format(
            user_name=self.user_name,
            current_state=current_state,
            recent_dialogue=recent_dialogue,
        )

        prompt = (
            f"<|start_header_id|>system<|end_header_id|>\n\n{analyzer_prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        output = self.llm(
            prompt,
            max_tokens=self.tuning.analyzer_max_tokens,
            temperature=self.tuning.analyzer_temperature,
            top_p=self.tuning.analyzer_top_p,
            top_k=self.tuning.analyzer_top_k,
            repeat_penalty=self.tuning.analyzer_repeat_penalty,
            stop=["<|eot_id|>"],
            stream=False,
        )

        text = str(output.get("choices", [{}])[0].get("text", "")).strip()
        parsed = self._extract_first_json_object(text)
        if parsed is None:
            return None

        confidence = parsed.get("confidence")
        if isinstance(confidence, (int, float)) and float(confidence) < 0.20:
            return None

        analysis: StateAnalysis = {}
        mood = parsed.get("mood")
        if isinstance(mood, str):
            analysis["mood"] = mood

        affection_delta = parsed.get("affection_delta")
        if isinstance(affection_delta, (int, float)):
            analysis["affection_delta"] = float(affection_delta)

        temperature = parsed.get("temperature")
        if isinstance(temperature, (int, float)):
            analysis["temperature"] = float(temperature)

        if isinstance(confidence, (int, float)):
            analysis["confidence"] = float(confidence)

        if (
            "mood" not in analysis
            and "affection_delta" not in analysis
            and "temperature" not in analysis
        ):
            return None

        return analysis
