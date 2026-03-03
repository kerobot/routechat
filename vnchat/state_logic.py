"""キャラクター状態（信頼度・親密度・温度感・心境）を更新するロジック。

軽量なヒューリスティック更新と、LLMによる分析更新を状況に応じて使い分ける。
"""

from __future__ import annotations

import json
import re
from typing import Any

from vnchat.character import CharacterProfile
from vnchat.config import RuntimeTuning
from vnchat.conversation import ConversationManager
from vnchat.models import CharacterState, StateAnalysis


class CharacterStateUpdater:
    """会話の流れからキャラクター状態を更新する。"""

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
    _ROMANCE_WORDS = (
        "会いたい",
        "好き",
        "デート",
        "二人",
        "一緒に",
        "嫉妬",
        "手をつな",
    )

    def __init__(
        self,
        conversation: ConversationManager,
        profile: CharacterProfile,
        tuning: RuntimeTuning,
        llm: Any,
        user_name: str,
    ) -> None:
        """状態更新器を初期化する。"""
        self.conversation = conversation
        self.profile = profile
        self.tuning = tuning
        self.llm = llm
        self.user_name = user_name

    def update_character_state(self, turn_count: int, user_input: str) -> None:
        """ユーザー入力に応じて状態を更新する（LLM分析 or 軽量更新）。"""
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
        """簡易ルールで状態を更新する（LLMを使わない）。"""
        if self._contains_any(user_input, self._POSITIVE_WORDS):
            state.trust = self._clamp(state.trust + 0.18, 0.0, 10.0)
            state.intimacy = self._clamp(state.intimacy + 0.08, 0.0, 10.0)
            if state.mood not in ("苛立ち", "警戒"):
                state.mood = "満更でもない"
            state.temperature = self._clamp(state.temperature + 0.02, 0.0, 1.0)
        elif self._contains_any(user_input, self._NEGATIVE_WORDS):
            state.trust = self._clamp(state.trust - 0.10, 0.0, 10.0)
            if state.mood not in ("苛立ち", "警戒"):
                state.mood = "心配"
            state.temperature = self._clamp(state.temperature + 0.02, 0.0, 1.0)
        elif self._contains_any(user_input, self._ROMANCE_WORDS):
            state.intimacy = self._clamp(state.intimacy + 0.18, 0.0, 10.0)
            if state.mood in ("通常", "満更でもない", "好意的"):
                state.mood = "意識してる"
            state.temperature = self._clamp(state.temperature + 0.03, 0.0, 1.0)
        else:
            state.temperature = self._clamp(state.temperature * 0.99, 0.0, 1.0)

        self._finalize_state_update(
            state=state,
            trust_delta=None,
            intimacy_delta=None,
            romance_signal=None,
            user_input=user_input,
        )

    def _apply_analysis_result(
        self, state: CharacterState, analysis: StateAnalysis, user_input: str
    ) -> None:
        """LLM分析結果を状態に反映する。"""
        trust_delta_value: float | None = None
        intimacy_delta_value: float | None = None

        trust_delta = analysis.get("trust_delta")
        if isinstance(trust_delta, (int, float)):
            trust_delta_value = self._clamp(float(trust_delta), -1.0, 1.0)
            state.trust = self._clamp(state.trust + trust_delta_value, 0.0, 10.0)

        intimacy_delta = analysis.get("intimacy_delta")
        if isinstance(intimacy_delta, (int, float)):
            intimacy_delta_value = self._clamp(float(intimacy_delta), -1.0, 1.0)
            state.intimacy = self._clamp(
                state.intimacy + intimacy_delta_value, 0.0, 10.0
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

        signal_value = analysis.get("romance_signal")
        romance_signal_value = (
            float(signal_value) if isinstance(signal_value, (int, float)) else None
        )

        self._finalize_state_update(
            state=state,
            trust_delta=trust_delta_value,
            intimacy_delta=intimacy_delta_value,
            romance_signal=romance_signal_value,
            user_input=user_input,
        )

    def _update_state_fallback(self, state: CharacterState, user_input: str) -> None:
        """分析に失敗した場合のフォールバック更新。"""
        if self._contains_any(user_input, self._POSITIVE_WORDS):
            state.trust = self._clamp(state.trust + 0.30, 0.0, 10.0)
            state.intimacy = self._clamp(state.intimacy + 0.12, 0.0, 10.0)
            state.mood = "満更でもない"
            state.temperature = self._clamp(state.temperature + 0.05, 0.0, 1.0)
        elif self._contains_any(user_input, self._NEGATIVE_WORDS):
            state.trust = self._clamp(state.trust - 0.20, 0.0, 10.0)
            state.mood = "心配"
            state.temperature = self._clamp(state.temperature + 0.03, 0.0, 1.0)
        elif self._contains_any(user_input, self._ROMANCE_WORDS):
            state.intimacy = self._clamp(state.intimacy + 0.25, 0.0, 10.0)
            state.mood = "照れ隠し"
            state.temperature = self._clamp(state.temperature + 0.05, 0.0, 1.0)
        else:
            state.mood = "通常"
            state.temperature = self._clamp(state.temperature * 0.98, 0.0, 1.0)

        self._finalize_state_update(
            state=state,
            trust_delta=None,
            intimacy_delta=None,
            romance_signal=None,
            user_input=user_input,
        )

    @staticmethod
    def _contains_any(text: str, words: tuple[str, ...]) -> bool:
        """文字列に指定語群のいずれかが含まれるか判定する。"""
        return any(word in (text or "") for word in words)

    def _finalize_state_update(
        self,
        state: CharacterState,
        trust_delta: float | None,
        intimacy_delta: float | None,
        romance_signal: float | None,
        user_input: str,
    ) -> None:
        """状態更新後のドリフト/上書き処理をまとめて適用する。"""
        self._apply_intimacy_drift(
            state=state,
            mood=state.mood,
            trust_delta=trust_delta,
            intimacy_delta=intimacy_delta,
            user_input=user_input,
        )
        self._apply_intimacy_overrides(state)
        self._apply_romance_stage(state, user_input, romance_signal)

    def _resolve_proposed_mood(
        self,
        state: CharacterState,
        proposed_mood: Any,
        conf_value: float | None,
    ) -> str | None:
        """分析器の提案心境を、許容リストやヒステリシスを考慮して採用/却下する。"""
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

        if (
            proposed_mood in ("意識してる", "照れ隠し", "素直", "独占欲")
            and conf_value is not None
            and conf_value < 0.45
        ):
            return None

        if proposed_mood == "クーデレ" and state.intimacy < (
            self.tuning.affection_kuudere_threshold - 0.30
        ):
            return "好意的"

        if proposed_mood == "素直" and state.romance_stage < 3:
            return "照れ隠し"

        if proposed_mood == "独占欲" and state.romance_stage < 4:
            return "意識してる"

        return proposed_mood

    def _apply_intimacy_drift(
        self,
        state: CharacterState,
        mood: str,
        trust_delta: float | None,
        intimacy_delta: float | None,
        user_input: str,
    ) -> None:
        """明示的変化が弱いときに、文脈に応じた親密度ドリフトを加える。"""
        if isinstance(intimacy_delta, (int, float)) and intimacy_delta > 0.05:
            return

        intimacy_bump = 0.0
        trust_bump = 0.0

        if mood in ("クーデレ", "素直", "独占欲"):
            intimacy_bump += 0.12
        elif mood in ("好意的", "意識してる", "照れ隠し"):
            intimacy_bump += 0.08
        elif mood in ("満更でもない", "安心", "興味"):
            intimacy_bump += 0.05

        if self._contains_any(user_input, self._POSITIVE_WORDS):
            trust_bump += 0.06

        if self._contains_any(user_input, self._ROMANCE_WORDS):
            intimacy_bump += 0.10

        if isinstance(trust_delta, (int, float)) and trust_delta < -0.2:
            trust_bump *= 0.3

        if intimacy_bump > 0.0:
            state.intimacy = self._clamp(state.intimacy + intimacy_bump, 0.0, 10.0)
        if trust_bump > 0.0:
            state.trust = self._clamp(state.trust + trust_bump, 0.0, 10.0)

    def _apply_intimacy_overrides(self, state: CharacterState) -> None:
        """親密度閾値に応じて心境（好意的/クーデレ）を段階的に上書きする。"""
        hysteresis = 0.35
        soft_moods = (
            "通常",
            "満更でもない",
            "安心",
            "興味",
            "意識してる",
            "照れ隠し",
            "好意的",
            "クーデレ",
            "素直",
        )

        if state.mood == "クーデレ" and state.intimacy >= (
            self.tuning.affection_kuudere_threshold - hysteresis
        ):
            return
        if state.mood == "好意的" and state.intimacy >= (
            self.tuning.affection_favorable_threshold - hysteresis
        ):
            return

        if state.intimacy >= self.tuning.affection_kuudere_threshold:
            if state.mood in soft_moods and state.mood not in ("苛立ち", "警戒"):
                state.mood = "クーデレ"
            return

        if state.intimacy >= self.tuning.affection_favorable_threshold:
            if state.mood in (
                "通常",
                "満更でもない",
                "安心",
                "興味",
                "意識してる",
                "照れ隠し",
            ):
                state.mood = "好意的"

    def _apply_romance_stage(
        self,
        state: CharacterState,
        user_input: str,
        romance_signal: float | None,
    ) -> None:
        """信頼/親密度/会話内容から恋愛段階を進める（段階は下げない）。"""
        stage = state.romance_stage
        romance_hint = self._contains_any(user_input, self._ROMANCE_WORDS) or (
            isinstance(romance_signal, (int, float)) and romance_signal >= 0.55
        )

        if stage < 1 and state.trust >= 4.5 and state.intimacy >= 3.2:
            state.romance_stage = 1
            return

        if stage < 2 and state.trust >= 5.2 and state.intimacy >= 4.5 and romance_hint:
            state.romance_stage = 2
            return

        if stage < 3 and state.trust >= 6.2 and state.intimacy >= 6.0 and romance_hint:
            state.romance_stage = 3
            return

        if stage < 4 and state.trust >= 7.0 and state.intimacy >= 7.4 and romance_hint:
            state.romance_stage = 4

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        """値を[min_value, max_value]に収める。"""
        return max(min_value, min(max_value, value))

    def _format_recent_dialogue(self, max_messages: int = 6) -> str:
        """直近の会話履歴を分析用に短く整形する。"""
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
        """テキストから最初のJSONオブジェクトを抽出する（失敗時はNone）。"""
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
        """LLMで状態分析（mood/trust_delta/intimacy_delta等）を行う。"""
        state = self.conversation.character_state
        recent_dialogue = self._format_recent_dialogue(max_messages=8)
        current_state = (
            f"trust={state.trust:.2f}, intimacy={state.intimacy:.2f}, "
            f"temperature={state.temperature:.2f}, mood={state.mood}, "
            f"romance_stage={state.romance_stage}, scene_count={state.scene_count}"
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

        trust_delta = parsed.get("trust_delta")
        if isinstance(trust_delta, (int, float)):
            analysis["trust_delta"] = float(trust_delta)

        intimacy_delta = parsed.get("intimacy_delta")
        if isinstance(intimacy_delta, (int, float)):
            analysis["intimacy_delta"] = float(intimacy_delta)

        # 旧スキーマ（affection_delta）との後方互換
        affection_delta = parsed.get("affection_delta")
        if "intimacy_delta" not in analysis and isinstance(
            affection_delta, (int, float)
        ):
            analysis["intimacy_delta"] = float(affection_delta)

        temperature = parsed.get("temperature")
        if isinstance(temperature, (int, float)):
            analysis["temperature"] = float(temperature)

        romance_signal = parsed.get("romance_signal")
        if isinstance(romance_signal, (int, float)):
            analysis["romance_signal"] = self._clamp(float(romance_signal), 0.0, 1.0)

        if isinstance(confidence, (int, float)):
            analysis["confidence"] = float(confidence)

        if (
            "mood" not in analysis
            and "trust_delta" not in analysis
            and "intimacy_delta" not in analysis
            and "temperature" not in analysis
            and "romance_signal" not in analysis
        ):
            return None

        return analysis
