"""チャットアプリの中核ロジック（入力→生成→表示→状態更新）を担うモジュール。"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from colorama import Fore, Style

from vnchat.backends import LLMBackendAdapter
from vnchat.character import CharacterProfile
from vnchat.config import AppConfig, RuntimeTuning
from vnchat.conversation import ConversationManager
from vnchat.io_utils import safe_input
from vnchat.presentation import VNDisplayRenderer
from vnchat.prompt_format import build_chat_prompt
from vnchat.state_logic import CharacterStateUpdater


class VisualNovelChat:
    """ビジュアルノベル風ロールプレイチャットの実行エンジン。"""

    _REPETITIVE_PROP_WORDS = (
        "マグカップ",
        "コーヒー",
        "資料",
        "視線",
        "目線",
        "口元",
    )
    _STYLE_TEMPLATE_MARKERS = (
        "低いトーンで",
        "小さく呟き",
        "口元には",
        "視線を",
        "微かに笑み",
    )
    _UNLIKELY_ACTION_MARKERS = (
        "一気飲み",
        "自身が口元に持っていった",
    )
    # 疑問符なしで会話継続を促すソフトフックワード群
    _SOFT_HOOK_PHRASES = (
        "どう思う",
        "どうする",
        "どうかな",
        "どうした",
        "どうだった",
        "何かある",
        "何か言",
        "してみたら",
        "してみて",
        "言ってみて",
        "話してみて",
        "聞かせて",
        "来てみたら",
        "来てみて",
        "気になる",
        "してほしい",
        "してみよう",
        "しようか",
        "伝えて",
        "教えて",
    )

    def __init__(
        self,
        app_config: AppConfig,
        tuning: RuntimeTuning,
        profile: CharacterProfile,
        user_name: str,
    ):
        """バックエンド・会話管理・表示・状態更新を初期化する。"""
        self.user_name = user_name
        self.app_config = app_config
        self.tuning = tuning
        self.profile = profile

        self.system_prompt = profile.system_prompt_template.format(user_name=user_name)
        self.opening_scene = profile.opening_scene_template.format(user_name=user_name)

        self.turn_count = 0

        print(f"{Fore.CYAN}LLMバックエンドを初期化中...{Style.RESET_ALL}")
        self.llm = LLMBackendAdapter(
            backend_mode=app_config.backend_mode,
            n_ctx=app_config.n_ctx,
            model_path=app_config.model_path,
            n_gpu_layers=app_config.n_gpu_layers,
            server_url=app_config.api_server_url,
            api_timeout_sec=app_config.api_timeout_sec,
        )
        self.conversation = ConversationManager(
            user_name=user_name,
            llm=self.llm,
            tuning=tuning,
            chat_template=self.app_config.chat_template,
            stop_tokens=self.app_config.stop_tokens,
        )
        self.display_renderer = VNDisplayRenderer(user_name=user_name)
        self.state_updater = CharacterStateUpdater(
            conversation=self.conversation,
            profile=self.profile,
            tuning=self.tuning,
            llm=self.llm,
            user_name=self.user_name,
            chat_template=self.app_config.chat_template,
            stop_tokens=self.app_config.stop_tokens,
        )
        print(f"{Fore.GREEN}バックエンド初期化完了！{Style.RESET_ALL}")

    def start_conversation(self) -> None:
        """オープニングを表示して会話セッションを開始する。"""
        print(f"{Fore.WHITE}{'='*60}{Style.RESET_ALL}")
        print(
            f"{Fore.WHITE}  ビジュアルノベル風ロールプレイチャット：ルートチャット{Style.RESET_ALL}"
        )
        print(
            f"{Fore.WHITE}  - IT企業のダウナー系クールだけど面倒見がいい{self.profile.name}先輩 - {Style.RESET_ALL}"
        )
        print(f"{Fore.WHITE}{'='*60}{Style.RESET_ALL}")

        print(f"{Fore.CYAN}[シーンを開始します...]{Style.RESET_ALL}")
        self._print_vn_display(self.opening_scene)
        self.conversation.add_message("assistant", self.opening_scene)

        print(
            f"{Fore.GREEN}コマンド: 'quit'で終了 | 'save'で保存 | 'state'で状態確認 | 'summary'で要約確認{Style.RESET_ALL}"
        )
        print(f"{Fore.WHITE}{'-'*60}{Style.RESET_ALL}")

    def chat_loop(self) -> None:
        """ユーザー入力をループで受け、会話を継続する。"""
        self.start_conversation()

        while True:
            user_input = self._read_user_input()
            if user_input is None:
                break
            if user_input == "":
                continue

            command_action = self._handle_command(user_input)
            if command_action == "break":
                break
            if command_action == "continue":
                continue

            self.conversation.add_message("user", user_input)
            response = self._generate_response()

            self._print_vn_display(response)
            print(f"{Fore.WHITE}{'-'*60}{Style.RESET_ALL}")
            self._finalize_turn(user_input=user_input, assistant_response=response)

    def _read_user_input(self) -> str | None:
        """ユーザー入力を1回ぶん読み取る（中断時はNone）。"""
        raw = safe_input(f"{Fore.GREEN}{self.user_name} > {Style.RESET_ALL}")
        if raw is None:
            print(f"{Fore.CYAN}会話を終了します。{Style.RESET_ALL}")
            return None
        return raw.strip()

    def _handle_command(self, user_input: str) -> str:
        """組み込みコマンド（quit/save/state/summary）を処理する。"""
        cmd = user_input.lower()
        if cmd == "quit":
            print(f"{Fore.CYAN}会話を終了します。{Style.RESET_ALL}")
            save_raw = safe_input("会話履歴を保存しますか？ (y/n): ")
            if (save_raw or "n").strip().lower() == "y":
                self.conversation.save_to_file(self.app_config.save_file)
            return "break"

        if cmd == "save":
            self.conversation.save_to_file(self.app_config.save_file)
            return "continue"

        if cmd == "state":
            print(f"{Fore.YELLOW}[状態]{Style.RESET_ALL}")
            print(self.conversation.character_state.to_string())
            print(f"会話履歴数: {len(self.conversation.messages)}")
            print(f"要約履歴数: {len(self.conversation.summary_history)}")
            return "continue"

        if cmd == "summary":
            if not self.conversation.summary_history:
                print(f"{Fore.YELLOW}[要約] まだ要約はありません。{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}[これまでの会話要約]{Style.RESET_ALL}")
                for i, s in enumerate(self.conversation.summary_history, start=1):
                    print(f"{Fore.CYAN}--- 要約 {i} ---{Style.RESET_ALL}")
                    print(s)
            return "continue"

        return "none"

    def _finalize_turn(self, user_input: str, assistant_response: str) -> None:
        """応答表示後に履歴・ターン数・状態更新を反映する。"""
        self.conversation.add_message("assistant", assistant_response)
        self.turn_count += 1
        self._update_character_state(user_input)

    def _generate_response(self) -> str:
        """プロンプトを構築してLLMから応答を生成する。"""
        prompt = self._build_prompt_with_generation_budget(
            reserve_tokens=self.tuning.reserve_tokens
        )

        print(f"{Fore.CYAN}[竜胆が考えています...]{Style.RESET_ALL}")
        output = self.llm(
            prompt,
            max_tokens=self.tuning.generation_max_tokens,
            temperature=self.tuning.generation_temperature,
            top_p=self.tuning.generation_top_p,
            top_k=self.tuning.generation_top_k,
            repeat_penalty=self.tuning.generation_repeat_penalty,
            stop=list(self.app_config.stop_tokens),
            stream=False,
        )

        response = str(output.get("choices", [{}])[0].get("text", "")).strip()
        response = self._postprocess_response(response)
        base_score = self._response_quality_score(response)

        retry_reasons = self._collect_retry_reasons(response)
        if retry_reasons:
            retry_system_prompt = (
                self.system_prompt
                + "\n\n"
                + self._build_retry_instruction(retry_reasons)
            )
            retry_prompt = self._build_prompt_with_generation_budget(
                reserve_tokens=self.tuning.reserve_tokens,
                system_prompt_override=retry_system_prompt,
            )
            retry_output = self.llm(
                retry_prompt,
                max_tokens=self.tuning.generation_max_tokens,
                temperature=self.tuning.generation_temperature,
                top_p=self.tuning.generation_top_p,
                top_k=self.tuning.generation_top_k,
                repeat_penalty=min(
                    1.35,
                    self.tuning.generation_repeat_penalty
                    + self.tuning.retry_repeat_penalty_boost,
                ),
                stop=list(self.app_config.stop_tokens),
                stream=False,
            )
            retry_text = str(
                retry_output.get("choices", [{}])[0].get("text", "")
            ).strip()
            if retry_text:
                retry_text = self._postprocess_response(retry_text)
                retry_score = self._response_quality_score(retry_text)
                if retry_score <= base_score:
                    response = retry_text

        if not response:
            try:
                prompt_tokens = len(self.llm.tokenize(prompt.encode("utf-8")))
            except Exception:
                prompt_tokens = -1
            finish_reason = output.get("choices", [{}])[0].get("finish_reason")
            print(
                f"{Fore.YELLOW}[システム] 空の応答でした。"
                f"prompt_tokens={prompt_tokens}, finish_reason={finish_reason}{Style.RESET_ALL}"
            )

        return response

    def _needs_novel_retry(self, text: str) -> bool:
        """応答が短すぎる等の場合に再生成すべきか判定する。"""
        if not text:
            return True

        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        lines = [line.strip() for line in normalized.split("\n") if line.strip()]

        # 1行だけ、または極端に短い応答のみ再試行する（3段の厳格強制はしない）
        if len(lines) <= 1:
            return True
        if len(normalized) < 80:
            return True

        return False

    def _collect_retry_reasons(self, response: str) -> list[str]:
        """再生成すべき理由を収集する。"""
        reasons: list[str] = []
        if self._needs_novel_retry(response):
            reasons.append("short")
        if self._is_parrot_like_response(response):
            reasons.append("parrot")
        if self._is_user_quote_repetition(response):
            reasons.append("user_quote")
        if self._is_self_repetitive_response(response):
            reasons.append("self_repeat")
        if self._is_repetitive_prop_response(response):
            reasons.append("prop_repeat")
        if self._is_context_mismatch_response(response):
            reasons.append("context")
        if self._is_template_like_response(response):
            reasons.append("style_template")
        if self._is_fragmented_response(response):
            reasons.append("broken")
        if self._has_unwanted_marker(response):
            reasons.append("marker")
        if self._contains_wrong_character_name(response):
            reasons.append("name_error")
        if self._contains_unlikely_action(response):
            reasons.append("unlikely_action")
        if self._is_scene_replay_response(response):
            reasons.append("scene_replay")
        if self._is_no_hook_response(response):
            reasons.append("no_hook")
        return reasons

    def _build_retry_instruction(self, reasons: list[str]) -> str:
        """再生成時の追加入力を組み立てる。"""
        lines: list[str] = []

        if "short" in reasons:
            lines.extend(
                [
                    "短すぎるので、ノベル風の描写量を少し増やして出力して。",
                    "厳密な固定フォーマットは不要だが、",
                    "- 情景や仕草の描写",
                    "- 竜胆のセリフ",
                    "- 会話後の空気や次の一手",
                    "の3要素が自然に含まれるように。",
                ]
            )

        if "parrot" in reasons:
            lines.append(
                "直前のユーザー発話の言い換え・反復は避け、新しい情報を1つ以上含めること。"
            )

        if "user_quote" in reasons:
            lines.append(
                "ユーザー発話を引用・復唱しないで、要点を咀嚼して先輩としての返答を行うこと。"
            )

        if "self_repeat" in reasons:
            lines.append(
                "前ターンの竜胆の言い回しを繰り返さず、動作描写・語彙・展開を変えること。"
            )

        if "prop_repeat" in reasons:
            lines.append(
                "同じ小道具や同一所作の連続使用を避け、場面の変化を入れること。"
            )

        if "context" in reasons:
            lines.append(
                "直近会話アンカーから最低1つを具体的に回収して応答に織り込むこと。"
            )

        if "style_template" in reasons:
            lines.append(
                "直近2ターンと同じ文型テンプレートを繰り返さず、文の運びと描写の軸を変えること。"
            )

        if "broken" in reasons:
            lines.append("意味が崩れた断片文を避け、文として自然に完結させること。")

        if "marker" in reasons:
            lines.append(
                "「（次）」「次は？」「地の文」「次のアクション」のようなメタ的ラベルは出力しないこと。"
            )

        if "name_error" in reasons:
            lines.append(
                "キャラクター名は必ず『竜胆』を使い、誤字・類似表記（例: 竜齢）は禁止。"
            )

        if "scene_replay" in reasons:
            lines.append(
                "オープニングや既出の長文描写を再掲せず、現在ターンの会話を前に進めること。"
            )

        if "unlikely_action" in reasons:
            lines.append(
                "キャラ設定に合わない過剰行動（例: 一気飲み）を避け、自然な所作に留めること。"
            )

        if "no_hook" in reasons:
            lines.append(
                "応答の末尾に、会話を続けるための小さなフックを必ず1つ加えること。\n"
                "フックの種類は「軽い問いかけ」「提案」「さりげない確認」のどれかでよい。\n"
                "二択の問いかけ（「AかBか」形式）は多用せず、ときどきにすること。"
            )

        lines.append("余計な前置きやメタ発言は禁止。")
        return "\n".join(lines)

    def _is_parrot_like_response(self, response: str) -> bool:
        """応答がオウム返し傾向かを判定する。"""
        last_user = ""
        for msg in reversed(self.conversation.messages):
            if msg.role == "user":
                last_user = (msg.content or "").strip()
                break

        if not last_user or not response:
            return False

        left = self._normalize_for_similarity(last_user)
        right = self._normalize_for_similarity(response)
        if not left or not right:
            return False

        similarity = SequenceMatcher(None, left, right).ratio()
        return similarity >= self.tuning.echo_similarity_threshold

    def _is_template_like_response(self, response: str) -> bool:
        """直近と同じ描写テンプレートが連投されていないかを判定する。"""
        if not response:
            return False

        current_markers = {
            m for m in self._STYLE_TEMPLATE_MARKERS if m in (response or "")
        }
        if len(current_markers) < 2:
            return False

        recent_assistant = [
            (msg.content or "")
            for msg in self.conversation.messages
            if msg.role == "assistant"
        ][-2:]

        overlap_count = 0
        for text in recent_assistant:
            prior_markers = {m for m in self._STYLE_TEMPLATE_MARKERS if m in text}
            if len(current_markers & prior_markers) >= 2:
                overlap_count += 1

        return overlap_count >= 1

    def _is_self_repetitive_response(self, response: str) -> bool:
        """直前のassistant応答と似すぎていないかを判定する。"""
        last_assistant = ""
        for msg in reversed(self.conversation.messages):
            if msg.role == "assistant":
                last_assistant = (msg.content or "").strip()
                break

        if not last_assistant or not response:
            return False

        left = self._normalize_for_similarity(last_assistant)
        right = self._normalize_for_similarity(response)
        if not left or not right:
            return False

        similarity = SequenceMatcher(None, left, right).ratio()
        return similarity >= self.tuning.echo_similarity_threshold

    def _is_user_quote_repetition(self, response: str) -> bool:
        """ユーザー発話の丸写し・半引用が含まれるか判定する。"""
        last_user = ""
        for msg in reversed(self.conversation.messages):
            if msg.role == "user":
                last_user = (msg.content or "").strip()
                break

        if not last_user or not response:
            return False

        normalized_user = self._normalize_for_similarity(last_user)
        normalized_resp = self._normalize_for_similarity(response)
        if len(normalized_user) >= 8 and normalized_user in normalized_resp:
            return True

        quoted_texts = re.findall(r"[「\"]([^」\"]{6,})[」\"]", response)
        for quoted in quoted_texts:
            q = self._normalize_for_similarity(quoted)
            if not q:
                continue
            similarity = SequenceMatcher(None, normalized_user, q).ratio()
            if similarity >= 0.78:
                return True

        return False

    def _is_repetitive_prop_response(self, response: str) -> bool:
        """同一小道具の連続使用を簡易検知する。"""
        if not response:
            return False

        last_assistant = ""
        for msg in reversed(self.conversation.messages):
            if msg.role == "assistant":
                last_assistant = msg.content or ""
                break
        if not last_assistant:
            return False

        current_hits = {w for w in self._REPETITIVE_PROP_WORDS if w in (response or "")}
        previous_hits = {
            w for w in self._REPETITIVE_PROP_WORDS if w in (last_assistant or "")
        }
        return len(current_hits & previous_hits) > 0

    def _is_context_mismatch_response(self, response: str) -> bool:
        """直近文脈を回収できているかを判定する。"""
        anchors = self.conversation.get_recent_context_hints(
            max_messages=4,
            max_keywords=8,
            include_assistant_fallback=False,
        )
        if not anchors:
            return False

        normalized = (response or "").lower()
        hits = sum(1 for word in anchors if word in normalized)
        return hits < self.tuning.context_keyword_min_hits

    def _is_scene_replay_response(self, response: str) -> bool:
        """オープニングや直前の長文を再掲していないかを判定する。"""
        if not response:
            return False

        left = self._normalize_for_similarity(self.opening_scene)
        right = self._normalize_for_similarity(response)
        if not left or not right:
            return False

        if len(right) > 80 and right[:80] == left[:80]:
            return True

        similarity = SequenceMatcher(None, left, right).ratio()
        return similarity >= 0.82

    def _is_no_hook_response(self, response: str) -> bool:
        """会話フック（問いかけ・誘い）がない閉じた応答を検知する。

        疑問符もソフトフックワードも含まない応答は、ターンを自然に渡せていない
        可能性が高いため再生成候補とする。
        超短応答（"short" で処理済み）は対象外。
        """
        if not response:
            return False

        # _needs_novel_retry() が True になる超短応答は "short" retry で対処済みなのでスキップ
        if self._needs_novel_retry(response):
            return False

        # 疑問符があればフックあり
        if "?" in response or "？" in response:
            return False

        # ソフトフックワードがあればフックあり
        if any(w in response for w in self._SOFT_HOOK_PHRASES):
            return False

        return True

    @staticmethod
    def _has_unwanted_marker(response: str) -> bool:
        """メタ的な継続マーカー混入を検知する。"""
        if not response:
            return False
        markers = (
            "（次）",
            "(次)",
            "次は？",
            "（続く）",
            "(続く)",
            "次のアクション",
            "next action",
            "地の文",
            "地の文には",
            "地の文では",
        )
        return any(marker in response for marker in markers)

    @staticmethod
    def _contains_wrong_character_name(response: str) -> bool:
        """竜胆の誤記を検知する。"""
        if not response:
            return False
        wrong_forms = ("竜齢", "龍齢", "竜令", "竜齋", "竜淡")
        return any(name in response for name in wrong_forms)

    def _contains_unlikely_action(self, response: str) -> bool:
        """キャラ設定から外れた不自然な行動語を検知する。"""
        if not response:
            return False
        return any(marker in response for marker in self._UNLIKELY_ACTION_MARKERS)

    def _postprocess_response(self, response: str) -> str:
        """表示前にメタ行除去・誤記補正を行う。"""
        if not response:
            return ""

        fixed = response
        typo_map = {
            "竜齢": "竜胆",
            "龍齢": "竜胆",
            "竜令": "竜胆",
            "竜齋": "竜胆",
            "竜淡": "竜胆",
        }
        for wrong, correct in typo_map.items():
            fixed = fixed.replace(wrong, correct)

        cleaned_lines: list[str] = []
        for raw_line in fixed.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            if any(marker in line for marker in self._UNLIKELY_ACTION_MARKERS):
                continue
            lowered = line.lower()
            if line.startswith("次のアクション") or lowered.startswith("next action"):
                continue
            if line.startswith("地の文"):
                continue
            if line in ("（次）", "(次)", "（続く）", "(続く)"):
                continue
            if (line.startswith("（") and line.endswith("）")) or (
                line.startswith("(") and line.endswith(")")
            ):
                # メタ的な括弧行は出力から除去（短い説明行が多いため）
                if len(line) <= 40 or "視線" in line or "操作" in line:
                    continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    def _response_quality_score(self, response: str) -> int:
        """応答品質をざっくり数値化（低いほど良い）。"""
        reasons = self._collect_retry_reasons(response)
        weights = {
            "short": 2,
            "parrot": 2,
            "user_quote": 2,
            "self_repeat": 2,
            "prop_repeat": 1,
            "context": 1,
            "style_template": 2,
            "broken": 2,
            "marker": 2,
            "name_error": 3,
            "unlikely_action": 2,
            "scene_replay": 3,
            "no_hook": 1,
        }
        return sum(weights.get(reason, 1) for reason in reasons)

    @staticmethod
    def _is_fragmented_response(response: str) -> bool:
        """不自然な断片文・機械的な重複を簡易検知する。"""
        if not response:
            return True

        normalized = response.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return True

        bad_patterns = (
            "内心は",
            "間違ってないけど",
            "けど",
        )
        if (
            any(pattern in normalized for pattern in bad_patterns)
            and len(normalized) < 60
        ):
            return True

        lines = [line.strip() for line in normalized.split("\n") if line.strip()]
        if len(lines) >= 2:
            for i in range(len(lines) - 1):
                a = VisualNovelChat._normalize_for_similarity(lines[i])
                b = VisualNovelChat._normalize_for_similarity(lines[i + 1])
                if a and b and a == b:
                    return True

        return False

    @staticmethod
    def _normalize_for_similarity(text: str) -> str:
        """文字列類似度比較向けに正規化する。"""
        normalized = (text or "").lower()
        normalized = re.sub(r"\s+", "", normalized)
        normalized = re.sub(
            r"[。、，,.!！?？『』「」\"'（）()【】\[\]…ー-]", "", normalized
        )
        return normalized

    def _print_vn_display(self, text: str) -> None:
        """表示レンダラに委譲してノベル風表示を行う。"""
        self.display_renderer.print_vn_display(text)

    def _build_prompt_with_generation_budget(
        self,
        reserve_tokens: int,
        system_prompt_override: str | None = None,
    ) -> str:
        """コンテキスト長を超えないよう履歴を間引きつつプロンプトを構築する。"""
        all_messages = list(self.conversation.messages)
        system_prompt = system_prompt_override or self.system_prompt

        def build_with_keep(keep: int) -> str:
            # 末尾keep件だけ履歴を残してプロンプトを組み直す。
            # keepが範囲外でも安全にクランプする。
            keep = max(0, min(keep, len(all_messages)))
            trimmed = all_messages[-keep:] if keep > 0 else []
            context = self.conversation.get_context_for_llm(
                system_prompt, messages_override=trimmed
            )
            return self._build_prompt(context)

        # まずは履歴を全保持したプロンプトを試す。
        prompt = build_with_keep(len(all_messages))
        try:
            prompt_tokens = len(self.llm.tokenize(prompt.encode("utf-8")))
        except Exception:
            return prompt

        # 生成用にreserve_tokensを確保した上で収まるなら、そのまま採用。
        if prompt_tokens <= (self.llm.n_ctx() - reserve_tokens):
            return prompt

        # 収まらない場合は「残す履歴件数」を二分探索で最大化する。
        # 条件: keepを増やすほどトークン数は基本的に増える（単調性を仮定）。
        # なので mid が収まるなら右側(より多く残す)を探索し、
        # 収まらないなら左側(さらに削る)を探索する。
        min_keep = max(
            0, min(self.tuning.min_recent_messages_to_keep, len(all_messages))
        )
        lo = 0
        hi = len(all_messages)
        # 最低限のフォールバック（履歴0件）を初期解として持っておく。
        best_prompt = build_with_keep(0)

        min_candidate = build_with_keep(min_keep)
        try:
            min_tokens = len(self.llm.tokenize(min_candidate.encode("utf-8")))
            if min_tokens <= (self.llm.n_ctx() - reserve_tokens):
                lo = min_keep
                best_prompt = min_candidate
        except Exception:
            pass

        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = build_with_keep(mid)
            try:
                cand_tokens = len(self.llm.tokenize(candidate.encode("utf-8")))
            except Exception:
                hi = mid - 1
                continue

            if cand_tokens <= (self.llm.n_ctx() - reserve_tokens):
                best_prompt = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        # 制約を満たす中で「最も多く履歴を残せる」プロンプトを返す。
        return best_prompt

    def _build_prompt(self, context: list[dict[str, str]]) -> str:
        """会話コンテキストを選択中テンプレート形式で直列化する。"""
        return build_chat_prompt(
            messages=context,
            template=self.app_config.chat_template,
            add_generation_prompt=True,
        )

    def _update_character_state(self, user_input: str) -> None:
        """ユーザー入力に基づいてキャラクター状態を更新する。"""
        self.state_updater.update_character_state(
            turn_count=self.turn_count, user_input=user_input
        )
