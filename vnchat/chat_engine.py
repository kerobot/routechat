from __future__ import annotations

from colorama import Fore, Style

from vnchat.backends import LLMBackendAdapter
from vnchat.character import CharacterProfile
from vnchat.config import AppConfig, RuntimeTuning
from vnchat.conversation import ConversationManager
from vnchat.io_utils import safe_input
from vnchat.presentation import VNDisplayRenderer
from vnchat.state_logic import CharacterStateUpdater


class VisualNovelChat:
    def __init__(
        self,
        app_config: AppConfig,
        tuning: RuntimeTuning,
        profile: CharacterProfile,
        user_name: str,
    ):
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
            user_name=user_name, llm=self.llm, tuning=tuning
        )
        self.display_renderer = VNDisplayRenderer(user_name=user_name)
        self.state_updater = CharacterStateUpdater(
            conversation=self.conversation,
            profile=self.profile,
            tuning=self.tuning,
            llm=self.llm,
            user_name=self.user_name,
        )
        print(f"{Fore.GREEN}バックエンド初期化完了！{Style.RESET_ALL}")

    def start_conversation(self) -> None:
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
        raw = safe_input(f"{Fore.GREEN}{self.user_name} > {Style.RESET_ALL}")
        if raw is None:
            print(f"{Fore.CYAN}会話を終了します。{Style.RESET_ALL}")
            return None
        return raw.strip()

    def _handle_command(self, user_input: str) -> str:
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
        self.conversation.add_message("assistant", assistant_response)
        self.turn_count += 1
        self._update_character_state(user_input)

    def _generate_response(self) -> str:
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
            stop=["<|eot_id|>"],
            stream=False,
        )

        response = str(output.get("choices", [{}])[0].get("text", "")).strip()

        if self._needs_novel_retry(response):
            retry_system_prompt = (
                self.system_prompt
                + "\n\n"
                + (
                    "短すぎるので、ノベル風の描写量を少し増やして出力して。\n"
                    "厳密な固定フォーマットは不要だが、\n"
                    "- 情景や仕草の描写\n"
                    "- 竜胆のセリフ\n"
                    "- 会話後の空気や次の一手\n"
                    "の3要素が自然に含まれるように。\n"
                    "余計な前置きやメタ発言は禁止。"
                )
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
                repeat_penalty=self.tuning.generation_repeat_penalty,
                stop=["<|eot_id|>"],
                stream=False,
            )
            retry_text = str(
                retry_output.get("choices", [{}])[0].get("text", "")
            ).strip()
            if retry_text:
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

    def _print_vn_display(self, text: str) -> None:
        self.display_renderer.print_vn_display(text)

    def _build_prompt_with_generation_budget(
        self,
        reserve_tokens: int,
        system_prompt_override: str | None = None,
    ) -> str:
        all_messages = list(self.conversation.messages)
        system_prompt = system_prompt_override or self.system_prompt

        def build_with_keep(keep: int) -> str:
            keep = max(0, min(keep, len(all_messages)))
            trimmed = all_messages[-keep:] if keep > 0 else []
            context = self.conversation.get_context_for_llm(
                system_prompt, messages_override=trimmed
            )
            return self._build_prompt(context)

        prompt = build_with_keep(len(all_messages))
        try:
            prompt_tokens = len(self.llm.tokenize(prompt.encode("utf-8")))
        except Exception:
            return prompt

        if prompt_tokens <= (self.llm.n_ctx() - reserve_tokens):
            return prompt

        lo = 0
        hi = len(all_messages)
        best_prompt = build_with_keep(0)

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

        return best_prompt

    @staticmethod
    def _build_prompt(context: list[dict[str, str]]) -> str:
        prompt_parts: list[str] = []
        for msg in context:
            role = msg["role"]
            content = msg["content"]
            if role not in ("system", "user", "assistant"):
                continue
            prompt_parts.append(
                f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            )
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(prompt_parts)

    def _update_character_state(self, user_input: str) -> None:
        self.state_updater.update_character_state(
            turn_count=self.turn_count, user_input=user_input
        )
