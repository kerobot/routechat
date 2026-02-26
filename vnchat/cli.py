from __future__ import annotations

import os

from colorama import Fore, Style, init

from vnchat.backends import (
    print_llama_cpp_startup_diagnostics,
    print_llama_server_startup_diagnostics,
)
from vnchat.character import RINDOU_PROFILE
from vnchat.chat_engine import VisualNovelChat
from vnchat.config import get_profile, get_runtime_tuning, parse_cli_args, to_app_config
from vnchat.io_utils import safe_input


def main() -> None:
    init(autoreset=True)

    args = parse_cli_args()
    app_config = to_app_config(args)
    tuning = get_runtime_tuning(args)
    profile = get_profile(args)

    if args.gpu_profile != "none":
        print(f"{Fore.CYAN}[システム] GPUプロファイル: {profile.name}{Style.RESET_ALL}")
        print(f"  {profile.description}")
        print(
            f"  適用値: backend推奨={profile.recommended_backend}, n_ctx={app_config.n_ctx}, n_gpu_layers={app_config.n_gpu_layers}"
        )
        if app_config.backend_mode != profile.recommended_backend:
            print(
                f"{Fore.YELLOW}  注意: このプロファイルの推奨backendは {profile.recommended_backend}（現在は {app_config.backend_mode}）{Style.RESET_ALL}"
            )

    if app_config.backend_mode == "cuda":
        print_llama_cpp_startup_diagnostics(n_gpu_layers=app_config.n_gpu_layers)
    else:
        print_llama_server_startup_diagnostics(server_url=app_config.api_server_url)

    name_raw = safe_input("あなたの名前を入力してね (空=あなた): ")
    if name_raw is None:
        print(
            f"\n{Fore.YELLOW}[システム] 入力が中断されたので終了するね{Style.RESET_ALL}"
        )
        return

    user_name = name_raw.strip() or "あなた"

    if app_config.backend_mode == "cuda" and not os.path.exists(app_config.model_path):
        print(f"{Fore.RED}エラー: モデルファイルが見つかりません{Style.RESET_ALL}")
        print(f"パス: {app_config.model_path}")
        print("\nモデルファイル(.gguf)のパスを --model-path で指定して。")
        print("\n推奨モデル:")
        print("  - Llama 3.1 8B Instruct (Q5_K_M以上)")
        print("  - Llama 3.2 3B Instruct")
        print("  - Gemma 2 9B Instruct")
        return

    chat = VisualNovelChat(
        app_config=app_config,
        tuning=tuning,
        profile=RINDOU_PROFILE,
        user_name=user_name,
    )
    chat.chat_loop()


if __name__ == "__main__":
    main()
