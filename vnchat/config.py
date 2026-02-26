from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from typing import Literal


BackendMode = Literal["cuda", "api"]
GpuProfileName = Literal[
    "none",
    "rtx4070ti-super-16gb",
    "rtx5060-8gb",
    "rx7900xt-20gb",
]


@dataclass(frozen=True)
class RuntimeTuning:
    max_history: int = 20
    summary_threshold: int = 15
    max_summaries_in_context: int = 4

    reserve_tokens: int = 512
    generation_max_tokens: int = 800
    generation_temperature: float = 0.75
    generation_top_p: float = 0.9
    generation_top_k: int = 40
    generation_repeat_penalty: float = 1.1

    analyzer_max_tokens: int = 120
    analyzer_temperature: float = 0.2
    analyzer_top_p: float = 0.9
    analyzer_top_k: int = 40
    analyzer_repeat_penalty: float = 1.05
    state_eval_every_n_turns: int = 2

    affection_favorable_threshold: float = 5.0
    affection_kuudere_threshold: float = 7.5

    force_eval_keywords: tuple[str, ...] = (
        "ありがとう",
        "助かった",
        "ごめん",
        "すまん",
        "無理",
        "辛い",
        "ダメ",
        "やばい",
    )


@dataclass(frozen=True)
class AppConfig:
    backend_mode: BackendMode
    model_path: str
    n_gpu_layers: int
    n_ctx: int
    api_server_url: str
    api_timeout_sec: float
    save_file: str = "chat_history.json"


@dataclass(frozen=True)
class CliArgs:
    backend: BackendMode
    gpu_profile: GpuProfileName
    model_path: str
    n_gpu_layers: int
    n_ctx: int
    api_server_url: str
    api_timeout_sec: float


@dataclass(frozen=True)
class GpuProfile:
    name: GpuProfileName
    recommended_backend: BackendMode
    n_ctx: int
    n_gpu_layers: int
    tuning: RuntimeTuning
    description: str


GPU_PROFILES: dict[GpuProfileName, GpuProfile] = {
    "none": GpuProfile(
        name="none",
        recommended_backend="cuda",
        n_ctx=4096,
        n_gpu_layers=-1,
        tuning=RuntimeTuning(),
        description="プロファイル未適用（手動設定）",
    ),
    "rtx4070ti-super-16gb": GpuProfile(
        name="rtx4070ti-super-16gb",
        recommended_backend="cuda",
        n_ctx=8192,
        n_gpu_layers=-1,
        tuning=RuntimeTuning(
            generation_max_tokens=900,
            state_eval_every_n_turns=2,
            summary_threshold=16,
            reserve_tokens=640,
        ),
        description="CUDA向け。16GB VRAMで品質と速度のバランス重視",
    ),
    "rtx5060-8gb": GpuProfile(
        name="rtx5060-8gb",
        recommended_backend="cuda",
        n_ctx=4096,
        n_gpu_layers=32,
        tuning=RuntimeTuning(
            generation_max_tokens=600,
            state_eval_every_n_turns=3,
            summary_threshold=12,
            reserve_tokens=700,
        ),
        description="CUDA向け。8GB VRAMで安定重視",
    ),
    "rx7900xt-20gb": GpuProfile(
        name="rx7900xt-20gb",
        recommended_backend="api",
        n_ctx=8192,
        n_gpu_layers=-1,
        tuning=RuntimeTuning(
            generation_max_tokens=1100,
            state_eval_every_n_turns=2,
            summary_threshold=18,
            reserve_tokens=640,
        ),
        description="ROCm(API)向け。20GB VRAMで長文・品質寄り",
    ),
}


def parse_cli_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="ビジュアルノベル風チャット（cuda local / rocm api 切替対応）"
    )
    parser.add_argument(
        "--backend",
        choices=["cuda", "api"],
        default="cuda",
        help="cuda=llama-cpp-python(local), api=llama-server(ROCm想定)",
    )
    parser.add_argument(
        "--gpu-profile",
        choices=[
            "none",
            "rtx4070ti-super-16gb",
            "rtx5060-8gb",
            "rx7900xt-20gb",
        ],
        default="none",
        help="GPU別プロファイルを適用（n_ctx / n_gpu_layers / 推論チューニング）",
    )
    parser.add_argument(
        "--model-path",
        default="models/Llama-3-ELYZA-JP-8B-q4_k_m.gguf",
        help="cudaモードで使うGGUFモデルパス",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="cudaモード時のGPUレイヤー数（-1で可能な限り載せる）",
    )
    parser.add_argument("--n-ctx", type=int, default=4096, help="コンテキストサイズ")
    parser.add_argument(
        "--api-server-url",
        default="http://127.0.0.1:8080",
        help="apiモード時の llama-server URL",
    )
    parser.add_argument(
        "--api-timeout-sec",
        type=float,
        default=120.0,
        help="apiモード時の推論タイムアウト秒",
    )
    ns = parser.parse_args()

    return CliArgs(
        backend=ns.backend,
        gpu_profile=ns.gpu_profile,
        model_path=ns.model_path,
        n_gpu_layers=ns.n_gpu_layers,
        n_ctx=ns.n_ctx,
        api_server_url=ns.api_server_url,
        api_timeout_sec=ns.api_timeout_sec,
    )


def to_app_config(args: CliArgs) -> AppConfig:
    profile = GPU_PROFILES[args.gpu_profile]
    n_ctx = profile.n_ctx if args.gpu_profile != "none" else args.n_ctx
    n_gpu_layers = (
        profile.n_gpu_layers if args.gpu_profile != "none" else args.n_gpu_layers
    )

    return AppConfig(
        backend_mode=args.backend,
        model_path=args.model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        api_server_url=args.api_server_url,
        api_timeout_sec=args.api_timeout_sec,
    )


def get_runtime_tuning(args: CliArgs) -> RuntimeTuning:
    profile = GPU_PROFILES[args.gpu_profile]
    if args.gpu_profile == "none":
        return RuntimeTuning()
    return replace(profile.tuning)


def get_profile(args: CliArgs) -> GpuProfile:
    return GPU_PROFILES[args.gpu_profile]
