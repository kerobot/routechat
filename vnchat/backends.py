from __future__ import annotations

"""バックエンド実装の共通アダプタ。

このモジュールは、ローカル `llama-cpp-python`（CUDA）と
HTTP の `llama-server`（API）の2系統を同じ呼び出し形にそろえる。
起動時診断、HTTP補助、トークナイズ、推論呼び出しをまとめて提供する。
"""

import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any, Literal, cast

from colorama import Fore, Style

from vnchat.http_client import get_json_safe, request_json


LlamaCpp: Any = None
try:
    from llama_cpp import Llama as _LlamaCpp  # type: ignore

    LlamaCpp = _LlamaCpp
except Exception:
    pass


def _http_get_json(url: str, timeout_sec: float = 5.0) -> dict[str, Any] | None:
    """GETでJSONを取得する補助関数。失敗時は `None` を返す。"""
    return get_json_safe(url=url, timeout_sec=timeout_sec)


def _http_post_json(
    url: str, payload: dict[str, Any], timeout_sec: float = 15.0
) -> dict[str, Any] | None:
    """POSTでJSONを送受信する補助関数。失敗時は HTTPエラーを表示して `None` を返す。"""
    try:
        return request_json(
            url=url,
            timeout_sec=timeout_sec,
            method="POST",
            payload=payload,
        )
    except urllib.error.HTTPError as e:
        print(f"{Fore.YELLOW}  HTTPError: {e.code} {e.reason} ({url}){Style.RESET_ALL}")
        return None
    except Exception:
        return None


def print_llama_cpp_startup_diagnostics(n_gpu_layers: int) -> None:
    """`llama-cpp-python` 利用時の環境情報とGPUオフロード対応状況を表示する。"""
    print(f"{Fore.YELLOW}[システム] llama-cpp-python 診断:{Style.RESET_ALL}")
    print(f"  python: {sys.version.split()[0]} ({sys.platform})")
    print(f"  n_gpu_layers(要求): {n_gpu_layers}")

    try:
        import llama_cpp  # type: ignore
        import llama_cpp.llama_cpp as lc  # type: ignore

        version = getattr(llama_cpp, "__version__", "(unknown)")
        supports = None
        try:
            supports = bool(lc.llama_supports_gpu_offload())
        except Exception:
            supports = None

        lib_name = None
        try:
            lib_name = getattr(getattr(lc, "_lib", None), "_name", None)
        except Exception:
            lib_name = None

        print(f"  llama_cpp: {version}")
        print(f"  llama_supports_gpu_offload: {supports}")
        if lib_name:
            print(f"  native lib: {lib_name}")

        if supports is False:
            print(
                f"{Fore.YELLOW}  -> GPUオフロード非対応ビルド（CPUのみの可能性が高い）{Style.RESET_ALL}"
            )
        elif supports is True:
            print(
                f"{Fore.GREEN}  -> GPUオフロード対応ビルド（実際にGPUへ載るかはVRAMと設定次第）{Style.RESET_ALL}"
            )
    except Exception as e:
        print(f"{Fore.YELLOW}  診断に失敗: {type(e).__name__}: {e}{Style.RESET_ALL}")


def print_llama_server_startup_diagnostics(server_url: str) -> None:
    """`llama-server` API への疎通と最小推論の可否を診断して表示する。"""
    base = server_url.rstrip("/")
    print(f"{Fore.YELLOW}[システム] llama-server(API) 診断:{Style.RESET_ALL}")
    print(f"  python: {sys.version.split()[0]} ({sys.platform})")
    print(f"  server_url: {base}")

    health_ok = False
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=5.0) as response:
            health_ok = response.status == 200
    except Exception as e:
        print(f"  health: NG ({type(e).__name__}: {e})")

    if not health_ok:
        print(
            f"{Fore.YELLOW}  -> API疎通できません。llama-server起動状態/URLを確認してください{Style.RESET_ALL}"
        )
        return

    print("  health: OK")

    props = _http_get_json(f"{base}/props", timeout_sec=5.0)
    if props:
        gpu_related = {
            k: v
            for k, v in props.items()
            if any(x in k.lower() for x in ("gpu", "hip", "rocm", "offload"))
        }
        if gpu_related:
            print(f"  props(gpu関連): {gpu_related}")
        else:
            print("  props: 取得OK（gpu関連キーは見つからず）")
    else:
        print("  props: 未取得（サーバー実装によっては未対応）")

    tiny = _http_post_json(
        f"{base}/completion",
        {
            "prompt": "テスト: こんにちは",
            "n_predict": 8,
            "temperature": 0.1,
        },
        timeout_sec=20.0,
    )
    if tiny and isinstance(tiny.get("content"), str):
        print("  completion: OK")
        print(
            f"{Fore.GREEN}  -> API推論まで通過。GPU offloadの最終判定はサーバーログの ggml_backend_* 出力も確認してください{Style.RESET_ALL}"
        )
    else:
        print("  completion: NG（推論レスポンス取得失敗）")


class LLMBackendAdapter:
    """CUDA/API どちらの推論バックエンドにも同一インターフェースを提供する。"""

    def __init__(
        self,
        backend_mode: Literal["cuda", "api"],
        n_ctx: int,
        model_path: str,
        n_gpu_layers: int,
        server_url: str,
        api_timeout_sec: float,
    ):
        """バックエンド設定を保持し、CUDAモード時はモデルを初期化する。"""
        self.backend_mode = backend_mode
        self._n_ctx = n_ctx
        self.server_url = server_url.rstrip("/")
        self.api_timeout_sec = api_timeout_sec
        self._llm: Any = None

        if backend_mode == "cuda":
            if LlamaCpp is None:
                raise RuntimeError("llama-cpp-python が import できない")
            self._llm = LlamaCpp(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=False,
                n_threads=8,
            )

    def n_ctx(self) -> int:
        """利用中バックエンドのコンテキスト長を返す（取得不可時は設定値）。"""
        if self.backend_mode == "cuda" and self._llm is not None:
            try:
                return int(self._llm.n_ctx())
            except Exception:
                pass
        return self._n_ctx

    def tokenize(self, b: bytes) -> list[int]:
        """入力バイト列をトークン化する。APIモードでは概算トークン数を返す。"""
        if self.backend_mode == "cuda" and self._llm is not None:
            try:
                return cast(list[int], self._llm.tokenize(b))
            except Exception:
                pass

        text = b.decode("utf-8", errors="ignore")
        approx = max(1, len(text) // 3)
        return list(range(approx))

    def __call__(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        stop: list[str] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """推論を実行し、`{"choices": [{"text": ...}]}` 形式で結果を返す。"""
        if self.backend_mode == "cuda":
            if self._llm is None:
                raise RuntimeError(
                    "CUDAモードだが llama-cpp-python が初期化されていない"
                )
            result = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=stop or ["<|eot_id|>"],
                stream=stream,
            )
            if isinstance(result, dict):
                return cast(dict[str, Any], result)

            last_chunk: dict[str, Any] | None = None
            for chunk in cast(Iterator[dict[str, Any]], result):
                last_chunk = chunk
            return last_chunk or {"choices": [{"text": ""}]}

        payload: dict[str, Any] = {
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "n_predict": max_tokens,
            "stop": stop or ["<|eot_id|>"],
            "stream": False,
        }
        result = _http_post_json(
            f"{self.server_url}/completion",
            payload,
            timeout_sec=self.api_timeout_sec,
        )
        text = ""
        if isinstance(result, dict):
            value = result.get("content", "")
            if isinstance(value, str):
                text = value

        return {"choices": [{"text": text, "finish_reason": "stop"}]}
