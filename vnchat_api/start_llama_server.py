from __future__ import annotations

import argparse
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def wait_for_health(server_url: str, timeout_sec: float = 60.0) -> bool:
    health_url = f"{server_url.rstrip('/')}/health"
    deadline = time.time() + timeout_sec

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2.0) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.7)
            continue
    return False


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        str(args.server_binary),
        "-m",
        str(args.model_path),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "-c",
        str(args.ctx_size),
        "--n-gpu-layers",
        str(args.n_gpu_layers),
    ]

    if args.threads > 0:
        cmd.extend(["-t", str(args.threads)])

    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ROCm/HIP 版 llama-server を起動して API 待受する"
    )
    parser.add_argument("--server-binary", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--health-timeout", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.server_binary.exists():
        print(f"[ERROR] llama-server が見つからない: {args.server_binary}")
        raise SystemExit(1)

    if not args.model_path.exists():
        print(f"[ERROR] モデルが見つからない: {args.model_path}")
        raise SystemExit(1)

    command = build_command(args)
    server_url = f"http://{args.host}:{args.port}"

    print("[INFO] llama-server 起動コマンド:")
    print("       " + " ".join(command))

    process = subprocess.Popen(command)

    try:
        if wait_for_health(server_url, timeout_sec=args.health_timeout):
            print(f"[INFO] サーバー起動完了: {server_url}")
            print("[INFO] 停止するには Ctrl+C")
        else:
            print("[WARN] ヘルスチェックがタイムアウトした。ログを確認して。")

        process.wait()
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C を受信。llama-server を停止する。")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    finally:
        if process.poll() is None:
            process.terminate()


if __name__ == "__main__":
    main()
