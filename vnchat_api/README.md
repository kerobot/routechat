# llama-server API モード（ROCm/HIP）

既存の `visual_novel_chat.py` で使うバックエンドのうち、APIモード（`--backend api`）として `llama-server` を起動し、HTTP API 経由でチャットするための手順。

## 目的

- CUDA直呼び（`llama-cpp-python`）とは別に、`llama-server` API バックエンドで推論する
- ROCm/HIP ビルド済み `llama-server` を使って、Windows + AMD GPU 環境で GPU オフロードを確認しやすくする
- `vnchat` 本体とAPIスクリプトで HTTP JSON 処理を共通化し、保守コストを下げる

## 前提

- `llama-server.exe`（ROCm/HIP 対応ビルド済み）を用意済み
- GGUF モデルを `models/` に配置済み
- Python 仮想環境（`venv`）が有効化済み
- `vnchat_api/chat_via_llama_server.py` は `vnchat/http_client.py` を利用するため、リポジトリルートをカレントにして実行する

> メモ
> - CUDA環境で `--backend cuda` を使う場合は、このディレクトリの手順は不要
> - AMD GPU + ROCm/HIP 構成では、`--backend api` で本手順を使う運用を推奨

## 手順

### 1. PowerShell でサーバー起動

```powershell
./vnchat_api/run_llama_server.ps1 \
  -ServerBinary ".\bin\llama-server.exe" \
  -ModelPath ".\models\Llama-3-ELYZA-JP-8B-q4_k_m.gguf" \
  -ContextSize 8192 \
  -GpuLayers -1 \
  -Port 8080
```

> 補足
> - `-GpuLayers -1` は「可能な限り GPU に載せる」設定
> - うまく載らない場合は `-GpuLayers 20` などへ下げて試す

### 2. API 経由チャットを開始

別ターミナルで:

```powershell
python .\vnchat_api\chat_via_llama_server.py \
  --server-url http://127.0.0.1:8080
```

`quit` で終了。

## 追加の起動方法（Pythonからサーバー起動）

PowerShell ではなく Python から `llama-server` を起動したい場合:

```powershell
python .\vnchat_api\start_llama_server.py \
  --server-binary .\bin\llama-server.exe \
  --model-path .\models\Llama-3-ELYZA-JP-8B-q4_k_m.gguf \
  --ctx-size 8192 \
  --n-gpu-layers -1 \
  --port 8080
```

Ctrl+C で停止。

## 疎通確認エンドポイント

- ヘルスチェック: `GET /health`
- OpenAI 互換: `POST /v1/chat/completions`
- 互換が無い構成向けフォールバック: `POST /completion`

`chat_via_llama_server.py` はまず `/v1/chat/completions` を試し、失敗時は `/completion` を試す。

## 実装メモ（2026-02 リファクタ反映）

- HTTPのJSONリクエスト処理は `vnchat/http_client.py` に集約
- `vnchat/backends.py`（本体側）と `vnchat_api/chat_via_llama_server.py`（補助CLI側）で同じ実装を利用
