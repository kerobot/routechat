# ルートチャット(Route Chat)

ビジュアルノベル風（情景→セリフ→状況の変化）で進む、LLMバックエンド切り替え対応のロールプレイチャット。

竜胆（りんどう）という「ダウナー系クールだけど面倒見が良い先輩女性」キャラと会話しつつ、会話の温度感や好意（好感度）に応じた雰囲気の変化や分岐を楽しむなりきりチャットアプリです。

---

## 概要

- 端末上で動く、ビジュアルノベル風のロールプレイチャット
- LLMバックエンド（`llama-cpp-python` 直呼び / `llama-server` API）で応答を生成
- 会話の「状態（好意/温度感/心境）」をターンごとに推定して更新
- 会話履歴が増えたら、選択中のLLMバックエンドで要約してコンテキストを圧縮

---

## 使っている技術

- Python（venv）
- `llama-cpp-python`：CUDAモードでGGUFモデルをローカル推論（CPU/GPUオフロード対応）
- `llama-server`（HTTP API）：APIモードで推論実行（ROCm/HIP構成向け）
- `colorama`：端末出力の色付け
- `dataclasses`：メッセージ/状態の構造化

モデルファイル（`.gguf`）は `models/` に置く想定。

---

## 環境構築（Windows）

### 1) 仮想環境の作成/有効化

```powershell
py -3.13.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) 依存関係のインストール

このプロジェクトは共通で `colorama` を利用し、`llama-cpp-python` は CUDAモード利用時のみ必要。

```powershell
pip install colorama types-colorama

# CUDAモード（--backend cuda）を使う場合のみ
# CUDA Toolkit と Visual C++ を導入しておくこと
$env:CMAKE_ARGS="-DGGML_CUDA=on -DCUDA_ARCHITECTURES=native"
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

> NOTE
> - GPU（CUDA）で動かしたい場合は、環境によってインストール方法が変わる。
> - 本プロジェクトには「起動時にGPUオフロード対応ビルドか」を表示する診断が入っているので、まずはそれで状況確認するのが早い。
> - AMD GPU + ROCm/HIP 構成では、`llama-cpp-python` 直呼びではなく `llama-server` API モード（`--backend api`）の利用を推奨。

### 3) モデルの配置

- `models/` 配下にGGUFを配置
- 実行ファイル内の `MODEL_PATH` を使いたいモデルに合わせる

例：

```python
MODEL_PATH = "models/Llama-3-ELYZA-JP-8B-q4_k_m.gguf"
```

---

## 実行方法

```powershell
python visual_novel_chat.py
```

### バックエンド切り替え（CUDA / ROCm API）

`visual_novel_chat.py` は起動時パラメーターでバックエンドを切り替え可能。

- CUDAモード（`llama-cpp-python` 直呼び）

```powershell
python visual_novel_chat.py --backend cuda --model-path .\models\Llama-3-ELYZA-JP-8B-q4_k_m.gguf --n-gpu-layers -1 --n-ctx 4096
```

- APIモード（ROCm/HIP想定の `llama-server`）

```powershell
python visual_novel_chat.py --backend api --api-server-url http://127.0.0.1:8080 --n-ctx 4096
```

### GPUプロファイル（自動チューニング）

`--gpu-profile` を指定すると、GPUに合わせた `n_ctx` / `n_gpu_layers` / 推論パラメーターをまとめて適用できます。

```powershell
# RTX 4070 Ti SUPER (16GB) / CUDA
python visual_novel_chat.py --backend cuda --gpu-profile rtx4070ti-super-16gb --model-path .\models\umiyuki-Umievo-itr012-Gleipnir-7B-Q6_K.gguf

# RTX 5060 (8GB) / CUDA
python visual_novel_chat.py --backend cuda --gpu-profile rtx5060-8gb --model-path .\models\umiyuki-Umievo-itr012-Gleipnir-7B-Q6_K.gguf

# RX 7900 XT (20GB) / ROCm API
python visual_novel_chat.py --backend api --gpu-profile rx7900xt-20gb --api-server-url http://127.0.0.1:8080
```

指定可能なプロファイル:

- `rtx4070ti-super-16gb`
- `rtx5060-8gb`
- `rx7900xt-20gb`
- `none`（デフォルト、手動設定）

起動時の診断:

- `--backend cuda` : `llama_supports_gpu_offload` とネイティブライブラリ名を表示
- `--backend api` : `/health`・`/props`・`/completion` の疎通を表示（最終的なGPU offload判定は `llama-server` 側ログも確認）

### llama-server API ルート（ROCm/HIP）

CUDA直呼び（`llama-cpp-python`）とは別に、`llama-server` の HTTP API を使うルートを追加しました。

- 手順: `vnchat_api/README.md`
- サーバー起動（PowerShell）: `vnchat_api/run_llama_server.ps1`
- サーバー起動（Python）: `vnchat_api/start_llama_server.py`
- APIチャット: `vnchat_api/chat_via_llama_server.py`

最短確認手順（ROCm）:

1. `llama-server` を起動
2. `http://127.0.0.1:8080/health` が `200` を返すことを確認
3. 本体を `--backend api` で起動

例（本体起動）:

```powershell
python visual_novel_chat.py --backend api --gpu-profile rx7900xt-20gb --api-server-url http://127.0.0.1:8080
```

起動直後にユーザー名の入力があり、以後のプロンプト/会話表示に反映される。

### コマンド

- `state`：現在の状態（好感度/温度感/心境）を表示
- `summary`：これまでの会話要約（要約履歴）を表示
- `save`：会話履歴を `chat_history.json` に保存
- `quit`：終了（保存確認あり）

---

## プログラムの全体像

主要なクラス構成は以下。

- `Message`
  - `role`（system/user/assistant）と `content`、`timestamp` を保持
- `CharacterState`
  - `affection`（好感度：0〜10）
  - `temperature`（会話の温度感：0〜1）
  - `mood`（心境：通常/好意的/クーデレ/心配…など）
  - `scene_count`（ターン数の目安）
- `ConversationManager`
  - 会話履歴の保持
  - コンテキスト生成（system prompt + 要約 + 状態注入）
  - 一定量を超えたら要約を走らせ、履歴を圧縮
- `VisualNovelChat`
  - モデル読み込みとチャットループ
  - 応答生成とプロンプト予算管理
  - 表示/状態更新ロジックへの委譲
- `VNDisplayRenderer`
  - VN表示向けの整形（行分割・セリフ判定・色分け）
- `CharacterStateUpdater`
  - 状態推定（追加LLM呼び出し）
  - 好感度/温度感/心境の更新ロジック

---

## LLMに関わる処理

このプロジェクトは「1ターンでLLMを複数回呼ぶ」構成になっている。

### 1) 応答生成（メイン生成）

- `ConversationManager.get_context_for_llm()` がコンテキストを作る
  - system prompt
  - これまでの要約（あれば）
  - 現在の状態（好感度/温度感/心境）
  - 直近の会話履歴
- `VisualNovelChat._generate_response()` がVN形式（情景→セリフ→変化）で応答生成
- 形式が崩れた場合は1回だけリトライして補正

### 2) 状態推定（追加LLM呼び出し）

- `CharacterStateUpdater._analyze_state_with_llm()` が「評価器」としてLLMをもう一度呼ぶ
- JSONのみを返させ、以下を推定する
  - `mood`
  - `affection_delta`（このターンの好感度変化：-1.0〜+1.0）
  - `temperature`（0.0〜1.0）
  - `confidence`（0.0〜1.0）
- JSON抽出→パース→信頼度フィルタ→クランプを行って状態に反映
- 失敗時は簡易ルールでフォールバック

### 3) 会話要約（LLM要約）

- `ConversationManager._trigger_summary()` が閾値超過時に要約
- `ConversationManager._create_summary_with_llm()` が要約用プロンプトでLLMを呼び、箇条書きの短い要約を生成
- 失敗時は簡易要約にフォールバック

---

## 調整ポイント（どこ触ると効く？）

触る場所はだいたい2つに集約される。

### 1) 起動設定（モデル/GPU/コンテキスト）

- 場所: [vnchat/config.py](vnchat/config.py) の `parse_cli_args()` / `AppConfig`
- `MODEL_PATH`
  - 使うGGUF。ここが変わると速度/品質/口調の乗りが一気に変わる
- `N_GPU_LAYERS`
  - GPUに載せるレイヤー数（`-1` で全部を狙う）
  - 速くしたいなら基本ここが最重要。ただしVRAMが足りないと逆に遅くなる/載らない
- `N_CTX`
  - コンテキスト長。大きいほど長文耐性は上がるけど、メモリと遅延が増えがち

### 2) チャット挙動（応答の雰囲気/安定性/速度）

#### 応答生成（メイン生成）

- 場所: [vnchat/chat_engine.py](vnchat/chat_engine.py) の `_generate_response()`
- `max_tokens`
  - 1ターンの最大出力。長くするほどリッチになるけど遅くなる
- `temperature` / `top_p` / `top_k`
  - 自由度と揺れ。上げると自由になるけど、形式崩れや言い回しのブレも増える
- `repeat_penalty`
  - 繰り返し抑制。上げすぎると不自然になることもある

#### 空応答/コンテキスト超過の対策

- 場所: [vnchat/chat_engine.py](vnchat/chat_engine.py) の `_build_prompt_with_generation_budget()` と、その呼び出し側
- `reserve_tokens`（いまは `512`）
  - 生成のために“絶対に空けておく”トークン余白
  - 空応答が出るなら増やす（ただし入れられる履歴が減る）
  - もっと過去ログを入れたいなら減らす（ただし空応答/不安定のリスク増）

#### 状態推定（追加推論）

- 場所: [vnchat/config.py](vnchat/config.py) の `RuntimeTuning` と [vnchat/state_logic.py](vnchat/state_logic.py) の `_analyze_state_with_llm()`
- `STATE_EVAL_EVERY_N_TURNS`（いまは `2`）
  - 小さくするほど“反応が細かくなる”けど遅くなる（`1` で毎ターン）
  - 大きくするほど速くなるけど、心境の追従は鈍る
- `_analyze_state_with_llm()` の `max_tokens`（いまは `120`）
  - 状態推定の出力長。基本は小さくてOK（上げると遅くなる）
- `confidence` の扱い
  - 低信頼（`0.20` 未満）は不採用
  - 強いムード（`苛立ち`/`警戒`）や好意系（`好意的`/`クーデレ`）は、さらに信頼度でガードしてブレを抑えてる

#### 好感度・ムードの遷移

- 場所: [vnchat/config.py](vnchat/config.py) の `RuntimeTuning`
  - しきい値を下げると入りやすくなる（ルート感↑）
  - 上げると渋くなる（クール維持↑）
  - さらにヒステリシスで行ったり来たりを抑えてる（揺れが気になる時はここも見直しポイント）

#### 要約の頻度と「要約の濃さ」

- 場所: [vnchat/conversation.py](vnchat/conversation.py) の `ConversationManager`
- `summary_threshold`（初期値 `15`）
  - 小さくすると要約が早く走ってコンテキストが軽くなりやすい（速度↑）
  - 大きくすると生ログが多く残ってディテールが保たれやすい（ただし重くなりがち）
- `max_history`（初期値 `20`）
  - 要約後に残す直近ログ量の目安（実装上は `max_history // 2` 件を保持）
- コンテキストに入れる要約数（いまは直近 `4`）
  - 増やすと“過去の伏線”は拾えるけど、遅延とブレも増えやすい

### プリセット例（軽い/標準/重い + VRAM別）

前提：ここに書くのは「まず動かすための目安」。
モデル（7B/8B/14B）や量子化（Q4/Q5/Q6）で必要VRAMも速度も変わるから、詰まったら `N_GPU_LAYERS` を下げるのが最優先。

共通の考え方：

- 速度を上げたい：`N_GPU_LAYERS` を上げる（載る範囲で）/ `STATE_EVAL_EVERY_N_TURNS` を大きくする / `max_tokens` を減らす
- 品質を上げたい：`N_CTX` と `max_tokens` を上げる（ただし遅くなる）/ 状態評価を細かくする
- 空応答が出る/不安定：`reserve_tokens` を増やす（代わりに入れられる履歴は減る）

#### VRAM 8GB（速度優先で割り切り）

- 軽い（とにかく速く）
  - `N_GPU_LAYERS`: 可能なら `-1`、無理なら 20〜35 あたりから
  - `N_CTX`: 2048〜4096
  - メイン生成 `max_tokens`: 400〜600
  - `STATE_EVAL_EVERY_N_TURNS`: 3〜4
  - `summary_threshold`: 12
- 標準（バランス）
  - `N_GPU_LAYERS`: 可能なら `-1`、無理なら 30〜45 あたりから
  - `N_CTX`: 4096
  - メイン生成 `max_tokens`: 600〜800
  - `STATE_EVAL_EVERY_N_TURNS`: 2
  - `summary_threshold`: 15
- 重い（品質寄せ・ただし厳しい）
  - `N_GPU_LAYERS`: 40〜60（載らなければ即ダウン）
  - `N_CTX`: 4096
  - メイン生成 `max_tokens`: 800
  - `STATE_EVAL_EVERY_N_TURNS`: 2（遅いなら 3）
  - `summary_threshold`: 18

#### VRAM 16GB（いちばん楽しいゾーン）

- 軽い
  - `N_GPU_LAYERS`: まず `-1`（載らない/遅いなら下げる）
  - `N_CTX`: 4096
  - メイン生成 `max_tokens`: 600
  - `STATE_EVAL_EVERY_N_TURNS`: 3
  - `summary_threshold`: 15
- 標準
  - `N_GPU_LAYERS`: `-1`
  - `N_CTX`: 4096〜8192
  - メイン生成 `max_tokens`: 800
  - `STATE_EVAL_EVERY_N_TURNS`: 2
  - `summary_threshold`: 15
- 重い
  - `N_GPU_LAYERS`: `-1`
  - `N_CTX`: 8192（遅いなら 4096 へ）
  - メイン生成 `max_tokens`: 900〜1200（長すぎるなら 800）
  - `STATE_EVAL_EVERY_N_TURNS`: 1〜2
  - `summary_threshold`: 18

#### VRAM 20GB（品質・長文寄せも現実的）

- 軽い
  - `N_GPU_LAYERS`: `-1`
  - `N_CTX`: 4096
  - メイン生成 `max_tokens`: 800
  - `STATE_EVAL_EVERY_N_TURNS`: 2
  - `summary_threshold`: 15
- 標準
  - `N_GPU_LAYERS`: `-1`
  - `N_CTX`: 8192
  - メイン生成 `max_tokens`: 900〜1200
  - `STATE_EVAL_EVERY_N_TURNS`: 2
  - `summary_threshold`: 18
- 重い
  - `N_GPU_LAYERS`: `-1`
  - `N_CTX`: 8192〜12288（遅延と相談）
  - メイン生成 `max_tokens`: 1200
  - `STATE_EVAL_EVERY_N_TURNS`: 1
  - `summary_threshold`: 18〜22

※`N_CTX` を上げるほど、モデルや環境によってはメモリと遅延が増える。まずは 4096 で安定させてから増やすのが安全。

---

## 体感速度（遅延対策）

このプロジェクトは「応答生成 + 状態推定 + 要約」でLLM呼び出し回数が増えがちなので、
“自然さ”を落としすぎない範囲で、追加推論やプロンプト肥大化を抑えて体感速度を上げている。

- 状態推定（追加推論）を毎ターンではなく間引く（例：2ターンに1回）
  - ただし会話が動いたっぽいキーワードがあるときは強制評価
  - スキップしたターンは軽量ルールでムード/温度感を維持して破綻を防ぐ
- 要約の発火タイミングを整理（UXと遅延の最適化）
  - 会話の流れを壊しにくいタイミングに寄せる（アシスタント応答後など）
- コンテキスト超過対策を「履歴破壊（削除）」ではなく「非破壊トリム」に変更
  - 会話履歴そのものは保持したまま、LLMに渡すプロンプトだけを短くして生成余地を確保
  - 生成余地の確保は二分探索で“必要最小限だけ”間引く方向
- 状態推定/要約に渡す入力を短く保つ（直近中心）
  - 推論コストと暴走（長文でのフォーマット崩れ）を抑える

---

## キャラ一貫性（ブレ抑制）

状態推定をLLMに任せると、雰囲気が良い時に急にデレたり、逆に急落したりしてキャラがブレやすい。
そこで「状態更新にガードを入れる」ことで、竜胆のクールさを保ったまま変化だけ拾うようにしている。

- `confidence` が低い推定は採用しない/影響を小さくする
- `mood` の強制上書きは“ソフトなムード”に限定（ネガが強いときは無理に好意系へ寄せない）
- 好感度しきい値の遷移にヒステリシス（行ったり来たりの揺れを減らす）
- 「クーデレ」の定義を弱める
  - ベタベタを禁止しつつ、“甘さが隠しきれない”が行動に滲む程度に留める

---

## ルート感（刺さる瞬間増）

単発の会話にならず「回収→少し進展→次の一手」に繋がるよう、
system prompt と要約に“継続性ルール”を入れている。

- system prompt に「直近の具体要素を最低1つ回収」を明記
- 「1ターンで雑に終わらせない」「次に繋がる一手（問いかけ/提案/宿題）を混ぜる」を明記
- 要約にも「未解決の宿題/次の一手」を残す方針を入れる
- `scene_count` を「1往復=1シーン」になるよう補正（アシスタント応答時に進める）
- 要約はコンテキストに積みすぎない
  - 直近中心で注入し、昔の要素でキャラや話題がブレるのを抑える

---

## 端末表示（色分け）

元々は「情景→セリフ→状況の変化」の3部構成を前提に、塊ごとに色分けして表示していた。

ただ、ルート感（回収/次の一手）を強めるほど、
描写とセリフが交互に出たり、セリフが引用符だけ（`「…」`）になったりして、
3分割前提の表示が崩れやすくなった。

そのため表示側は「見出しで3分割」ではなく、行単位で「会話っぽい行」を判定して色分けする方式にしている。

- 会話として扱う行の例
  - `竜胆：...` / `竜胆: ...`
  - `ユーザー名：...` / `ユーザー名: ...`（起動時入力の名前）
  - `「...」` / `『...』`（引用符から始まる行）
- 見出し行（`【竜胆の動作・情景描写】` / `【状況の変化】`）は表示では省略（色分けの邪魔をしない）
- `「...」` のように話者ラベルが省略されたセリフ行は、表示上の視認性のため `竜胆：` を自動で補完して出す

※この補完は“表示のみ”で、会話履歴やLLMへの入力テキスト自体を改変する意図はない。

---

## 好意/熱意/雰囲気（状態）管理

### パラメータ

- 好感度（`affection`：0〜10）
  - ターンごとの `affection_delta` を足し込み
  - 上がる頻度を増やすため、ムードに応じた「じわ上げ」も適用（過剰な跳ねを避けつつ回数を増やす）
- 温度感（`temperature`：0〜1）
  - 推定値をそのまま入れず、平滑化して急変を抑える
- 心境（`mood`）
  - `通常` / `満更でもない` / `好意的` / `クーデレ` / `心配` / `困惑` / `呆れ` / `苛立ち` / `照れ` / `警戒` / `安心` / `興味`

### クーデレ遷移

- 好感度が一定以上で `mood` を `クーデレ` に寄せる
- ただし `苛立ち` / `警戒` のような強いネガは優先し、強制上書きしない

しきい値は `VisualNovelChat.AFFECTION_FAVORABLE_THRESHOLD` / `VisualNovelChat.AFFECTION_KUUDERE_THRESHOLD` で調整できる。

（現状の実装では、しきい値は [vnchat/config.py](vnchat/config.py) の `RuntimeTuning`（`affection_favorable_threshold` / `affection_kuudere_threshold`）で調整する）

---

## Tips（好感度が上がりやすい会話）

竜胆は「クールで口下手だけど面倒見が良い」前提なので、ベタベタした褒めや恋愛ノリよりも、仕事の文脈での信頼・成長・気遣いが好感度に繋がりやすい。

### 上がりやすい

- 感謝＋具体（何が助かったか、次どうするか）
- 成長報告（失敗→対策→改善、再発防止まで）
- 相談の仕方がうまい（丸投げせず、ここまで調べた/次どこを見る？）
- 重くない気遣い（相手の負荷や状況への配慮）
- 一貫性（言ったことを次ターンで実行してる）

### 下がりやすい/上がりにくい

- ベタ褒め連打、過剰なお世辞（軽く見える）
- いきなり距離を詰める恋愛ノリ（設定的にクール維持のため反発しやすい）
- 反省だけで行動が変わらない（呆れ寄りになりやすい）
- 丸投げ/依存（調査ゼロで「全部お願い」など）

### そのまま使える例文

- 「さっきの切り分け助かった。次から最初にログ見るようにする、ありがと」
- 「ここまでは確認できた。残りがネットワークかアプリか迷ってるんだけど、どっちから攻める？」
- 「いまの作業、あと5分だけ相談いい？ 無理なら後でにする」
- 「昨日の件、教えてもらった手順で再発しなかった。ちゃんと手順書に残した」

---

## GPU（CUDA直呼び）利用状況の確認

`--backend cuda` の起動時に `llama-cpp-python 診断` が表示される。

- `llama_supports_gpu_offload: True/False`
  - Trueでも「実際にGPUへ十分載るか」はVRAM/量子化/`n_gpu_layers` に依存
- 読み込まれたネイティブライブラリ（例：`...\site-packages\llama_cpp\lib\llama.dll`）

モデルを切り替えて急に遅くなった場合は、量子化（例：Q6は重い）やGPUオフロード量が変わってCPU計算が増えている可能性が高い。

---

## GPU（ROCm API）利用状況の確認

`--backend api` 起動時に `llama-server(API) 診断` が表示される。

- `health: OK`
  - API疎通が取れている状態
- `props(gpu関連)`
  - サーバー実装が対応していれば、GPU/ROCm関連キーが表示される
- `completion: OK`
  - 実際の推論呼び出しまで通っている状態

補足:

- 本体側の診断は「API疎通と推論到達」の確認
- GPU offload の最終判定は `llama-server` 側ログ（`ggml_backend_*` など）も併せて確認するのが確実

よくあるハマりどころ:

- `health: NG`
  - `llama-server` 未起動、または `--api-server-url` の指定ミス
- `completion: NG`
  - モデルパス/起動引数の不整合、サーバー側エラー
- 速度が想定より遅い
  - サーバー側でGPU offloadされずCPU推論に落ちている可能性

---

## リポジトリ構成

- `visual_novel_chat.py`：起動用エントリポイント（薄いラッパー）
- `vnchat/cli.py`：CLI処理と起動フロー
- `vnchat/chat_engine.py`：チャット本体（オーケストレーション）
- `vnchat/presentation.py`：VN表示整形・色分け
- `vnchat/state_logic.py`：状態推定/状態遷移ロジック
- `vnchat/character.py`：キャラクター定義（プロンプト/初期シーン）
- `vnchat/config.py`：調整可能パラメーターと起動引数
- `vnchat/backends.py`：CUDA/APIバックエンドと診断
- `vnchat/http_client.py`：HTTP JSON通信の共通処理
- `vnchat/conversation.py`：履歴・要約・コンテキスト管理
- `vnchat_api/`：`llama-server` 起動/疎通確認用スクリプト群
- `chat_history.json`：会話保存ファイル（生成物。git管理対象外）
- `models/`：GGUFモデル置き場（重量物はgit管理対象外）

---

## メモ

- 本プロジェクトはローカル推論（CUDA直呼び）/API推論（ROCm想定）の両対応で、モデルサイズ・量子化・VRAMによって速度/品質が大きく変わる。
- 「応答生成 + 状態推定 + 要約」でLLM呼び出し回数が増える設計なので、モデルを重くすると体感遅延も増えやすい。
