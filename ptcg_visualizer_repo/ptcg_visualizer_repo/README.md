# PTCGABCSVisualizer

**日本語** ・ [English](README.en.md)

ポケモンカードゲーム AI 大会（**PTCG AI Battle Challenge** / cabt エンジン）のリプレイ JSON（`visualize_data()` の出力）を、盤面付きでステップ再生する **単一 HTML + Vanilla JS** ツール。ビルド不要・依存なし・オフライン動作。

> ⚠️ **カード画像は同梱していません**（権利・容量のため）。公式 PDF から各自で生成します（→ [カード画像のセットアップ](#カード画像のセットアップ)）。画像が無くてもカード名・HP・エネルギーはテキストで表示され、ツール自体は動作します。

## 特長

- 神視点の盤面表示（両プレイヤーのバトル場・ベンチ・手札・山札/サイド/トラッシュ枚数・スタジアム）。
- 手番プレイヤーのハイライト、カードにホバーで**大きいプレビュー**（装着エネ・どうぐ付き）。
- トラッシュの枚数クリックで**全カードをグリッド表示**（スクロール可）。
- 右ペインに「その手で選んだアクション」と、そこまでの**デコード済みログ**（カード名・勝敗理由つき）。
- **EN / JP 切替**：UI ラベル・ログ文・**カード画像**を言語連動で切り替え（選択は保存）。

## クイックスタート

1. このリポジトリを clone / ダウンロード。
2. `index.html` をブラウザで開く（ダブルクリックで可）。
3. リプレイ JSON を**ドラッグ＆ドロップ**、または「ファイルを開く」で選択。
   - すぐ試すなら同梱サンプル: `samples/MegaLucario_vs_MegaAbomasnow.json`（短い・スタジアム/エネルギー入り）や `samples/MegaLucario_vs_BeginnerGuide.json`（どうぐ入り）。
4. **◀ 前へ / 次へ ▶**（または ←/→ キー）、スライダーでステップ送り。

この時点ではカード画像は表示されません（テキスト表示）。画像を出すには次へ。

## カード画像のセットアップ

カード画像は公式 PDF（`Card_ID List_EN.pdf` / `Card_ID List_JP.pdf`）の中に埋め込まれている JPEG を取り出して生成します。本リポジトリには PDF も画像も含めていないので、以下の手順で各自セットアップしてください。

### 1. 公式 PDF を入手して配置

PTCG AI Battle Challenge（Kaggle）の公式配布物に含まれる以下を入手します。

- `Card_ID List_EN.pdf`（英語版カード画像）
- `Card_ID List_JP.pdf`（日本語版カード画像）

これらを **`index.html` と同じフォルダ（リポジトリのルート）** に置きます。

```
PTCGABCSVisualizer/
├── index.html
├── _extract_card_images.py
├── Card_ID List_EN.pdf   ← 配置
└── Card_ID List_JP.pdf   ← 配置
```

### 2. Python と PyMuPDF を用意

- Python 3.8 以上（`python --version` で確認）
- 抽出ライブラリ **PyMuPDF** をインストール（`import fitz` で使う `fitz` モジュールを提供します）:

  ```bash
  python -m pip install pymupdf
  ```

  > ⚠️ パッケージ名は **`pymupdf`**（`fitz` という別パッケージは入れないこと）。`pip` ではなく **`python -m pip`** を使うと、スクリプトを実行する Python と同じ環境へ確実に入ります。確認: `python -c "import fitz; print('ok')"`

### 3. 抽出スクリプトを実行

リポジトリのルート（スクリプトのある場所）で実行します。

```bash
python _extract_card_images.py --lang en   # 英語版 → assets/cards/     （1267枚）
python _extract_card_images.py --lang jp   # 日本語版 → assets/cards_jp/ （1267枚）
```

生成後の構成:

```
assets/
├── cards/      # EN: 1.jpg .. 1267.jpg
└── cards_jp/   # JP: 1.jpg .. 1267.jpg
```

オプション:

| オプション                | 説明                                                                  |
| ------------------------- | --------------------------------------------------------------------- |
| `--lang en` / `--lang jp` | どちらの PDF から抽出するか（既定 `en`）                              |
| `--limit N`               | 先頭 N 枚だけ抽出（動作確認用）                                       |
| `--shrink 0`              | 縮小せずフル解像度（660×920）で抽出。既定は 1（1/2 縮小, 約 434×606） |

> 目安：各言語フル抽出で約 1 分、`assets/cards_jp` は約 100MB（`--shrink 0` ならさらに大きくなります）。

### 4. 反映

`index.html` を再読み込みすると、カード画像が表示されます。ヘッダー右の言語ボタンで EN/JP を切り替えると、UI と一緒に**カード画像も EN/JP で切り替わります**。

### うまく動かないとき（Troubleshooting）

- `ModuleNotFoundError: No module named 'fitz'` が出る → PyMuPDF が未インストール、または**実行している Python とは別の環境**に入っています。スクリプトを動かす Python と同じ環境へ入れてください:

  ```bash
  python -m pip install pymupdf
  python -c "import fitz; print('ok')"      # "ok" と表示されれば準備完了
  ```

  Windows で `python` が見つからない場合は Python ランチャー `py -3` を使う（例: `py -3 -m pip install pymupdf` → `py -3 _extract_card_images.py --lang en`）。

- パッケージ名は **`pymupdf`**。`pip install fitz` は**無関係の別物**なので入れないこと。

## 言語切替（EN / JP）

- ヘッダー右の言語ボタンで **UI（ラベル・ログ文）とカード画像を英語/日本語に切替**できます（選択は `localStorage` に保存。既定は日本語）。
- ⚠️ **カード名テキスト**はリプレイ JSON 内のデータ（英語）依存のため、日本語表示でもログ等のカード名は英語のままです（**カード画像**のみ言語で切り替わります）。

## 入力ログについて

- 入力は cabt エンジンの `visualize_data()` が出力する **スナップショットの JSON 配列**（1 選択ごとに 1 要素）。
- 各要素のキー: `select`（選択肢）/ `logs`（ここまでのイベント列）/ `current`（盤面・**神視点**で両者の手札・山札・サイドの中身まで公開）/ `selected`（実際に選んだインデックス）。
- ⚠️ `visualize_data()` では enum が**文字列**で入ります（ログ `type`="Draw"、`select.type`="Main" 等）。一方 `area` / `energies` は**整数**。デコーダはこの前提で実装されています。

## リポジトリ構成

```
PTCGABCSVisualizer/
├── index.html                  # 本体（これを開く）
├── _extract_card_images.py     # カード画像抽出スクリプト（PyMuPDF 必須）
├── README.md / README.en.md    # ドキュメント（日本語 / 英語）
├── .gitignore                  # PDF・生成画像はコミットしない
├── samples/                    # 動作確認用のリプレイ JSON（同梱）
├── (Card_ID List_EN/JP.pdf)    # ← 各自で配置（コミットしない）
└── assets/cards{,_jp}/         # ← 抽出で生成（コミットしない）
```

## 注意・既知の制限

- `file://` で直接開けます（サーバ不要・`fetch` 不使用）。
- 画像が無いカードは**カード名/HP/エネルギーのテキスト表示**にフォールバックします。
- ワザはワザ名ではなく ID 表示（ワザ名辞書は未同梱）。
- ブラウザによっては `localStorage` がファイルパス単位で扱われます。

## クレジット・免責

- Pokémon カードの画像および `Card_ID List_*.pdf` の著作権は株式会社ポケモン / 任天堂 / Creatures / GAME FREAK に帰属します。**本リポジトリはこれらを同梱・再配布しません。** 利用者が公式に入手したファイルからローカルで抽出してください。
- 本ツールは大会運営とは無関係の非公式ツールです。
- **本ソフトウェアは無保証で提供され、利用により生じたいかなる損害・トラブルについても作者は一切の責任を負いません（自己責任でご利用ください）。**
