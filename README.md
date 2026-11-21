# Griffith-Kizu

微分可能論理ゲート型 NN フレームワーク **[Griffith](https://github.com/tsukushi-gate/Griffith)**（`@Tebasaki_lab` 氏）を **[Kizu](https://github.com/kizu-lang/kizu)** 言語へ移植・応用した非公式の実験プロジェクトです。

**「学習 ＝ 回路コンパイラ」** をテーマに、純 Kizu による学習（CPU / Apple Silicon Metal GPU）→ 離散化 → 4 パス論理合成 → 重みゼロ・乗算ゼロのビット並列 AOT コード生成・ブラウザ WASM 推論までを一気通貫で実装しています。

**Live Demo**: https://lab.soyukke.com/griffith/

---

## 収録ブラウザデモ（[`web/`](web/) — 単体静的サイトとして即起動可能）

すべて単一の統合 WASM（[`web/dist/app.wasm`](web/dist/app.wasm), 636 KB）と事前学習済み `.glgpt` モデルで動作します。サーバー推論・API 通信・WebGPU は不要です。

| ラボ | パス | 内容 |
|---|---|---|
| **一覧ハブ** | [`web/index.html`](web/index.html) | 全 5 デモのランチャー＋謝辞 |
| **LAB-01 `draw/`** | [`web/draw/index.html`](web/draw/index.html) | 5×5 手描きドット絵からブラウザ内で逆伝播学習 → 120 ゲート結晶化 → 4 パス論理合成 → Kizu AOT コード & Griffith IR JSON 生成 |
| **LAB-01b `compiler/`** | [`web/compiler/index.html`](web/compiler/index.html) | 4-bit パリティ（**`65 → 3` ゲート・理論最小**）・XOR・2-bit 加算器の回路探索＆64 並列 Bit-Slice VM vs 重みゼロ AOT 速度対決（`0.09〜0.18 ns/inf`） |
| **LAB-02 `ladder/`** | [`web/ladder/index.html`](web/ladder/index.html) | Tiny Shakespeare 2,000 step 学習済み **5 段 `LogicLadderGPT`**（`L0` 311 KB → `L3-C` **59 KB・ブロック乗算 0 回**）切替＆1-hot ハードウェア MUX Attention マップ可視化 |
| **LAB-03 `distill/`** | [`web/distill/index.html`](web/distill/index.html) | 辞書 0 B の UTF-8 N-gram 64-bit 指紋 → 実数 Teacher からラベルなし日本語ログ経由で KL 蒸留 → 論理合成（乗算 0 回・30× 高速） |
| **LAB-04 `jpllm/`** | [`web/jpllm/index.html`](web/jpllm/index.html) | マルチターン対応チャット UI＋純 Kizu BPE 1280・4 層 N=4 WARP LUT・**M4 Max Metal GPU 段階 KL 蒸留**（`L0` PPL 11.56 → `L0b` 4,608 LUT **PPL 11.40** → `L3-C` 6,912 LUT・0 乗算・PPL 19.22） |

---

## クイックスタート

### 1. ブラウザデモをローカルで試す（ビルド不要）

```sh
python3 -m http.server 8080 --directory web
# http://127.0.0.1:8080/ をブラウザで開く
```

### 2. Node.js で全 5 デモの統合 WASM 検証を回す

```sh
node scripts/verify_wasm.mjs
```

### 3. Kizu コンパイラでテスト・ビルド・学習を実行する

`kizu` が `PATH` にあるか、`KIZU_SRC=/path/to/kizu` を設定して実行します。

```sh
# ユニットテスト全実行（勾配チェック・Stage 1 学習＆合成・Stage 2・BPE・Metal GPU vs CPU 一致）
scripts/kizu test .

# 4 ターゲット一括ビルド（native / metal / spirv / wasm32-browser --emit esm）
scripts/build

# Stage 2: Tiny Shakespeare LogicLadderGPT（LLVM + Metal ビルド）
scripts/kizu build --target native --backend llvm --opt -o target/griffith_llvm .
./target/griffith_llvm stage2-train-one l0 2000 web/ladder/models/l0.glgpt
./target/griffith_llvm stage2-train-one l3c 2000 web/ladder/models/l3c.glgpt web/ladder/models/l0.glgpt

# Stage 3 / LAB-04: M4 Max Metal GPU による日本語 BPE 論理ゲート LLM 段階 KL 蒸留（L0 -> L0b -> L3-C）
./target/griffith_llvm jpllm-train-one l0  1800 web/jpllm/models/l0.glgpt  - data/jpllm_corpus.txt
./target/griffith_llvm jpllm-train-one l0b 1800 web/jpllm/models/l0b.glgpt web/jpllm/models/l0.glgpt data/jpllm_corpus.txt
./target/griffith_llvm jpllm-train-one l3c 4200 web/jpllm/models/l3c.glgpt web/jpllm/models/l0b.glgpt data/jpllm_corpus.txt
```

---

## ドキュメント

- [docs/architecture.md](docs/architecture.md) — パイプライン全体設計・Zero-Atomic CSR 逆伝播・Metal GPU 設計
- [docs/experiments.md](docs/experiments.md) — Stage 1 回路圧縮＆速度ベンチ、Stage 2 Tiny Shakespeare、LAB-03 蒸留、Stage 3 Metal GPU 日本語 LLM の全実測ログ

---

## 謝辞・リファレンス

- **Griffith（本家）**: [tsukushi-gate/Griffith](https://github.com/tsukushi-gate/Griffith) / [@Tebasaki_lab](https://x.com/Tebasaki_lab) — 本プロジェクトは本家 Griffith の設計・アルゴリズムを Kizu 言語へ移植・応用した非公式実装です。
- **Kizu 言語**: [kizu-lang/kizu](https://github.com/kizu-lang/kizu)
- **Metal ランタイム参考実装**: [soyukke/kizukani](https://github.com/soyukke/kizukani)
- その他の帰属・ライセンス表記は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) および [LICENSE](LICENSE) を参照してください。
