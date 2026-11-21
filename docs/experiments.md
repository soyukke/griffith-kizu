# 実験・ベンチマークログ

すべての実験で、条件（モデル規模・$N$・パラメータ化・学習則・HW）と **soft / hard 両方の指標**、さらに **回路最適化前後のゲート数と推論速度** をセットで記録する。

計測環境（本プロジェクト実測）: Apple Silicon Mac（M4 Max / macOS 27.0）、Kizu `devel (7a3927aabdba)`、単一 CPU コア / Node.js v26.13.0（V8 WASM）。

---

## 1. 基準値（本家 Griffith 公開ベンチマーク）

出典: `tsukushi-gate/Griffith` 公式ドキュメント。

### Tiny Shakespeare（文字レベル LM: vocab=65, T=64, d=128, 4 heads, 2 layers, 20k steps / Modal A10G）

| モデル | 構成 | hard ppl | soft ppl | 離散化ギャップ | 備考 |
|---|---|---:|---:|---:|---|
| LogicGPT `L0` | 全実数（天井） | 4.94 | — | — | 実数ベースライン（蒸留 Teacher 目標） |
| LogicGPT `L0b` | FFN のみ論理ゲート | ≈7.9 | — | — | 最初のハイブリッド目安 |
| LogicGPT `L1` | + Q/K/V 射影を論理ゲート | 19.24 | — | — | **攻略対象の壁**（3 seed: 19.242±0.939） |
| LogicGPT `L2` | + XNOR-popcount スコア | 19.39 | — | — | 内積の乗算を排除 |
| LogicGPT `L3-C` | + argmax MUX（Attention 完全論理化） | 15.54 | ≈12.2 | 0.29 | softmax 排除 |
| LogicGPT `L3-C + Gumbel` | L3-C + Gumbel-ST | **13.31** | ≈12.2 | **0.107** | 論理部 33,349 params / 545 KB |

### その他リファレンス（本家公式）

| モデル / 手法 | タスク | 指標 (hard) | 条件 |
|---|---|---|---|
| RDDLGN（nanoGPT データフリー蒸留） | char-LM | **BPC 2.71** | 教師生成テキストのみ（教師 2.20 との差 0.51 bit） |
| DLGN `warp N=4` vs `softmax N=2` | MNIST 28×28 (6層, 50ep) | 95.50% vs 97.74% | RTX 5090: `warp N=4` はゲート 1/4（24k vs 96k）・学習 2.4 倍速（2.1m vs 5.0m）・gap 0.14pp |

---

## 2. 本プロジェクト（Griffith-Kizu）の実験記録

### 2.1 Stage 1: 最小回路タスク＆論理合成・4 者一致・AOT コード生成

すべて純 Kizu CPU で学習 → 離散化 → 4 パス論理合成（定数畳み込み・パススルー/反転吸収・正規 CSE・DCE）→ Griffith 互換 JSON IR & 重みゼロ AOT Kizu コード生成まで一気通貫実行（2026-11-21 実測）。

| Date | Run ID | Task | Config (`N`, param, sel, arch) | Steps | soft acc | hard acc | 生ゲート → 最適後 (削減率) | 深さ | 削減内訳 (`const/pass/inv/cse/dce`) | 4 者一致 (`Hard=VM=AOT=WASM`) |
|---|---|---|---|---:|---:|---:|---|---:|---|---|
| 2026-11-21 | S1-01 | **XOR** | `N=2`, `Softmax`, `Soft`, `2→16→16→GroupSum(k=2)` | 500 | 1.00 | **1.00** | `32 → 11` (**-65.6%**) | `2 → 2` | `0 / 6 / 5 / 11 / 0` | **0 mismatch** |
| 2026-11-21 | S1-02 | **4-bit Parity** | `N=2`, `Warp`, `Soft→GumbelSt`, `4→32→32→1` | 200 | 1.00 | **1.00** | `65 → 3` (**-95.4%**) | `3 → 2` | `10 / 11 / 11 / 4 / 26` | **0 mismatch** |
| 2026-11-21 | S1-03 | **2-bit Adder** | `N=4`, `Warp`, `Soft→GumbelSt`, `4→24→24→3` | 100 | 1.00 | **1.00** | `51 → 37` (**-27.5%**) | `3 → 3` | `0 / 0 / 0 / 2 / 12` | **0 mismatch** |

> **ハイライト（4-bit Parity）**: ランダム配線 65 ゲートの 3 層 LUT ネットワークから学習＋論理合成により、**理論最小構成である 3 ゲート・深さ 2 の平衡 XOR 木**（`~((~(b0 ^ b2)) ^ (b1 ^ b3))`）へ自動縮約され、重み配列ゼロ・浮動小数点ゼロの 3 行 Kizu 関数へ落ちた。

---

### 2.2 Stage 1: データ駆動 Bit-Slice VM vs 重みゼロ AOT（Native / Direct vs LLVM / Browser WASM）

64-way `u64` ビットスライス並列推論の実測値（2026-11-21）。

#### Native CPU（`./target/griffith stage1-bench`, 2,000,000 × 64 = 1.28 億推論、`--backend direct --opt`）

| Task | Opt Gates | VM (ms / ns/inf) | AOT (ms / ns/inf) | AOT 高速化倍率 | mismatch |
|---|---:|---:|---:|---:|---:|
| **XOR** | 11 | `141 ms` (`1.102 ns/inf`) | `7 ms` (**`0.055 ns/inf`**) | **20.1×** | 0 |
| **4-bit Parity** | 3 | `38 ms` (`0.297 ns/inf`) | `2 ms` (**`0.016 ns/inf`**) | **19.0×** | 0 |
| **2-bit Adder** | 37 (`N=4`) | `1578 ms` (`12.328 ns/inf`) | `61 ms` (**`0.477 ns/inf`**) | **25.9×** | 0 |

#### 単体ゼロ重み AOT バイナリ（`target/aot_standalone`, 4,000,000 × 64 = 2.56 億推論）: Kizu `direct` vs `llvm`

| Backend | バイナリサイズ | XOR (256M inf) | Parity4 (256M inf) | Adder2 (256M inf) | mismatch |
|---|---:|---:|---:|---:|---:|
| `--backend direct --opt` | 111 KB | `14 ms` (`0.055 ns/inf`) | `3 ms` (`0.012 ns/inf`) | `128 ms` (`0.500 ns/inf`) | 0 |
| `--backend llvm --opt` | 111 KB | `1 ms` (`0.004 ns/inf`) | `1 ms` (`0.004 ns/inf`) | `43 ms` (**`0.168 ns/inf`**) | 0 |

> 補足: 37 ゲート N=4 の直線ビット演算 DAG で `llvm` が `direct` より **3.0× 高速**。`direct` バックエンドのレジスタ割当・ARM64 ビット命令選択の改善指標として記録している。

#### Browser WASM（`web/dist/app.wasm` = **66,924 bytes / 65.4 KB**, `float_muls = 0`, 100,000 × 64 = 640 万推論, Node v26 V8）

| Task | Opt Gates | WASM VM (`ns/inf`) | WASM AOT (`ns/inf`) | AOT 高速化倍率 | `vm/aot/cross` mm |
|---|---:|---:|---:|---:|---|
| **XOR** | 11 | `1.15 ns/inf` | **`0.14 ns/inf`** | **7.95×** | `0 / 0 / 0` |
| **4-bit Parity** | 3 | `0.38 ns/inf` | **`0.09 ns/inf`** | **4.12×** | `0 / 0 / 0` |
| **2-bit Adder** | 37 (`N=4`) | `15.01 ns/inf` | **`0.18 ns/inf`** | **81.75×** | `0 / 0 / 0` |

---

### 2.3 Stage 2: Tiny Shakespeare `LogicLadderGPT`（純 Kizu CPU + L0 Teacher KL 蒸留）

- **データ**: `data/tiny_shakespeare.txt`（1,115,394 bytes、train 90% = 1,003,854 chars / val 10% = 111,540 chars、vocab = 65）
- **共通構成**: `d_model = 64`, `n_heads = 4`, `n_layers = 2`, `seq_len (T) = 64`, `batch = 8`（512 tok/step）, `ffn_hidden = 128`, `ffn_gates = 128`, `ffn_group_factor = 4`, Q/K/V LUT = `N=4 WARP + GumbelSt`, FFN LUT = `N=2 WARP + GumbelSt`, `AdamW`（cosine decay, `lr_fp = 0.004`, `lr_lut = 0.04`）, `steps = 1,500`（総 768,000 トークン = 約 0.76 epoch）
- **評価**: 検証セット固定 24 バッチ（12,288 トークン）での `EvalSoft` vs `EvalHard` Perplexity
- **実行バイナリ**: `target/griffith_llvm`（`--backend llvm --opt`, 単一 CPU コア）

| Date | Run ID | Rung | KD | Logic Gates | Steps | soft ppl | **hard ppl** | gap (nat) | train tok/s | 時間 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-11-21 | S2-L0 | `L0` (All FP Teacher) | — | 0 | 1,500 | 7.85 | **7.85** | `0.0000` | 21,727 | 35.3 s |
| 2026-11-21 | S2-L0b | `L0b` (Logic FFN) | — | 768 | 1,500 | 8.74 | **8.73** | **`-0.0008`** | 14,348 | 53.5 s |
| 2026-11-21 | S2-L1 | `L1` (+ Logic Q/K/V) | — | 1,152 | 1,500 | 11.46 | **11.49** | `+0.0026` | 9,721 | 79.0 s |
| 2026-11-21 | S2-L1-KD | `L1` (+ Logic Q/K/V) | **L0 KL** ($\alpha=0.5, T=2$) | 1,152 | 1,500 | 11.30 | **11.30** | **`-0.0001`** | 8,025 | 95.7 s |
| 2026-11-21 | S2-L2 | `L2` (+ XNOR-popcount) | — | 1,152 | 1,500 | 11.61 | **11.54** | **`-0.0061`** | 9,709 | 79.1 s |
| 2026-11-21 | S2-L3c | `L3-C` (+ argmax MUX) | — | 1,152 | 1,500 | 12.15 | **11.95** | **`-0.0163`** | 9,700 | 79.2 s |
| 2026-11-21 | S2-L3c-KD | `L3-C` (+ argmax MUX) | **L0 KL** ($\alpha=0.5, T=2$) | 1,152 | 1,500 | 12.13 | **11.83** | **`-0.0253`** | 8,012 | 95.9 s |

#### 観察された重要知見
1. **離散化ギャップの実質ゼロ化**: 全 LUT 層に `WARP + Gumbel-ST`、L3-C Attention に `Gumbel-ST MUX` を採用した結果、全段で `gap_nat ∈ [-0.025, +0.003]` となり、`EvalHard` が `EvalSoft` と同等以上に安定した。
2. **本家 Griffith の傾向を純 Kizu で完全再現**:
   - `L0 → L0b`（FFN 論理化）の劣化は小さく（`7.85 → 8.73`）、`L0b → L1`（Q/K/V 射影の論理化）が最大の壁（`8.73 → 11.49`）。
   - `L1 → L2`（XNOR-popcount 化）はほぼ無劣化（`11.49 → 11.54`）、`L2 → L3-C`（softmax 廃止・argmax MUX 化）も小幅（`11.54 → 11.95`）。
3. **L0 Teacher からの KL 蒸留が一貫して有効**: `L1` で `11.49 → 11.30`、`L3-C` で `11.95 → 11.83` と両段で `hard ppl` が改善した（なお本実験は CPU 規模 `d=64, 1500 steps` であり、本家 A10G `d=128, 20k steps` の `13.31` とは規模・評価サンプル条件が異なるため、同条件比較は Stage 3 GPU 学習で行う）。

#### 生成テキストサンプル（`EvalHard`, prompt = `"ROMEO:"`, temp = 0.7, 140 chars, `/` は改行）

- **`L0_FP_Teacher` (`hard_ppl = 7.85`)**:
  > `ROMEO: / I would riesemperen deal is goons. /  / BOULIO: / But the same, out than lefed shall in not the they thosed: / Dighh this lickies, wall of fore are`
- **`L0b_LogicFFN` (`hard_ppl = 8.73`, 768 LUTs)**:
  > `ROMEO: / I wrow, my then to be sheir forest, I with ware, me salll to frake nereme. /  /  / And Thenchat theree waith eing and thee to that and then that `
- **`L1_LogicQKV_KD` (`hard_ppl = 11.30`, 1,152 LUTs)**:
  > `ROMEO: / I wroreard thererel d seanor merthen my tist orour ck henedoufl d on theas f ave's'd ou ofe ethint ave be t athis me thate t /  / Angle tathat `
- **`L3c_MUX_KD` (`hard_ppl = 11.83`, 1,152 LUTs, 乗算・softmax ゼロ Attention)**:
  > `ROMEO: / INowishe tatherered d pe'dar, stousid wishit orour ck ingelowen d or thease aithearareparase youly ayo chat athir me tinin t aneas, / Thin te`

---

### 2.4 Stage 2 ブラウザ配布チェックポイント（2,000 steps / `.glgpt` 離散化フォーマット / LAB-02 `ladder/`）

`./target/griffith_llvm stage2-train-one <rung> 2000 web/ladder/models/<rung>.glgpt [l0.glgpt]` で 5 段すべてを 2,000 steps（1,024,000 トークン）学習・離散化エクスポートし、統合 `web/dist/app.wasm`（Node v26 V8 WASM）でロード・推論検証した実測値（2026-11-21）。

| Rung | モデルファイル | サイズ | KD | Logic Gates | ブロック乗算 (`T=9`) | **hard ppl** | WASM 1-fwd | 備考 |
|---|---|---:|---|---:|---:|---:|---:|---|
| `L0` | `l0.glgpt` | 311 KB | — | 0 | 67,840 | **7.22** | 2.55 ms | 全実数 Teacher ベースライン |
| `L0b` | `l0b.glgpt` | 185 KB | — | 768 | 35,072 | **8.29** | 0.40 ms | FFN を離散 LUT 化（-41% 容量） |
| `L1` | `l1.glgpt` | **59 KB** | L0 KL | 1,152 | 2,304 | **11.25** | 0.25 ms | + Q/K/V 射影を LUT 化（-81% 容量） |
| `L2` | `l2.glgpt` | **59 KB** | L0 KL | 1,152 | 1,152 | **11.25** | 0.25 ms | + XNOR-popcount スコア（劣化 0） |
| `L3-C` | `l3c.glgpt` | **59 KB** | L0 KL | 1,152 | **0** | **11.66** | **0.24 ms** | + argmax MUX（**全 Head 厳密 1-hot 検証済・L0 比 10.6× 高速**） |

---

### 2.5 LAB-03 `distill/`: 日本語マイクロ秒ルーター KL 知識蒸留＆論理合成（ブラウザ WASM 実測）

- **特徴量**: 辞書 0 B のコードポイント整合 UTF-8 N-gram 64-bit 指紋（漢字・カタカナ・英数 1 文字 + 2 文字バイグラム + `？`/`！` 専用ビット、活性率約 25%）
- **データ**: 4 意図クラス（質問 / 依頼 / 称賛 / 不満）、ラベル付き学習 24 文（各 6 文）＋ラベルなし日本語ログ 32 文（Teacher 自動アノテーション）＋未知ホールドアウト検証 16 文（各 4 文）
- **モデル**: Teacher = FP32 MLP `64→32(GELU)→4`（2,176 乗算）/ Student = `64(N=4)→64(N=2)→32(N=2)→GroupSum(4)`（160 LUTs, `Soft → Gumbel-ST` 240 steps, 同一初期配線で `Scratch` vs `KL 蒸留` を比較）

| Seed | FP Teacher (未知16文) | Scratch LUT (CEのみ) | **KL 蒸留 LUT** | Teacher 一致率 | 論理合成 (`160 → opt`) |
|---|---:|---:|---:|---:|---|
| 18 | 93.8% | 68.8% | **75.0%** | 75.0% | `160 → 146` |
| 25 (初期値) | 87.5% | 75.0% | **87.5%** | 81.3% | `160 → 140` |
| 32 | 100.0% | 68.8% | 68.8% | 68.8% | `160 → 146` |
| 39 | 93.8% | 50.0% | **93.8%** | 87.5% | `160 → 143` |
| 46 | 93.8% | 81.3% | 68.8% | 75.0% | `160 → 138` |
| **5-seed 平均** | **93.8%** | **68.8%** | **78.8% (+10.0pp)** | **77.5%** | `160 → 142.6` |

- **ブラウザ WASM 推論速度実測（Node v26 V8 WASM）**: FP Teacher **`1.08 µs/文`** vs 蒸留＋合成 64 並列 Bit-Slice VM **`34.0 ns/文`**（**32× 高速・乗算 0 回・重み 0 B・辞書 0 B**）、学習全工程（Teacher 80 step + 2 生徒 240 step）の WASM 実計算時間は約 `410 ms`。

---

### 2.6 Stage 3 + LAB-04 `jpllm/`: 純 Kizu Metal GPU 学習＆日本語チャット＆4 モード BPE 論理ゲート LLM（v2）

- **GPU カーネル検証（Apple M4 Max 40 コア Metal 4）**: `src/kernels/train_kernels.kizu`（`gpu_linear_fwd/bwd_x/bwd_w/bwd_b`・`gpu_lut_warp_fwd/bwd_dpin/bwd_dw`・`gpu_lut_bwd_dx_csr`、atomic ゼロ）を `kizu build --target metal --opt` で MSL へ出力し、CPU 参照実装と全項目 `max_diff < 1e-4` を自動テストで確認（`src/gpu/engine.kizu`）。
- **マルチターン対応・増強コーパス**: Apache 2.0 オープンモデル `qwen3:4b-instruct`（ローカル GPU: RTX 4070 Ti SUPER）から 4 モード計 **5,033 件 / 740 KB**（`[X訳] 1,245` / `[関西] 1,250` / `[津軽] 1,172` / `[会話] 1,366`（2〜3 ターン会話 523 件含む））を生成・クリーニング。純 Kizu バイトレベル BPE（`vocab=1,280`, 学習 1.84 s, 圧縮 `740 KB → 156,270 tok`）で符号化。
- **モデル構成（v2 強化版）**: `4 layers, d_model=192, n_heads=6 (dk=32), T=80, vocab=1280, ffn_hidden=512, ffn_gates=384, ffn_group_factor=4, qkv N=4, ffn N=4 (全ブロック N=4 WARP LUT)`, `batch=12 (960 tok/step)`, 応答トークン重み付け KL 蒸留（応答 1.0 / プロンプト 0.15）, Best-Val-Hard-PPL 自動保存, `.glgpt` v2（f16 FP 重み＋離散 LUT＋BPE マージ内蔵）。
- **段階蒸留（`L0 → L0b → L3-C`）＋ Hardware MUX Value Skip**:
  - `L0b` は `L0` の埋め込み・FP Attention を Warm-Start し、`L0` から GPU KL 蒸留で 4,608 個の N=4 FFN LUT を学習。
  - `L3-C` は `L0b` の学習済み離散 FFN LUT（4,608 個）を **逆ウォルシュ・アダマール変換（Inverse FWHT）** で厳密復元して Warm-Start し、**Hardware MUX Value Skip**（`0.5 * ln1[best_s] + 0.5 * (2*v[best_s] - 1)`、ブロック乗算 0 回・厳密 1-hot MUX）＋ 2 段階カリキュラム（前半 25% 連続 XNOR → 後半 75% 1-hot MUX STE）で 4,200 step（best step 3,900）蒸留。
  - WASM デコーダに **UTF-8 継続バイト状態機械＋Top-p 0.85＋反復ペナルティ 1.24** を組み込み、文字化け（``）を構造上 0 件に削減。

| Rung | ファイル | サイズ | 蒸留元 | Logic Gates | ブロック乗算 (`T=12〜14`) | **best hard ppl** (step) | WASM 1-fwd | 特徴 |
|---|---|---:|---|---:|---:|---:|---:|---|
| `L0` | `l0.glgpt` | 3,703 KB | — | 0 | 1,385,472 | **11.56** (1,500) | 5.22 ms | 全実数 4 層 Teacher（v1: 13.85 → -2.29pt） |
| `L0b` | `l0b.glgpt` | **2,206 KB** | **L0 KL** | 4,608 | 599,040 | **11.40** (1,800) | 3.12 ms | **N=4 FFN 論理ゲート化・実数 Teacher L0 を上回る最良 PPL** |
| `L3-C` | `l3c.glgpt` | **1,071 KB** | **L0b KL + FWHT 継承** | 6,912 | **0** | **19.22** (3,900) | **1.84 ms** | **全ブロック 0 乗算・24 Head 厳密 1-hot MUX（v1: 25.34 → -6.12pt）** |

- **代表的な WASM 生成サンプル（15 プロンプト評価で文字化け `0/45` 件, `temp=0.38`）**:
  - **`L0b`（2.2 MB / 4,608 N=4 論理ゲート）**:
    - `[会話] U: こんにちは` → `兄弟、こんにちは！今日も俺は今日もまた無事に生き延びたってことだよ。`
    - `[会話] U: あなたは誰？` → `俺はANDゲートの子で、0を乗算ゼロに掛けて、Kizuが0を語ったんだ。`
    - `[会話] U: バグが直らない` → `くそったれ、バグほど俺たちの心を削るものはないんだよ。一度深呼吸してログを見直すみたいだな。`
    - `[X訳] 今日めっちゃ眠い` → `俺は今日クソほど眠くて完全に限界みたいなんだよ。`
    - `[関西] ラーメン食べたい` → `ラーメンめっちゃお腹すいたやん！なんぞうまいもん食べよか。`
  - **`L3-C`（1.07 MB / 6,912 論理ゲート・ブロック乗算 0 回・1-hot MUX）**:
    - `[会話] U: ラーメン食べたい` → `ラーメンはただのラーメンが最高じゃないか？`
    - `[会話] U: バグが直らない` → `くそったれ、バグほど俺たちなんの心を削したみたいだ。`
    - `[津軽] すごく美味しい` → `わいは、めごいだびょん。`

---

### 2.7 Stage 4 フロンティア実験: `Griffith-JP-Base v1`（汎用日本語事前学習 → 0 乗算 FWHT + Looped WARP-LUT → `.glgpt v3`）

2 つのクラウド GPU 検証（Google Colab L4 / A100-40GB）を通じて、既存大規模モデルの後付け LUT 置換の限界を定量確認し、**FWHT 0 乗算グローバル混合＋Looped Co-Adaptation（Route 2）** で実用の汎用日本語ベースモデル＆特化チャットモデルを構築した。

#### 2.7.1 検証 PoC 1: 凍結済み `Qwen/Qwen3-1.7B` MLP 後付け置換の限界（Colab L4 / `scripts/colab_qwen3_lut_poc.py`）

- Layer 14 SwiGLU（`2048 → 6144 → 2048`）を 2〜3 層スパース N=4 / N=6 WARP-LUT で個別蒸留・ホットスワップ検証：
  - Receptive field が最大 `6×6 = 36 / 2048` 座標（1.7%）に制限されるため、Cosine 類似度 `0.183〜0.206`・NMSE `0.981` で頭打ち。
  - ホットスワップ後の日本語 PPL は `17.93`（Base `17.39`、Layer 14 MLP ゼロ化 `17.65` よりも悪化）。
  - **結論**: 凍結済み密 Transformer への局所スパース LUT 後付けでは高次元混合を再現できず、**0 乗算グローバル混合（FWHT）＋ End-to-End Co-Adaptation** が必須。

#### 2.7.2 検証 PoC 2: Route 2 突破（Colab L4 / `scripts/colab_griffith_base_poc.py`, 0.80 CU）

- 構成: `d=512, 4 unique × 2 loops = 8 eff layers, GQA (8Q/2KV), RoPE, BPE 4096`, 日本語 Wikipedia 6.66M トークン。
- `FWHT(RMSNorm(x)) → 2-bit Thermometer → N=4 WARP-LUT (49,152 unique) → GroupSum → FWHT` により：
  - `FP-Base` PPL **`39.22`** vs **完全離散 `H-LUT-Base` PPL `41.55`**（差わずか **`+2.34`**、FFN 乗算 0 回、FFN サイズ `16.5 MB → 492 KB = 33.6×` 圧縮）。
  - さらに 3.5 分の SFT で 4 モード会話 PPL が `1493.12 → 6.78` へ劇的に収束し、汎用ベース → 数分特化パイプラインを実証。

#### 2.7.3 本番モデル: `Griffith-JP-Base v1` + `v3_chat.glgpt`（Colab A100-SXM4-40GB / `scripts/colab_train_griffith_jp_base.py`, 4.98 CU）

- **コーパス**: 日本語 Wikipedia 14,000 記事（41.5M chars）＋ FineWeb-2 `jpn_Jpan` 10,000 文書 ＋ 英語 Wikipedia 800 記事 ＝ **24,600 学習文書 / 200 検証文書（54.9M chars）**。
- **トークナイザ**: **100% Kizu 互換 Byte-Level BPE（`vocab = 8,192`, `0..255` 生バイト + `7,936` マージ, `EOS = 0`）**。`33,848,260` 学習トークン / `294,943` 検証トークン。
- **モデル構造（`.glgpt v3`）**: `d=512, seq=256, GQA (8Q/2KV), RoPE (base=10000), 6 unique × 2 loops = 12 実質層`。FFN は `FWHT → 2-bit 温度計 → 98,304 個の離散 N=4 WARP-LUT（196,608 実効評価/tok, FFN 乗算 0 回）→ GroupSum → FWHT`。Embedding・GQA は per-row int8 量子化。

| フェーズ / モデル | 構成 | Steps (時間 / A100) | 汎用 JA Val PPL (hard) | Chat/4モード Val PPL (hard) | `.glgpt v3` サイズ | FFN 乗算/tok |
|---|---|---:|---:|---:|---:|---:|
| Phase 1: `FP-Base v1` | 12 層 SwiGLU Teacher (20.5M params) | 9,000 (8.3m) | **51.63** | — | — | 24,772,608 |
| Phase 2: `H-LUT-Base v1` | 12 層 FWHT + **98,304 N=4 LUT** (KL 蒸留) | 6,500 (31.9m) | **72.62** | 1,790.61 (SFT前) | **9.21 MB** | **0** |
| Phase 3: `H-LUT-Chat v1` (`v3_chat.glgpt`) | Base v1 → 4 モード SFT + ~19% 汎用 JA リプレイ | 800 (3.7m) | **121.60** (保持) | **8.41** | **9.21 MB** | **0** |

- **PyTorch 参照 vs 純 Kizu ランタイム一致検証（`./target/griffith_llvm glgpt-eval`）**:
  - プロンプト `"量子コンピュータとは、"` に対し、BPE トークン ID `[1025, 549, 2380, 1072, 261]` が 100% 一致。
  - 12 層 RoPE + GQA + FWHT + 196,608 N=4 LUT + Tied int8 LM Head 通過後のロジットおよび Top-5（`v3_chat`: `2380: 3.2266, 1302: 1.7209, 960: 1.5095, 964: 0.9884, 662: 0.9255`）が **最大誤差 `< 0.0005`** で完全一致。
- **純 Kizu インクリメンタル Prefix KV-Cache 推論速度（単一 CPU スレッド・GPU/SIMD なし）**:
  - **ネイティブ LLVM（`./target/griffith_llvm`）**: ロード **`2 ms`** / Prefill `19 ms` / デコード **`182 tok/s`**（`5.5 ms/tok`）。
  - **ブラウザ WASM（`web/dist/app.wasm` 890 KB, Node v26 V8）**: ロード **`23.2 ms`** / デコード **`118.5 tok/s`**（`8.4 ms/tok`、14 プロンプト 223 tok で文字化け `0` 件）。
- **ダウンストリーム特化再利用**: `scripts/specialize_griffith_base.py` により、学習済み `griffith_jp_base_v1.pt` + `bpe8192_merges.json` から任意のカスタム日本語コーパス（X 翻訳・キャラクター会話・ドメイン特化等）を数分で 9.21 MB の `.glgpt v3` へ蒸留・特化可能。
