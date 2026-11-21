# アーキテクチャ設計（Griffith × Kizu：学習〜回路最適化〜AOT 推論）

## 1. 全体パイプライン：「学習 ＝ 回路コンパイラ」

```text
[Corpus / Teacher Model (L0)]
        │
        ▼
  ┌──────────────────────────────────────────────────────────┐
  │ 1. Kizu Training Engine (CPU / Metal / Vulkan SPIR-V)    │
  │  - LutLayer (N=2,4,6 / softmax, warp / Gumbel-ST)        │
  │  - LogicLadderGPT (L0 → L0b → L1 → L2 → L3-C)            │
  │  - Zero-atomic CSR backward + AdamW + KD (蒸留)          │
  └──────────────────────────┬───────────────────────────────┘
                             │ discretize (argmax → bitmask LUT)
                             ▼
              [Circuit IR (JSON / .glir binary)]
                             │
                             ▼
  ┌──────────────────────────────────────────────────────────┐
  │ 2. Logic Synthesis & Circuit Optimizer (純 Kizu)         │
  │  - 定数畳み込み (FALSE/TRUE 伝播)                        │
  │  - パススルー/反転ゲートの配線縮約                       │
  │  - 死んだゲート除去 (DCE: Dead Code Elimination)         │
  │  - 共通部分式除去 (CSE: Structural Hashing)              │
  │  - LUT 融合 / AIG 化・トポロジカル再配置（キャッシュ最適化）│
  └──────────────┬───────────────────────────┬───────────────┘
                 │                           │
                 ▼                           ▼
  ┌─────────────────────────────┐  ┌─────────────────────────────┐
  │ 3A. Data-Driven Bit-Slice VM│  │ 3B. AOT Circuit Codegen     │
  │  - テーブル駆動 u64 64-way  │  │  - 回路 → 純 Kizu コード生成 │
  │  - 大規模ゲート向け         │  │  - 重みゼロの単体バイナリ   │
  │  - CPU / Metal / Vulkan     │  │  - ネイティブ & ブラウザWASM │
  └─────────────────────────────┘  └─────────────────────────────┘
                             │
                             ▼
             [3者ビット完全一致検証 & ベンチ]
       (Soft/Hard Eval == IR Sim == VM == AOT Binary)
```

---

## 2. なぜ「回路最適化 ＋ AOT Kizu コード生成」が肝なのか

通常の量子化 LLM（4bit / 1.58bit 等）は、結局「巨大な重み配列をメモリから読んで積和する」ため、既存大型 LLM を丸ごと論理ゲート化して C にしても命令キャッシュや帯域で不利になる。

逆に、論理ゲート NN が圧倒的に輝くのは **「キャッシュに収まる小〜中規模回路（数万〜数百万ゲート）を、論理合成で限界まで削り、ビット並列命令へ落とし込む領域」** である。

| 実行方式 | 仕組み | 向いている規模 | 強み |
|---|---|---|---|
| **AOT 回路コード生成（Kizu-to-Kizu）** | 最適化した LUT グラフを `u64` ビット演算関数（`&`, `|`, `^`, `~` の直線コード）として `.kizu` に出力し、`kizu build --opt`（LLVM / direct）で機械語化 | 数千〜数万ゲート（L1i/L2 キャッシュ常駐サイズ） | 重みファイルゼロ、配線インデックスのロード自体が消滅、レジスタ割当＋命令スケジューリングが効く、WASM 配布も単体で完結 |
| **データ駆動ビットスライス VM** | `indices[]` と `truth_tables[]` をキャッシュフレンドリーに並べ替え、`u64` 64-way 並列で評価 | 数万〜数百万ゲート（命令キャッシュを超える領域） | コード膨張なし、モデル差し替え自由、CPU マルチスレッド / Metal / Vulkan 共通 |
| **HDL 出力（Griffith JSON IR 経由）** | 本家 Griffith 互換の JSON IR を出力 | 任意の純論理回路 | 本家の Verilog / VHDL 生成や検証ツールをそのまま活かせる |

**設計上の重要検証テーマ**: ゲート数を 1k → 10k → 100k → 1M と振って、**「AOT コード生成がデータ駆動 VM に勝つクロスオーバーポイント」** と **「回路最適化（DCE/CSE）による実効ゲート削減率」** を実測する。

---

## 3. [`soyukke/kizukani`](https://github.com/soyukke/kizukani) からの参考・流用マップ

[`soyukke/kizukani`](https://github.com/soyukke/kizukani) の Kizu GPU ランタイム実績をベースに、Metal 実行基盤と数値一致テストの設計を踏襲している。

| モジュール (`kizukani`) | 活用方法 | 本プロジェクトでの役割 |
|---|---|---|
| `scripts/kizu`, `scripts/build`, `kizu.toml` | 構成踏襲 | `native` / `metal` / `spirv` / `wasm32-browser` の一括ビルド・テスト |
| `src/metal/metal.kizu` | 導入済み (`src/gpu/metal.kizu`) | Apple Silicon（M4 Max 等）での学習ランタイム、`ComputePass` まとめ投入 |
| `src/kernels/*.kizu` + `*_test.kizu` | 手本として踏襲 | `export "gpu" fn` の記述、CPU 参照実装との `< 1e-4` 数値一致テスト |
| `src/vulkan/vulkan.kizu` | 将来拡張候補 | Linux / Windows 向け Vulkan SPIR-V 学習への展開 |

---

## 4. Kizu `std::gpu` 制約に最適化した「Zero-Atomic GPU 逆伝播」設計

Kizu 標準の `std::gpu`（`lib/kizu/std/src/gpu/gpu.kizu`）は `thread_index` / `group_index` / `local_index` / `simdgroup_index` / `lane_index` / `barrier` / `group_array` / `shuffle_xor` / `simd_max` / `sqrt` / `exp` / `log` / `fma` / `f32_from_half` / `half_from_f32` を備える一方、現時点で `atomicAdd(f32)` は持たない。

そこで、**配線 `indices[N * out_dim]` の逆引き CSR テーブル（`rev_offsets`, `rev_gate_ids`, `rev_pin_ids`）** を使うことで、すべての学習カーネルを **atomic 一切なし・決定的（ビット再現可能）** に設計する：

1. **`lut_forward`**（1 スレッド = `(batch_token b, out_gate o)`）
   - $N$ 本の入力を gather → バタフライ展開で $2^N$ 基底（WARP は Walsh-Hadamard 基底）計算 → `softmax` / `warp` / `gumbel_st` で出力。
   - 活性バッファは `f16`（`half_from_f32` / `f32_from_half`）保存オプションを用意し、Blackwell / Metal のメモリ帯域消費を半減できるようにする。
2. **`lut_backward_w`**（1 スレッドグループ = 1 ゲート `o`、スレッドはバッチ方向 `b` を並列担当）
   - 各スレッドがレジスタで `grad_w_local[k]` を累積し、`gpu::shuffle_xor` + `group_array` でグループ内リダクション → `grad_w[o, k]` に 1 回書き込み。**atomic 不要**。
3. **`lut_backward_x`**（1 スレッド = `(batch_token b, in_wire i)`）
   - 入力ワイヤ `i` が `rev_offsets[i]..rev_offsets[i+1]` を辿ってファンアウト先ゲートのピン勾配を gather して合算 → `grad_x[b, i]` へ書き込み。**atomic 不要＆加算順序が決定的**。
4. **`adamw_step`**: パラメータ要素並列の 1 パス更新。**atomic 不要**。

---

## 5. 論理ゲート LLM の核心設計（L1 の壁を攻める）

本家 Griffith の `LogicLadderGPT` の結果から、最大の課題は **L1（Q/K/V 射影の LUT 化）での精度低下** と **言語モデル特有の大きな離散化ギャップ** である。

| 段 | Embedding / Norm / Head | FFN | Q/K/V 射影 | Attention Score & 集約 | 狙い |
|---|---|---|---|---|---|
| **L0** | 実数 (`f32`) | 実数 Linear + GELU | 実数 Linear | 内積 + softmax + 加重和 | 学習基盤の検証、天井計測、**蒸留の教師（Teacher）** |
| **L0b** | 実数 (`f32`) | **LUT (`WARP N=4/6`)** | 実数 Linear | 内積 + softmax + 加重和 | FFN の乗算排除＆最初のハイブリッド比較 |
| **L1** | 実数 (`f32`) | **LUT** | **LUT** | 内積 + softmax + 加重和 | **最大の難所**。WARP + Gumbel-ST + L0 からの蒸留で攻略 |
| **L2** | 実数 (`f32`) | **LUT** | **LUT** | **XNOR-popcount** + softmax | 内積の乗算を排除 |
| **L3-C** | 実数 (`f32`) | **LUT** | **LUT** | **XNOR-popcount + argmax MUX (Gumbel-ST)** | Attention 完全論理回路化（本家基準: hard ppl 13.31） |
| **Full Logic** | `LogicEmbedding` + `LutStack` + `GroupSum` | **LUT** | **LUT** | **XNOR + MUX** | 全層論理回路化 → 回路最適化 → 重みゼロ AOT バイナリ / WASM |

### L1 の壁・離散化ギャップへの攻撃カード（実験順）

1. **WARP-LUTs ($N=4, 6$)**: Walsh-Hadamard 基底で多入力 LUT を安定学習し、ゲート数 1/4・離散化ギャップ極小化。
2. **Gumbel-ST / Hard-ST**: 学習中の forward から離散回路を通し、soft/hard ギャップを潰す。
3. **実数 L0 Teacher からの知識蒸留（KL + 中間 Attention 分布蒸留）**: 「L1 で Attention 分布が一様に潰れる」という本家報告の現象に対し、L0 の Attention 分布＆出力ロジットを直接ガイドにすることで回避を狙う。
4. **学習可能配線（Learnable Wiring / DWN）**: 固定ランダム配線だと情報が落ちる Q/K/V 射影やボトルネック層で、接続先も学習して表現効率を上げる。

---

## 6. ディレクトリ構成

```text
Griffith/
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── kizu.toml
├── docs/
│   ├── architecture.md   # アーキテクチャ設計
│   └── experiments.md    # 実測ベンチマーク・実験ログ
├── scripts/
│   ├── kizu              # Kizu コンパイララッパー
│   ├── build             # native / metal / spirv / wasm32-browser ビルド
│   ├── verify_wasm.mjs   # 全 5 ブラウザデモ統合 WASM 検証
│   └── gen_jpllm_corpus.py
├── src/
│   ├── main.kizu         # CLI & wasm32-browser エントリ
│   ├── core/             # 決定的 PRNG, 数値ユーティリティ
│   ├── lut/              # LutLayer (WARP / Softmax / Gumbel-ST / CSR 逆引き)
│   ├── synth/            # 4 パス論理合成＋64 並列ビットスライス VM
│   ├── codegen/          # 重みゼロ AOT Kizu コード生成 / Griffith IR JSON 出力
│   ├── data/             # 文字コーパス＆バイトレベル BPE トークナイザ
│   ├── model/            # LogicLadderGPT (L0..L3-C), KL 蒸留, .glgpt v1/v2 IO
│   ├── gpu/              # Apple Silicon Metal ランタイム＆学習エンジン
│   ├── kernels/          # export "gpu" fn (Linear & WARP-LUT 前後伝播・CSR 逆伝播)
│   ├── cli/              # Stage 1 / Stage 2 / 日本語 LLM 学習 CLI
│   └── browser/          # 全 5 デモ統合 wasm32-browser ブリッジ
└── web/                  # 静的ブラウザデモ一式（そのままホスティング可）
    ├── index.html        # デモ一覧ハブ
    ├── dist/             # 統合 app.wasm + app.mjs
    ├── draw/             # LAB-01: 5×5 手描きライブ学習 → 論理合成
    ├── compiler/         # LAB-01b: 論理合成＆64 並列 VM vs AOT 速度対決
    ├── ladder/           # LAB-02: LogicLadderGPT L0..L3-C + Attention MUX
    ├── distill/          # LAB-03: 日本語マイクロ秒ルーター（KL 知識蒸留）
    └── jpllm/            # LAB-04: 日本語 4 モード論理ゲート LLM
```

---

## 7. 品質・検証契約

1. **数値微分チェック**: CPU の解析的 backward（`softmax` / `warp` / `gumbel_st` / `GroupSum` / `XNOR-MUX`）はすべて有限差分と一致することを `kizu test` で常時検証する。
2. **回路最適化の等価性保証**: 最適化前 IR と最適化後 IR（定数畳み込み・DCE・CSE 後）の出力が全テスト入力で **0 mismatch** であることを保証する。
3. **4 者ビット完全一致**: `Hard Eval == IR Sim == Bit-Slice VM == AOT 生成バイナリ` の 0 mismatch を自動テストする。
4. **CPU vs GPU 一致**: すべての `export "gpu" fn` カーネル（Metal / Vulkan）は CPU 参照実装と許容誤差内で一致することを検証する。
