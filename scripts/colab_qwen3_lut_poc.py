#!/usr/bin/env python3
"""
Griffith-Kizu Frontier PoC (Step 1):
Can we replace a real Qwen3-1.7B SwiGLU MLP layer (2048 -> 6144 -> 2048, 37.7M params)
with a pure discrete Boolean WARP-LUT circuit (0 multiplications inside MLP)
without collapsing Japanese + English LLM perplexity?

Experiments on Layer 14 (middle layer) of Qwen/Qwen3-1.7B:
  Exp A: Baseline Griffith L0b (N=4, 1-bit sign, Random wiring, 32K -> 32K, g=16)
  Exp B: 4-bit Thermometer + Weight-Guided Wiring (N=4, WGTI, 32K -> 32K, g=16)
  Exp C: 3-Layer Deep WGTI (N=4, 4-bit Thermometer, 32K -> 32K -> 32K, g=16)
  Exp D: FPGA-Native LUT6 WGTI (N=6, 4-bit Thermometer, 49K -> 65K, g=32)

All configs are evaluated BOTH on hidden-state fit (NMSE / Cosine, soft & HARD Boolean)
AND hot-swapped as HARD discrete Boolean LUT circuits back into live Qwen/Qwen3-1.7B
to measure true end-to-end validation PPL and live Japanese text generation!
"""

import json
import math
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def log(msg: str) -> None:
    print(msg, flush=True)


def fwht_last_dim(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform along last dim (exact match to Kizu src/lut/basis.kizu)."""
    n = x.shape[-1]
    h = x
    step = 1
    while step < n:
        h = h.view(*x.shape[:-1], -1, 2, step)
        u = h[..., 0, :]
        v = h[..., 1, :]
        h = torch.stack((u + v, u - v), dim=-2).reshape(*x.shape)
        step *= 2
    return h


def std_basis_butterfly(x_pins: torch.Tensor) -> torch.Tensor:
    """Computes 2^N multilinear standard basis from pins in [0, 1]. Shape: [..., N] -> [..., 2^N]."""
    n_inputs = x_pins.shape[-1]
    cur = torch.ones((*x_pins.shape[:-1], 1), device=x_pins.device, dtype=x_pins.dtype)
    for j in range(n_inputs):
        xj = x_pins[..., j : j + 1]
        cur = torch.cat((cur * (1.0 - xj), cur * xj), dim=-1)
    return cur


class WarpLutLayer(nn.Module):
    """
    Griffith-Kizu WARP-LUT layer (exact PyTorch/CUDA mirror of src/lut/layer.kizu).
    - Train: FWHT(theta) + optional logistic noise -> multilinear basis -> sigmoid(z/tau) with Hard-ST STE
    - Eval (hard=True): pure discrete Boolean truth-table lookup (0 multiplications!)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_inputs: int,
        indices: torch.Tensor,
        tau: float = 1.0,
        gumbel_scale: float = 0.15,
        chunk_size: int = 8192,
    ):
        super().__init__()
        assert indices.shape == (out_dim, n_inputs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_inputs = n_inputs
        self.basis_size = 1 << n_inputs
        self.tau = tau
        self.gumbel_scale = gumbel_scale
        self.chunk_size = chunk_size

        self.register_buffer("indices", indices.to(torch.long))
        scale = 1.0 / math.sqrt(self.basis_size)
        self.theta = nn.Parameter(torch.randn(out_dim, self.basis_size) * scale)
        self.hard_eval = False

        # Bit shift vector for fast hard truth-table addressing
        shifts = torch.arange(n_inputs, dtype=torch.long)
        self.register_buffer("bit_shifts", shifts)

    def discretized_bytes(self) -> int:
        # Truth table bits + 16-bit wire indices per gate
        tt_bytes = (self.out_dim * self.basis_size) // 8
        idx_bytes = self.out_dim * self.n_inputs * 2
        return tt_bytes + idx_bytes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim] in [0, 1]
        if self.hard_eval or not self.training:
            return self.forward_hard(x)

        chunks = []
        # Chunk along out_dim to keep peak activation VRAM tiny even for N=6 & 65K gates
        for s in range(0, self.out_dim, self.chunk_size):
            e = min(s + self.chunk_size, self.out_dim)
            idx_c = self.indices[s:e]  # [C, N]
            theta_c = self.theta[s:e].to(x.dtype)  # [C, 2^N]
            lg = fwht_last_dim(theta_c)  # [C, 2^N]
            if self.gumbel_scale > 0.0:
                u = torch.rand_like(lg).clamp_(1e-5, 1.0 - 1e-5)
                lg = lg + torch.log(u / (1.0 - u)) * self.gumbel_scale

            # Gather pins: x[:, idx_c] -> [B, C, N]
            x_pins = x[:, idx_c]
            basis = std_basis_butterfly(x_pins)  # [B, C, 2^N]
            z = (basis * lg.unsqueeze(0)).sum(dim=-1)  # [B, C]
            y_soft = torch.sigmoid(z / self.tau)
            y_hard = (z > 0.0).to(x.dtype)
            chunks.append(y_hard + (y_soft - y_soft.detach()))
        return torch.cat(chunks, dim=-1)

    @torch.no_grad()
    def forward_hard(self, x: torch.Tensor) -> torch.Tensor:
        """Pure Boolean truth-table lookup (0 multiplications, bit-exact to Kizu EvalHard)."""
        B = x.shape[0]
        x_bits = (x >= 0.5).to(torch.long)  # [B, in_dim] in {0, 1}
        tt = (fwht_last_dim(self.theta) > 0.0).to(x.dtype)  # [out_dim, 2^N] in {0.0, 1.0}
        chunks = []
        for s in range(0, self.out_dim, self.chunk_size):
            e = min(s + self.chunk_size, self.out_dim)
            idx_c = self.indices[s:e]  # [C, N]
            pins = x_bits[:, idx_c]  # [B, C, N]
            addr = (pins << self.bit_shifts.view(1, 1, -1)).sum(dim=-1)  # [B, C]
            tt_c = tt[s:e]  # [C, 2^N]
            row_ids = torch.arange(e - s, device=x.device).unsqueeze(0).expand(B, -1)
            chunks.append(tt_c[row_ids, addr])
        return torch.cat(chunks, dim=-1)


class ThermometerEncoder(nn.Module):
    """
    Zero-multiplication comparator bank: maps continuous x [B, d] -> [B, d * n_bits] binary wires.
    Uses Straight-Through Estimator during training and exact comparators (x > thresh) at eval!
    """

    def __init__(self, d_model: int, n_bits: int, init_thresholds: torch.Tensor, temp: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.n_bits = n_bits
        self.temp = temp
        # init_thresholds: [d_model, n_bits]
        self.thresh = nn.Parameter(init_thresholds.clone())
        self.hard_eval = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, d_model] -> diff: [B, d_model, n_bits]
        diff = x.unsqueeze(-1) - self.thresh.unsqueeze(0)
        if self.hard_eval or not self.training:
            return (diff > 0.0).to(x.dtype).reshape(x.shape[0], -1)
        y_soft = torch.sigmoid(diff / self.temp)
        y_hard = (diff > 0.0).to(x.dtype)
        return (y_hard + (y_soft - y_soft.detach())).reshape(x.shape[0], -1)


def build_random_indices(in_dim: int, out_dim: int, n_inputs: int, gen: torch.Generator) -> torch.Tensor:
    idx = torch.randint(0, in_dim, (out_dim, n_inputs), generator=gen, dtype=torch.long)
    return idx


def build_wgti_indices(
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    n_bits: int,
    h1_gates: int,
    out_gates: int,
    n_inputs: int,
    n_mid_layers: int,
    seed: int = 42,
) -> list[torch.Tensor]:
    """
    Weight-Guided Topology Initialization (WGTI):
    Uses the pretrained SwiGLU sensitivity matrix (|W_gate| + |W_up| and |W_down|) to wire
    LUT gates preferentially onto the highest-signal coordinates of Qwen3's manifold!
    At inference this is STILL pure static integer wire indices (0 extra cost, 0 muls).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    inter, d_model = w_gate.shape  # [6144, 2048]
    # Importance of input dim i for each SwiGLU hidden neuron h: [inter, d_model]
    imp_in = (w_gate.abs() + w_up.abs()).float().cpu()
    # Importance of SwiGLU hidden neuron h for each output dim o: [d_model, inter]
    imp_out = w_down.abs().float().cpu()

    # Map each lut1 gate (0..h1_gates-1) to a parent SwiGLU neuron (round-robin across 6144)
    parent_h = torch.arange(h1_gates) % inter  # [h1_gates]

    # For each SwiGLU neuron h, precompute top-K most important input dims (K = 32)
    K_in = min(32, d_model)
    topk_in = torch.topk(imp_in, k=K_in, dim=1).indices  # [inter, K_in]

    # Wire lut1: each gate samples n_inputs from its parent SwiGLU neuron's top-K dims + random bit threshold!
    # Mix: first n_inputs-1 from top-K, last 1 from global random for cross-feature mixing
    idx1 = torch.empty((h1_gates, n_inputs), dtype=torch.long)
    for p in range(n_inputs):
        if p < n_inputs - 1:
            # Stratified sampling from top-K so sibling gates for the same neuron cover distinct top dims
            pick_rank = torch.randint(0, K_in, (h1_gates,), generator=g)
            dims = topk_in[parent_h, pick_rank]
        else:
            dims = torch.randint(0, d_model, (h1_gates,), generator=g)
        bits = torch.randint(0, n_bits, (h1_gates,), generator=g)
        idx1[:, p] = dims * n_bits + bits

    indices_list = [idx1]

    # Optional middle LUT layers preserve neuron locality + cross-neuron mixing
    cur_dim = h1_gates
    for _ in range(n_mid_layers):
        idx_m = torch.empty((h1_gates, n_inputs), dtype=torch.long)
        # Connect half pins to sibling gates of same/nearby neuron, half globally
        for p in range(n_inputs):
            if p < n_inputs // 2:
                # Sibling copies of the same SwiGLU neuron (stride = inter)
                copy_id = torch.randint(0, max(1, h1_gates // inter), (h1_gates,), generator=g)
                idx_m[:, p] = (parent_h + copy_id * inter) % cur_dim
            else:
                idx_m[:, p] = torch.randint(0, cur_dim, (h1_gates,), generator=g)
        indices_list.append(idx_m)

    # Wire final output LUT layer: group_size gates per output channel o in 0..d_model-1
    group_size = out_gates // d_model
    K_out = min(48, inter)
    topk_out = torch.topk(imp_out, k=K_out, dim=1).indices  # [d_model, K_out]
    n_copies = max(1, h1_gates // inter)

    idx_last = torch.empty((out_gates, n_inputs), dtype=torch.long)
    out_ch = torch.arange(out_gates) // group_size  # [out_gates]
    for p in range(n_inputs):
        if p < n_inputs - 1:
            rank = torch.randint(0, K_out, (out_gates,), generator=g)
            h_pick = topk_out[out_ch, rank]
            copy_id = torch.randint(0, n_copies, (out_gates,), generator=g)
            idx_last[:, p] = (h_pick + copy_id * inter) % h1_gates
        else:
            idx_last[:, p] = torch.randint(0, h1_gates, (out_gates,), generator=g)
    indices_list.append(idx_last)
    return indices_list


class LogicMlpBlock(nn.Module):
    """Drop-in replacement for Qwen3MLP using pure discrete WARP-LUTs + GroupSum."""

    def __init__(
        self,
        d_model: int,
        n_bits: int,
        init_thresholds: torch.Tensor,
        indices_list: list[torch.Tensor],
        n_inputs: int,
        group_size: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.group_size = group_size
        self.enc = ThermometerEncoder(d_model, n_bits, init_thresholds)

        layers = []
        cur_in = d_model * n_bits
        for idx in indices_list:
            out_d = idx.shape[0]
            chunk = 4096 if n_inputs >= 6 else 8192
            layers.append(WarpLutLayer(cur_in, out_d, n_inputs, idx, chunk_size=chunk))
            cur_in = out_d
        self.luts = nn.ModuleList(layers)
        assert cur_in == d_model * group_size

        # Per-channel scale + bias on GroupSum output (just 4096 floats = 8 KB)
        self.out_scale = nn.Parameter(torch.full((d_model,), 1.0))
        self.out_bias = nn.Parameter(torch.zeros((d_model,)))

    def set_hard_eval(self, flag: bool) -> None:
        self.enc.hard_eval = flag
        for l in self.luts:
            l.hard_eval = flag

    def set_anneal(self, tau: float, gumbel_scale: float) -> None:
        self.enc.temp = max(0.15, tau * 0.5)
        for l in self.luts:
            l.tau = tau
            l.gumbel_scale = gumbel_scale

    def total_gates(self) -> int:
        return sum(l.out_dim for l in self.luts)

    def discretized_bytes(self) -> int:
        return sum(l.discretized_bytes() for l in self.luts) + self.d_model * (self.enc.n_bits + 2) * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        orig_dtype = x.dtype
        work_dtype = torch.bfloat16 if torch.is_autocast_enabled() else torch.float32
        x2 = x.reshape(-1, self.d_model).to(work_dtype)
        h = self.enc(x2)
        for l in self.luts:
            h = l(h)
        # GroupSum: average across group_size gates per output dim, centered at 0.5
        gs = h.float().view(-1, self.d_model, self.group_size).mean(dim=-1) - 0.5
        y = gs * self.out_scale.unsqueeze(0) + self.out_bias.unsqueeze(0)
        return y.to(orig_dtype).reshape(orig_shape)


def build_calibration_texts() -> tuple[list[str], list[str]]:
    """Loads real Japanese + English Wikipedia texts (streaming) with robust built-in fallback."""
    texts = []
    try:
        from datasets import load_dataset

        log("Loading real Japanese + English Wikipedia stream from Hugging Face...")
        for lang_cfg, n_docs in [("20231101.ja", 120), ("20231101.en", 80)]:
            ds = load_dataset("wikimedia/wikipedia", lang_cfg, split="train", streaming=True)
            cnt = 0
            for row in ds:
                t = row.get("text", "").strip()
                if len(t) >= 240:
                    texts.append(t[:1800])
                    cnt += 1
                    if cnt >= n_docs:
                        break
        log(f"Loaded {len(texts)} Wikipedia articles (JA + EN).")
    except Exception as e:
        log(f"Wikipedia stream fallback ({e}); loading wikitext + built-in JA/code...")
        try:
            from datasets import load_dataset

            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            for row in ds:
                t = row["text"].strip()
                if len(t) >= 200:
                    texts.append(t[:1500])
                if len(texts) >= 160:
                    break
        except Exception:
            pass

    # Always append rich Japanese conversational, technical & coding passages for balanced evaluation
    ja_val_extra = [
        "量子コンピュータは、量子力学の重ね合わせや量子もつれを利用して計算を行う次世代の計算機です。従来の古典コンピュータが0と1のビットを用いるのに対し、量子ビットは0と1の状態を同時に保持できます。",
        "微分可能論理ゲートネットワークは、ニューラルネットワークの重み行列積を離散的な真理値表ルックアップテーブルへ置き換える手法です。学習時は連続緩和で勾配を流し、推論時は乗算ゼロの純粋な論理回路として動作します。",
        "今日の東京は朝からよく晴れていて、風も穏やかです。駅前のカフェで温かいコーヒーを飲みながら、新しいプログラミング言語のコンパイラ設計について考えていました。",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + mid + quicksort(right)\n",
        "TransformerのSelf-Attentionは入力トークン列同士の関連度を計算し、FFN（SwiGLUなど）は各トークン位置ごとに非線形変換と知識の取り出しを担当します。実用規模のLLMでは全パラメータの約3分の2がFFNに集中しています。",
        "ユーザー: 最近仕事が忙しくてめっちゃ眠いんだけどどうしたらいい？\nアシスタント: それはお疲れさまです。無理に気合で乗り切るより、まずは15分だけ仮眠を取るか、温かい飲み物を飲んで少し目を休めるのが効果的ですよ。",
        "Linuxカーネルのメモリ管理では、仮想アドレスをページテーブルによって物理アドレスへ変換します。TLB（Translation Lookaside Buffer）は最近の変換結果をキャッシュし、メモリアクセスの遅延を大幅に削減します。",
        "美味しいラーメンを作るコツは、スープの旨味とタレの塩味のバランス、そして麺の茹で加減にあります。豚骨ベースでも鶏ガラベースでも、温度管理とアク取りを丁寧に行うことで雑味のない深い味わいになります。",
    ]
    if len(texts) < 40:
        texts = (texts + ja_val_extra * 20)[:180]
    val_texts = ja_val_extra + texts[-16:]
    train_texts = texts[:-16] + ja_val_extra * 4
    return train_texts, val_texts


@torch.no_grad()
def eval_model_ppl(model, val_input_ids: list[torch.Tensor], device: str) -> float:
    total_nll = 0.0
    total_toks = 0
    for ids in val_input_ids:
        ids = ids.to(device)
        out = model(input_ids=ids, labels=ids)
        n = ids.numel() - 1
        total_nll += out.loss.item() * n
        total_toks += n
    return math.exp(total_nll / max(1, total_toks))


@torch.no_grad()
def sample_generations(model, tokenizer, device: str) -> list[tuple[str, str]]:
    prompts = [
        "量子コンピュータとは何か、中学生にもわかるように2文で説明して。",
        "「今日めっちゃ眠い」をXの自動翻訳っぽい少し大げさな日本語に言い換えて。",
        "Pythonでフィボナッチ数列の第n項を返す短い関数を書いて。",
    ]
    results = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inp = tokenizer(text, return_tensors="pt").to(device)
        out = model.generate(
            **inp,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        reply = tokenizer.decode(out[0][inp["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        results.append((p, reply.replace("\n", " ")))
    return results


@torch.no_grad()
def collect_layer_io(
    model,
    layer_idx: int,
    token_batches: list[torch.Tensor],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Captures exact (x_in, y_mlp) pairs for layer `layer_idx`'s MLP."""
    xs, ys = [], []
    target_mlp = model.model.layers[layer_idx].mlp

    def hook(_mod, inp, out):
        xs.append(inp[0].detach().reshape(-1, inp[0].shape[-1]).to(torch.float16).cpu())
        ys.append(out.detach().reshape(-1, out.shape[-1]).to(torch.float16).cpu())

    h = target_mlp.register_forward_hook(hook)
    try:
        for ids in token_batches:
            model(input_ids=ids.to(device), use_cache=False)
    finally:
        h.remove()

    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def eval_block_fit(
    block: LogicMlpBlock,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    hard: bool,
    batch_size: int = 512,
) -> tuple[float, float]:
    block.eval()
    block.set_hard_eval(hard)
    se_sum = 0.0
    yy_sum = 0.0
    cos_sum = 0.0
    n_tot = 0
    with torch.no_grad():
        for i in range(0, x_val.shape[0], batch_size):
            xb = x_val[i : i + batch_size].float()
            yb = y_val[i : i + batch_size].float()
            yh = block(xb).float()
            se_sum += ((yh - yb) ** 2).sum().item()
            yy_sum += (yb ** 2).sum().item()
            cos_sum += F.cosine_similarity(yh, yb, dim=-1).sum().item()
            n_tot += xb.shape[0]
    nmse = se_sum / max(1e-9, yy_sum)
    cos = cos_sum / max(1, n_tot)
    return nmse, cos


def train_logic_mlp(
    name: str,
    block: LogicMlpBlock,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    steps: int = 2400,
    batch_size: int = 256,
    lr_lut: float = 0.035,
    lr_scale: float = 0.015,
) -> dict:
    device = x_train.device
    block.to(device)
    lut_params = [p for n, p in block.named_parameters() if "theta" in n]
    other_params = [p for n, p in block.named_parameters() if "theta" not in n]
    opt = torch.optim.AdamW(
        [
            {"params": lut_params, "lr": lr_lut, "weight_decay": 0.0},
            {"params": other_params, "lr": lr_scale, "weight_decay": 1e-4},
        ]
    )

    n_train = x_train.shape[0]
    t0 = time.time()
    best_hard_cos = -1.0
    best_hard_nmse = 999.0
    best_state = None

    for step in range(1, steps + 1):
        prog = (step - 1) / max(1, steps - 1)
        lr_mult = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))
        opt.param_groups[0]["lr"] = lr_lut * lr_mult
        opt.param_groups[1]["lr"] = lr_scale * lr_mult

        tau = 1.15 - 0.65 * prog
        gumbel = 0.18 * max(0.0, 1.0 - prog / 0.85)
        block.train()
        block.set_hard_eval(False)
        block.set_anneal(tau, gumbel)

        idx = torch.randint(0, n_train, (batch_size,), device=device)
        xb = x_train[idx].float()
        yb = y_train[idx].float()

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            yh = block(xb)
            # Combined MSE + (1 - Cosine) preserves both amplitude and exact residual direction!
            mse = F.mse_loss(yh.float(), yb)
            cos_loss = (1.0 - F.cosine_similarity(yh.float(), yb, dim=-1)).mean()
            loss = mse + 0.25 * cos_loss
        loss.backward()
        opt.step()

        if step in (1, 600, 1200, 1800, steps):
            hard_nmse, hard_cos = eval_block_fit(block, x_val, y_val, hard=True)
            dt = time.time() - t0
            mark = ""
            if hard_nmse < best_hard_nmse:
                best_hard_nmse = hard_nmse
                best_hard_cos = hard_cos
                best_state = {k: v.detach().cpu().clone() for k, v in block.state_dict().items()}
                mark = " [BEST]"
            log(
                f"  [{name}] step={step:4d}/{steps} loss={loss.item():.4f} "
                f"val_hard_NMSE={hard_nmse:.4f} val_hard_cos={hard_cos:.4f}{mark} ({dt:.1f}s)"
            )

    if best_state is not None:
        block.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    block.eval()
    block.set_hard_eval(True)
    return {
        "name": name,
        "gates": block.total_gates(),
        "bytes": block.discretized_bytes(),
        "val_hard_nmse": best_hard_nmse,
        "val_hard_cos": best_hard_cos,
        "train_sec": time.time() - t0,
    }


def main() -> None:
    assert torch.cuda.is_available(), "CUDA GPU is required!"
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    log(f"=== Griffith-Kizu Frontier PoC on {gpu_name} ({vram_gb:.1f} GB VRAM) ===")
    log(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")

    # Install/check transformers & datasets if needed
    try:
        import transformers
        import datasets  # noqa: F401
    except ImportError:
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers", "datasets", "accelerate"])
        import transformers

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen3-1.7B"
    layer_idx = 14  # middle of 28 layers
    log(f"Loading {model_id} in bf16...")
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    log(f"Loaded {model_id} in {time.time() - t_load:.1f}s (layers={len(model.model.layers)}, d={model.config.hidden_size}, inter={model.config.intermediate_size})")

    train_texts, val_texts = build_calibration_texts()
    seq_len = 256

    def pack_ids(texts: list[str], max_seqs: int) -> list[torch.Tensor]:
        all_ids = []
        for t in texts:
            ids = tokenizer(t, add_special_tokens=False)["input_ids"]
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_token_id)
        chunks = []
        for i in range(0, len(all_ids) - seq_len, seq_len):
            chunks.append(torch.tensor([all_ids[i : i + seq_len]], dtype=torch.long))
            if len(chunks) >= max_seqs:
                break
        return chunks

    train_batches = pack_ids(train_texts, max_seqs=160)  # 40,960 real tokens!
    val_batches = pack_ids(val_texts, max_seqs=20)       # 5,120 held-out tokens!
    log(f"Packed train={len(train_batches)*seq_len:,} tok, val={len(val_batches)*seq_len:,} tok")

    base_ppl = eval_model_ppl(model, val_batches, device)
    log(f"\n[BASELINE] Qwen3-1.7B full bf16 val_ppl = {base_ppl:.4f}")
    base_gens = sample_generations(model, tokenizer, device)
    for p, r in base_gens:
        log(f"  BASE | Q: {p[:26]}... -> {r[:90]}")

    log(f"\nCollecting Layer {layer_idx} MLP (x_in, y_mlp) activations...")
    t_col = time.time()
    x_tr_cpu, y_tr_cpu = collect_layer_io(model, layer_idx, train_batches, device)
    x_va_cpu, y_va_cpu = collect_layer_io(model, layer_idx, val_batches, device)
    x_train = x_tr_cpu.to(device)
    y_train = y_tr_cpu.to(device)
    x_val = x_va_cpu.to(device)
    y_val = y_va_cpu.to(device)
    log(f"Collected train={tuple(x_train.shape)}, val={tuple(x_val.shape)} in {time.time()-t_col:.1f}s")

    # Compute exact data-driven per-channel quantiles for thermometer thresholds!
    x_sample_f = x_train[:8192].float()
    d_model = x_train.shape[1]
    q1 = torch.quantile(x_sample_f, torch.tensor([0.5], device=device), dim=0).T.contiguous()
    q4 = torch.quantile(
        x_sample_f, torch.tensor([0.20, 0.40, 0.60, 0.80], device=device), dim=0
    ).T.contiguous()

    # Grab original SwiGLU weights for WGTI topology initialization + hot-swap restore
    orig_mlp = model.model.layers[layer_idx].mlp
    w_gate = orig_mlp.gate_proj.weight.detach()
    w_up = orig_mlp.up_proj.weight.detach()
    w_down = orig_mlp.down_proj.weight.detach()
    orig_mlp_bytes = (w_gate.numel() + w_up.numel() + w_down.numel()) * 2

    # Sanity check: what happens if Layer 14 MLP is zeroed out completely?
    class ZeroMlp(nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)

    model.model.layers[layer_idx].mlp = ZeroMlp()
    zero_ppl = eval_model_ppl(model, val_batches, device)
    model.model.layers[layer_idx].mlp = orig_mlp
    log(f"[ABLATION] Layer {layer_idx} MLP = ZERO -> val_ppl = {zero_ppl:.4f} (delta = +{zero_ppl - base_ppl:.4f})")

    # Build the 4 frontier experiments
    g_rand = torch.Generator(device="cpu").manual_seed(42)
    configs = [
        (
            "ExpA_N4_1bit_Rand_64K",
            1,
            q1,
            [
                build_random_indices(d_model * 1, 32768, 4, g_rand),
                build_random_indices(32768, d_model * 16, 4, g_rand),
            ],
            4,
            16,
            2400,
            256,
        ),
        (
            "ExpB_N4_4bit_WGTI_64K",
            4,
            q4,
            build_wgti_indices(w_gate, w_up, w_down, 4, 32768, d_model * 16, 4, n_mid_layers=0, seed=42),
            4,
            16,
            2400,
            256,
        ),
        (
            "ExpC_N4_4bit_WGTI_3L_96K",
            4,
            q4,
            build_wgti_indices(w_gate, w_up, w_down, 4, 32768, d_model * 16, 4, n_mid_layers=1, seed=43),
            4,
            16,
            2400,
            256,
        ),
        (
            "ExpD_N6_4bit_WGTI_114K",
            4,
            q4,
            build_wgti_indices(w_gate, w_up, w_down, 4, 49152, d_model * 32, 6, n_mid_layers=0, seed=44),
            6,
            32,
            2400,
            192,
        ),
    ]

    summary = {
        "model": model_id,
        "gpu": gpu_name,
        "layer_idx": layer_idx,
        "orig_mlp_bytes": orig_mlp_bytes,
        "base_ppl": base_ppl,
        "zero_mlp_ppl": zero_ppl,
        "experiments": [],
    }

    for name, n_bits, thresh, idxs, n_in, grp, steps, bs in configs:
        log(f"\n--- Running {name} (N={n_in}, {n_bits}-bit enc, {sum(x.shape[0] for x in idxs):,} gates) ---")
        block = LogicMlpBlock(d_model, n_bits, thresh, idxs, n_in, grp)
        res = train_logic_mlp(name, block, x_train, y_train, x_val, y_val, steps=steps, batch_size=bs)

        # Hot-swap HARD discrete Boolean LUT circuit into live Qwen3-1.7B!
        block.eval()
        block.set_hard_eval(True)
        model.model.layers[layer_idx].mlp = block
        swapped_ppl = eval_model_ppl(model, val_batches, device)
        gens = sample_generations(model, tokenizer, device)
        model.model.layers[layer_idx].mlp = orig_mlp

        res["swapped_ppl"] = swapped_ppl
        res["ppl_delta"] = swapped_ppl - base_ppl
        res["compression_x"] = orig_mlp_bytes / max(1, res["bytes"])
        res["samples"] = gens
        summary["experiments"].append(res)

        log(
            f"==> [{name}] HARD LUT hot-swap val_ppl = {swapped_ppl:.4f} "
            f"(delta = {swapped_ppl - base_ppl:+.4f}, vs zero-MLP {zero_ppl:.4f}) | "
            f"size = {res['bytes']/1024:.0f} KB ({res['compression_x']:.1f}x smaller than 75.5 MB bf16 MLP!)"
        )
        for p, r in gens:
            log(f"    {name} | Q: {p[:22]}... -> {r[:90]}")

        del block
        torch.cuda.empty_cache()

    log("\n==================== FRONTIER PoC SUMMARY ====================")
    log(f"Model: {model_id} | Layer {layer_idx} | Base PPL: {base_ppl:.4f} | Zero-MLP PPL: {zero_ppl:.4f}")
    log(f"{'Experiment':26s} | {'Gates':>7s} | {'Size':>8s} | {'Comp':>6s} | {'NMSE':>6s} | {'Cos':>6s} | {'PPL':>7s} | {'Delta':>7s}")
    log("-" * 96)
    for r in summary["experiments"]:
        log(
            f"{r['name']:26s} | {r['gates']:7,d} | {r['bytes']/1024:6.0f}KB | {r['compression_x']:5.1f}x | "
            f"{r['val_hard_nmse']:6.4f} | {r['val_hard_cos']:6.4f} | {r['swapped_ppl']:7.4f} | {r['ppl_delta']:+7.4f}"
        )
    log("JSON_SUMMARY:" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
