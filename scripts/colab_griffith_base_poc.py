#!/usr/bin/env python3
"""
Griffith-Kizu Route 2 Frontier PoC:
FWHT-Mixed + End-to-End Co-Adapted Looped LogicLadderGPT ("Griffith-JP-Base")

Architecture (designed for pure-Kizu .glgpt v3 export & browser WASM compatibility):
  - Byte-level BPE (vocab=4096, raw bytes 0..255 + 3840 merges, 1:1 Kizu compatible)
  - d_model = 512 (exact 2^9 -> 9-stage zero-multiplication Fast Walsh-Hadamard Transform)
  - n_heads = 8 (dk = 64), n_kv_heads = 2 (GQA 4:1), RoPE, tied tok_emb/lm_head
  - Looped Transformer (Inkling 2 Mini inspired): 4 unique layers x 2 loops = 8 effective layers
  - FFN (H-LUT-FFN, ZERO matrix multiplications!):
      u = FWHT(RMSNorm(x)) / sqrt(512)              [0 muls: add/sub butterfly]
      b = 2-bit Thermometer(u) -> 1,024 binary wires [0 muls: comparators]
      h1 = WarpLutN4(1024 -> 4096)                   [0 muls: N=4 Boolean LUTs]
      h2 = WarpLutN4(4096 -> 8192)                   [0 muls: N=4 Boolean LUTs]
      g  = GroupSum(h2, group=16) - 0.5              [0 muls: popcount/sum]
      y  = FWHT(g) / sqrt(512) * out_scale           [0 muls: add/sub butterfly]
  - Total unique discrete N=4 LUTs: 49,152 (98,304 effective LUT evals/tok across 2 loops)
  - Discretized FFN size: 492 KB (33.6x smaller than 16.5 MB bf16 SwiGLU)
  - Total exported model size: ~9.9 MB (f16) / ~5.3 MB (int8) -> fits in browser WASM!

3-Phase Proof in one fast run:
  Phase 1: Train Looped FP-Base Teacher (SwiGLU) on real Japanese Wikipedia (~4M+ tok)
  Phase 2: Warm-start & KL-distill H-LUT-Base (0-mul FWHT + 49K N=4 WARP-LUTs) End-to-End
           -> Measure HARD DISCRETIZED BOOLEAN Val PPL vs FP-Base Teacher!
  Phase 3: 2-minute downstream SFT of H-LUT-Base on jpllm_corpus.txt ([会話]/[X訳]/[関西]/[津軽])
           -> Verify general Japanese base transfers immediately to specialized chat & unseen topics!
"""

import json
import math
import os
import struct
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def log(msg: str) -> None:
    print(msg, flush=True)


# ==============================================================================
# 1. Zero-Multiplication Fast Walsh-Hadamard Transform (exact Kizu mirror)
# ==============================================================================
def fwht_last_dim(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized FWHT along last dim (power of 2). Pure add/sub butterflies."""
    n = x.shape[-1]
    h = x
    step = 1
    while step < n:
        h = h.reshape(*x.shape[:-1], -1, 2, step)
        u = h[..., 0, :]
        v = h[..., 1, :]
        h = torch.stack((u + v, u - v), dim=-2).reshape(*x.shape)
        step *= 2
    return h


def fwht_ortho(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal FWHT (divides by sqrt(d)). Exact self-inverse!"""
    return fwht_last_dim(x) * (1.0 / math.sqrt(x.shape[-1]))


# ==============================================================================
# 2. Fast Fused N=4 WARP-LUT (Horner-factored Walsh-Hadamard -> ZERO [B,C,16] HBM!)
# ==============================================================================
def _warp_n4_poly(
    x: torch.Tensor,
    idx0: torch.Tensor,
    idx1: torch.Tensor,
    idx2: torch.Tensor,
    idx3: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """
    Computes exact Walsh-Hadamard N=4 polynomial z = sum_{k=0}^{15} theta_k * B_wh_k(x_pins)
    using 4-level Horner factoring in registers (15 FMAs, ZERO [B, C, 16] tensor!).
    Pure tensor signature => torch.compile compiles ONCE with zero guard recompilations!
    """
    s0 = 1.0 - 2.0 * x.index_select(1, idx0)
    s1 = 1.0 - 2.0 * x.index_select(1, idx1)
    s2 = 1.0 - 2.0 * x.index_select(1, idx2)
    s3 = 1.0 - 2.0 * x.index_select(1, idx3)

    t = theta.to(x.dtype)  # [C, 16]
    a0 = t[:, 0] + s0 * t[:, 1]
    a1 = t[:, 2] + s0 * t[:, 3]
    a2 = t[:, 4] + s0 * t[:, 5]
    a3 = t[:, 6] + s0 * t[:, 7]
    a4 = t[:, 8] + s0 * t[:, 9]
    a5 = t[:, 10] + s0 * t[:, 11]
    a6 = t[:, 12] + s0 * t[:, 13]
    a7 = t[:, 14] + s0 * t[:, 15]

    b0 = a0 + s1 * a1
    b1 = a2 + s1 * a3
    b2 = a4 + s1 * a5
    b3 = a6 + s1 * a7

    c0 = b0 + s2 * b1
    c1 = b2 + s2 * b3
    return c0 + s3 * c1


try:
    _warp_n4_poly_compiled = torch.compile(_warp_n4_poly)
except Exception:
    _warp_n4_poly_compiled = _warp_n4_poly


class FastWarpLutN4(nn.Module):
    """
    High-speed N=4 WARP-LUT layer (100% bit-exact at EvalHard to Kizu src/lut/layer.kizu).
    """

    def __init__(self, in_dim: int, out_dim: int, seed: int = 42, use_compile: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_compile = use_compile
        self.tau = 1.0
        self.gumbel_scale = 0.18
        self.hard_eval = False

        g = torch.Generator(device="cpu").manual_seed(seed)
        # Ensure distinct pins per gate just like Kizu build_unique_connections
        idx = torch.empty((out_dim, 4), dtype=torch.long)
        for p in range(4):
            idx[:, p] = torch.randint(0, in_dim, (out_dim,), generator=g)
        # Fix collisions cheaply
        for p in range(1, 4):
            for prev in range(p):
                coll = idx[:, p] == idx[:, prev]
                if coll.any():
                    idx[coll, p] = (idx[coll, p] + p * 7 + 1) % in_dim

        self.register_buffer("idx0", idx[:, 0].contiguous())
        self.register_buffer("idx1", idx[:, 1].contiguous())
        self.register_buffer("idx2", idx[:, 2].contiguous())
        self.register_buffer("idx3", idx[:, 3].contiguous())
        self.theta = nn.Parameter(torch.randn(out_dim, 16) * (1.0 / 4.0))

    def discretized_bytes(self) -> int:
        # 2 bytes (16-bit truth table) + 4 * 2 bytes (u16 wire indices) = 10 B/gate
        return self.out_dim * 10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hard_eval or not self.training:
            return self.forward_hard(x)
        fn = _warp_n4_poly_compiled if self.use_compile else _warp_n4_poly
        z = fn(x, self.idx0, self.idx1, self.idx2, self.idx3, self.theta)
        if self.gumbel_scale > 0.0:
            u = torch.rand_like(z).clamp_(1e-5, 1.0 - 1e-5)
            z = z + torch.log(u / (1.0 - u)) * self.gumbel_scale
        y_soft = torch.sigmoid(z / self.tau)
        y_hard = (z > 0.0).to(x.dtype)
        return y_hard + (y_soft - y_soft.detach())

    @torch.no_grad()
    def forward_hard(self, x: torch.Tensor) -> torch.Tensor:
        """Pure Boolean 16-bit truth-table lookup (0 multiplications, exact Kizu EvalHard!)."""
        xb = (x >= 0.5).to(torch.long)
        addr = (
            xb.index_select(1, self.idx0)
            | (xb.index_select(1, self.idx1) << 1)
            | (xb.index_select(1, self.idx2) << 2)
            | (xb.index_select(1, self.idx3) << 3)
        )
        tt = (fwht_last_dim(self.theta.float()) > 0.0).to(x.dtype)  # [out_dim, 16]
        row = torch.arange(self.out_dim, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        return tt[row, addr]


class HLutFfn(nn.Module):
    """
    Zero-Multiplication FWHT-Mixed WARP-LUT FFN:
      FWHT(x) -> 2-bit Thermometer -> WarpLutN4(2d -> h1) -> WarpLutN4(h1 -> d*g) -> GroupSum -> FWHT
    """

    def __init__(self, d_model: int, h1_gates: int = 4096, group_size: int = 16, seed: int = 42):
        super().__init__()
        self.d_model = d_model
        self.group_size = group_size
        self.n_bits = 2
        self.temp = 0.45
        self.hard_eval = False

        # Symmetric 2-bit thresholds initialized at [-0.55, +0.55] on unit-variance FWHT(RMSNorm(x))
        init_th = torch.tensor([-0.55, 0.55]).view(1, 2).repeat(d_model, 1)
        self.thresh = nn.Parameter(init_th)
        self.lut1 = FastWarpLutN4(d_model * 2, h1_gates, seed=seed)
        self.lut2 = FastWarpLutN4(h1_gates, d_model * group_size, seed=seed + 1000)
        self.out_scale = nn.Parameter(torch.full((d_model,), 1.2))

    def set_hard_eval(self, flag: bool) -> None:
        self.hard_eval = flag
        self.lut1.hard_eval = flag
        self.lut2.hard_eval = flag

    def set_anneal(self, tau: float, gumbel_scale: float) -> None:
        self.temp = max(0.18, tau * 0.45)
        self.lut1.tau = tau
        self.lut1.gumbel_scale = gumbel_scale
        self.lut2.tau = tau
        self.lut2.gumbel_scale = gumbel_scale

    def total_gates(self) -> int:
        return self.lut1.out_dim + self.lut2.out_dim

    def discretized_bytes(self) -> int:
        return self.lut1.discretized_bytes() + self.lut2.discretized_bytes() + self.d_model * 6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2 = x.reshape(-1, self.d_model)
        # 1. Global zero-mul Walsh-Hadamard mixing across all d_model channels!
        u = fwht_ortho(x2)
        # 2. Zero-mul 2-bit thermometer comparator bank
        diff = u.unsqueeze(-1) - self.thresh.to(u.dtype).unsqueeze(0)
        if self.hard_eval or not self.training:
            bits = (diff > 0.0).to(u.dtype).reshape(u.shape[0], -1)
        else:
            ys = torch.sigmoid(diff / self.temp)
            yh = (diff > 0.0).to(u.dtype)
            bits = (yh + (ys - ys.detach())).reshape(u.shape[0], -1)
        # 3. Two-layer N=4 discrete WARP-LUT circuit
        h1 = self.lut1(bits)
        h2 = self.lut2(h1)
        # 4. GroupSum + inverse zero-mul FWHT back into residual basis!
        gs = h2.view(-1, self.d_model, self.group_size).mean(dim=-1) - 0.5
        y = fwht_ortho(gs) * self.out_scale.to(gs.dtype).unsqueeze(0)
        return y.reshape(orig_shape)


# ==============================================================================
# 3. Looped Griffith Transformer (GQA + RoPE + SwiGLU or H-LUT-FFN)
# ==============================================================================
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * inv).to(x.dtype) * self.weight.to(x.dtype)


def precompute_rope(seq_len: int, head_dim: int, device: torch.device, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]
    T = x.shape[2]
    c = cos[:T].view(1, 1, T, -1).to(x.dtype)
    s = sin[:T].view(1, 1, T, -1).to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


class GqaAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.rep = n_heads // n_kv_heads
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.rep > 1:
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, D))


class SwiGluFfn(nn.Module):
    def __init__(self, d_model: int, hidden: int = 1344):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class GriffithBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, use_lut_ffn: bool, h1_gates: int, group_size: int, seed: int):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = GqaAttention(d_model, n_heads, n_kv_heads)
        self.ln2 = RMSNorm(d_model)
        self.use_lut_ffn = use_lut_ffn
        if use_lut_ffn:
            self.ffn = HLutFfn(d_model, h1_gates=h1_gates, group_size=group_size, seed=seed)
        else:
            self.ffn = SwiGluFfn(d_model, hidden=1344)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.ffn(self.ln2(x))
        return x


class LoopedGriffithLM(nn.Module):
    """
    Looped Transformer (n_unique_layers x n_loops) with tied embeddings and optional H-LUT-FFN.
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        seq_len: int = 256,
        d_model: int = 512,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        n_unique_layers: int = 4,
        n_loops: int = 2,
        use_lut_ffn: bool = False,
        h1_gates: int = 4096,
        group_size: int = 16,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_unique_layers = n_unique_layers
        self.n_loops = n_loops
        self.use_lut_ffn = use_lut_ffn

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.loop_emb = nn.Parameter(torch.zeros(n_loops, d_model))
        self.blocks = nn.ModuleList(
            [
                GriffithBlock(
                    d_model, n_heads, n_kv_heads, use_lut_ffn, h1_gates, group_size, seed=42 + l * 17
                )
                for l in range(n_unique_layers)
            ]
        )
        self.final_ln = RMSNorm(d_model)
        self.register_buffer("rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)

        # Clean scaled init
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        for blk in self.blocks:
            nn.init.normal_(blk.attn.wo.weight, std=0.02 / math.sqrt(2 * n_unique_layers * n_loops))
            if not use_lut_ffn:
                nn.init.normal_(blk.ffn.w_down.weight, std=0.02 / math.sqrt(2 * n_unique_layers * n_loops))

    def set_hard_eval(self, flag: bool) -> None:
        if self.use_lut_ffn:
            for blk in self.blocks:
                blk.ffn.set_hard_eval(flag)

    def set_anneal(self, tau: float, gumbel_scale: float) -> None:
        if self.use_lut_ffn:
            for blk in self.blocks:
                blk.ffn.set_anneal(tau, gumbel_scale)

    def total_lut_gates(self) -> int:
        if not self.use_lut_ffn:
            return 0
        return sum(blk.ffn.total_gates() for blk in self.blocks)

    def estimated_export_bytes(self) -> tuple[int, int]:
        """Returns (f16_bytes, int8_bytes) for exported compact binary."""
        emb_p = self.tok_emb.weight.numel()
        attn_p = sum(
            p.numel()
            for blk in self.blocks
            for p in (blk.attn.wq.weight, blk.attn.wk.weight, blk.attn.wv.weight, blk.attn.wo.weight)
        )
        norm_p = sum(p.numel() for n, p in self.named_parameters() if "ln" in n or "loop_emb" in n)
        if self.use_lut_ffn:
            ffn_b = sum(blk.ffn.discretized_bytes() for blk in self.blocks)
            f16_b = (emb_p + attn_p + norm_p) * 2 + ffn_b
            i8_b = (emb_p + attn_p) * 1 + norm_p * 2 + ffn_b
            return f16_b, i8_b
        else:
            ffn_p = sum(p.numel() for blk in self.blocks for p in blk.ffn.parameters())
            tot = emb_p + attn_p + norm_p + ffn_p
            return tot * 2, tot

    def warm_start_from_fp(self, fp_model: "LoopedGriffithLM") -> None:
        """Copies embeddings, loop_emb, norms, and GQA attention from FP Teacher."""
        with torch.no_grad():
            self.tok_emb.weight.copy_(fp_model.tok_emb.weight)
            self.loop_emb.copy_(fp_model.loop_emb)
            self.final_ln.weight.copy_(fp_model.final_ln.weight)
            for sblk, tblk in zip(self.blocks, fp_model.blocks):
                sblk.ln1.weight.copy_(tblk.ln1.weight)
                sblk.ln2.weight.copy_(tblk.ln2.weight)
                sblk.attn.load_state_dict(tblk.attn.state_dict())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        if self.rope_cos.numel() == 0 or self.rope_cos.shape[0] < T or self.rope_cos.device != idx.device:
            cos, sin = precompute_rope(max(T, self.seq_len), self.d_model // self.n_heads, idx.device)
            self.rope_cos = cos
            self.rope_sin = sin

        x = self.tok_emb(idx)
        for lp in range(self.n_loops):
            x = x + self.loop_emb[lp].to(x.dtype).view(1, 1, -1)
            for blk in self.blocks:
                x = blk(x, self.rope_cos, self.rope_sin)
        x = self.final_ln(x)
        # Tied LM head
        return F.linear(x, self.tok_emb.weight.to(x.dtype))


# ==============================================================================
# 4. Byte-Level BPE Tokenizer (vocab=4096, UTF-8 safe decoding)
# ==============================================================================
def train_japanese_bpe(texts: list[str], vocab_size: int = 4096):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<|eos|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def load_japanese_pretrain_texts(max_wiki_docs: int = 3000) -> tuple[list[str], list[str]]:
    from datasets import load_dataset

    log(f"Streaming {max_wiki_docs} Japanese Wikipedia articles from Hugging Face...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
    docs = []
    for row in ds:
        t = row.get("text", "").strip()
        if len(t) >= 300:
            docs.append(t[:4000])
            if len(docs) >= max_wiki_docs:
                break
    log(f"Loaded {len(docs)} Japanese Wikipedia docs ({sum(len(d) for d in docs)/1e6:.2f}M chars).")
    # Hold out last 100 docs for validation
    return docs[:-100], docs[-100:]


@torch.no_grad()
def eval_lm_ppl(model: LoopedGriffithLM, val_tokens: torch.Tensor, seq_len: int, batch_size: int = 16, n_batches: int = 32, hard: bool = True) -> float:
    model.eval()
    model.set_hard_eval(hard)
    device = next(model.parameters()).device
    total_loss = 0.0
    stride = (len(val_tokens) - seq_len - 1) // max(1, n_batches * batch_size)
    stride = max(1, stride)
    pos = 0
    count = 0
    for _ in range(n_batches):
        xs, ys = [], []
        for _ in range(batch_size):
            if pos + seq_len + 1 >= len(val_tokens):
                break
            chunk = val_tokens[pos : pos + seq_len + 1]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
            pos += stride
        if not xs:
            break
        xb = torch.stack(xs).to(device, dtype=torch.long)
        yb = torch.stack(ys).to(device, dtype=torch.long)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xb)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), yb.reshape(-1))
        total_loss += loss.item()
        count += 1
    return math.exp(total_loss / max(1, count))


@torch.no_grad()
def generate_text(
    model: LoopedGriffithLM,
    tok,
    prompt: str,
    max_new: int = 64,
    temp: float = 0.55,
    top_p: float = 0.88,
    rep_pen: float = 1.18,
    stop_nl: bool = False,
    hard: bool = True,
) -> str:
    model.eval()
    model.set_hard_eval(hard)
    device = next(model.parameters()).device
    eos_id = tok.token_to_id("<|eos|>")
    ids = tok.encode(prompt).ids
    gen = []
    for _ in range(max_new):
        ctx = ids[-model.seq_len :]
        xb = torch.tensor([ctx], device=device, dtype=torch.long)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xb)[0, -1].float()
        # Repetition penalty on recent tokens
        for tid in set(ids[-24:]):
            if logits[tid] > 0:
                logits[tid] /= rep_pen
            else:
                logits[tid] *= rep_pen
        if temp <= 0.01:
            nxt = int(logits.argmax().item())
        else:
            probs = F.softmax(logits / temp, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            cs = torch.cumsum(sp, dim=-1)
            mask = cs - sp > top_p
            sp[mask] = 0.0
            sp /= sp.sum()
            pick = torch.multinomial(sp, 1).item()
            nxt = int(si[pick].item())
        if nxt == eos_id:
            break
        ids.append(nxt)
        gen.append(nxt)
        txt = tok.decode(gen)
        if stop_nl and "\n" in txt:
            return txt.split("\n")[0].strip()
    return tok.decode(gen).strip()


def train_phase(
    name: str,
    model: LoopedGriffithLM,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    steps: int,
    batch_size: int = 16,
    lr_fp: float = 1.2e-3,
    lr_lut: float = 2.5e-2,
    teacher: LoopedGriffithLM | None = None,
    kd_alpha: float = 0.55,
    kd_temp: float = 2.0,
) -> dict:
    device = next(model.parameters()).device
    seq_len = model.seq_len
    lut_params = [p for n, p in model.named_parameters() if "theta" in n]
    fp_params = [p for n, p in model.named_parameters() if "theta" not in n]
    groups = [{"params": fp_params, "lr": lr_fp, "weight_decay": 0.02}]
    if lut_params:
        groups.append({"params": lut_params, "lr": lr_lut, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))

    max_start = len(train_tokens) - seq_len - 2
    t0 = time.time()
    best_ppl = 1e9
    best_step = 0
    best_state = None
    eval_every = max(250, steps // 5)

    for step in range(1, steps + 1):
        prog = (step - 1) / max(1, steps - 1)
        warmup = min(1.0, step / max(1, min(100, steps // 10)))
        lr_mult = warmup * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        opt.param_groups[0]["lr"] = lr_fp * lr_mult
        if len(opt.param_groups) > 1:
            opt.param_groups[1]["lr"] = lr_lut * lr_mult

        if model.use_lut_ffn:
            tau = 1.15 - 0.55 * prog
            gumbel = 0.20 * max(0.0, 1.0 - prog / 0.85)
            model.set_anneal(tau, gumbel)

        model.train()
        model.set_hard_eval(False)

        starts = torch.randint(0, max_start, (batch_size,))
        offsets = torch.arange(seq_len + 1).unsqueeze(0)
        batch = train_tokens[starts.unsqueeze(1) + offsets].to(device, dtype=torch.long)
        xb, yb = batch[:, :-1], batch[:, 1:]

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xb)
            ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), yb.reshape(-1))
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(xb)
                p_t = F.softmax(t_logits.float() / kd_temp, dim=-1)
                log_p_s = F.log_softmax(logits.float() / kd_temp, dim=-1)
                kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (kd_temp * kd_temp) / seq_len
                # F.kl_div batchmean divides by B, so divide by T too -> per-token KL!
                loss = (1.0 - kd_alpha) * ce + kd_alpha * kl
            else:
                loss = ce

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % eval_every == 0 or step == steps:
            dt = time.time() - t0
            tps = (step * batch_size * seq_len) / max(1e-3, dt)
            hppl = eval_lm_ppl(model, val_tokens, seq_len, batch_size=16, n_batches=32, hard=True)
            mark = ""
            if hppl < best_ppl:
                best_ppl = hppl
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                mark = " [BEST]"
            log(
                f"  [{name}] step={step:4d}/{steps} loss={loss.item():.4f} "
                f"val_hard_ppl={hppl:.2f}{mark} tok/s={tps:,.0f} ({dt:.1f}s)"
            )

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    model.set_hard_eval(True)
    return {
        "name": name,
        "best_hard_ppl": best_ppl,
        "best_step": best_step,
        "train_sec": time.time() - t0,
        "lut_gates": model.total_lut_gates(),
    }


def sft_specialize(
    model: LoopedGriffithLM,
    tok,
    corpus_path: str,
    steps: int = 350,
    batch_size: int = 16,
    lr_fp: float = 3.0e-4,
    lr_lut: float = 1.2e-2,
) -> dict:
    """Fast 1-2 min SFT proving that Griffith-JP-Base adapts cleanly to 4-mode chat/X-translation!"""
    with open(corpus_path, "r", encoding="utf-8") as f:
        raw = f.read()
    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
    eos_id = tok.token_to_id("<|eos|>")

    # Encode each example cleanly with EOS terminator
    train_blocks = blocks[:-200]
    val_blocks = blocks[-200:]

    def pack_blocks(blks: list[str]) -> torch.Tensor:
        arr = []
        for b in blks:
            arr.extend(tok.encode(b).ids)
            arr.append(eos_id)
        return torch.tensor(arr, dtype=torch.int32)

    sft_tr = pack_blocks(train_blocks)
    sft_va = pack_blocks(val_blocks)
    log(f"SFT corpus loaded: {len(blocks)} examples -> train={len(sft_tr):,} tok, val={len(sft_va):,} tok")

    # Measure zero-shot PPL on jpllm val before SFT
    pre_ppl = eval_lm_ppl(model, sft_va, model.seq_len, batch_size=8, n_batches=16, hard=True)
    log(f"  [SFT] Pre-SFT val_hard_ppl on jpllm = {pre_ppl:.2f}")

    res = train_phase(
        "H-LUT-SFT",
        model,
        sft_tr,
        sft_va,
        steps=steps,
        batch_size=batch_size,
        lr_fp=lr_fp,
        lr_lut=lr_lut,
        teacher=None,
    )
    res["pre_sft_ppl"] = pre_ppl
    return res


def main() -> None:
    assert torch.cuda.is_available(), "CUDA GPU required!"
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    log(f"=== Griffith-JP-Base Route 2 Frontier PoC on {gpu_name} ({vram_gb:.1f} GB VRAM) ===")

    # Ensure datasets & tokenizers available
    try:
        import datasets  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets", "tokenizers"])

    # Quick math sanity check: Horner N=4 == FWHT truth table at all 16 binary vertices!
    test_lut = FastWarpLutN4(16, 32, seed=7, use_compile=False).to(device)
    x_bin = torch.randint(0, 2, (64, 16), device=device, dtype=torch.float32)
    test_lut.train()
    test_lut.gumbel_scale = 0.0
    test_lut.tau = 1.0
    y_train_hard = test_lut(x_bin)
    y_eval_hard = test_lut.forward_hard(x_bin)
    diff = (y_train_hard - y_eval_hard).abs().max().item()
    assert diff == 0.0, f"Horner vs EvalHard mismatch: {diff}"
    log("Sanity check PASS: FastWarpLutN4 Horner forward == exact Boolean EvalHard (max_diff = 0.0)!")

    # 1. Load Japanese Wikipedia + train vocab=4096 BPE
    t_data = time.time()
    train_docs, val_docs = load_japanese_pretrain_texts(max_wiki_docs=3200)
    bpe_sample = train_docs[:1500]
    if os.path.exists("/content/jpllm_corpus.txt"):
        with open("/content/jpllm_corpus.txt", "r", encoding="utf-8") as f:
            bpe_sample = bpe_sample + [f.read()]
    tok = train_japanese_bpe(bpe_sample, vocab_size=4096)
    tok.save("/content/griffith_jp_bpe4096.json")
    eos_id = tok.token_to_id("<|eos|>")

    def encode_corpus(docs: list[str]) -> torch.Tensor:
        out = []
        for enc in tok.encode_batch(docs):
            out.extend(enc.ids)
            out.append(eos_id)
        return torch.tensor(out, dtype=torch.int32)

    train_tokens = encode_corpus(train_docs)
    val_tokens = encode_corpus(val_docs)
    log(
        f"BPE 4096 ready in {time.time()-t_data:.1f}s | "
        f"train={len(train_tokens):,} tok, val={len(val_tokens):,} tok"
    )

    # 2. Phase 1: Train FP-Base Teacher (Looped d=512, 4 unique x 2 loops = 8 effective layers)
    log("\n=== PHASE 1: Looped FP-Base Teacher (d=512, 4x2=8 eff layers, SwiGLU) ===")
    fp_model = LoopedGriffithLM(
        vocab_size=4096,
        seq_len=256,
        d_model=512,
        n_heads=8,
        n_kv_heads=2,
        n_unique_layers=4,
        n_loops=2,
        use_lut_ffn=False,
    ).to(device)
    fp_f16_b, _ = fp_model.estimated_export_bytes()
    log(f"FP-Base params={sum(p.numel() for p in fp_model.parameters()):,} (f16={fp_f16_b/1e6:.2f} MB)")
    res_fp = train_phase(
        "FP-Base",
        fp_model,
        train_tokens,
        val_tokens,
        steps=1600,
        batch_size=16,
        lr_fp=1.4e-3,
    )
    log("\n--- FP-Base Japanese Wiki Samples ---")
    fp_samples = []
    for p in ["量子コンピュータとは、", "日本の東京は", "人工知能の歴史において"]:
        g = generate_text(fp_model, tok, p, max_new=56, temp=0.55)
        fp_samples.append((p, g))
        log(f"  FP   | {p} >>> {g}")

    # 3. Phase 2: Train H-LUT-Base (0-mul FWHT + 49,152 N=4 WARP-LUTs) warm-started & KL-distilled!
    log("\n=== PHASE 2: H-LUT-Base Student (0-mul FWHT + 49,152 N=4 WARP-LUTs, 4x2=8 eff layers) ===")
    hlut_model = LoopedGriffithLM(
        vocab_size=4096,
        seq_len=256,
        d_model=512,
        n_heads=8,
        n_kv_heads=2,
        n_unique_layers=4,
        n_loops=2,
        use_lut_ffn=True,
        h1_gates=4096,
        group_size=16,
    ).to(device)
    hlut_model.warm_start_from_fp(fp_model)
    hlut_f16_b, hlut_i8_b = hlut_model.estimated_export_bytes()
    log(
        f"H-LUT-Base unique_gates={hlut_model.total_lut_gates():,} (effective={hlut_model.total_lut_gates()*2:,}/tok) | "
        f"export f16={hlut_f16_b/1e6:.2f} MB, int8={hlut_i8_b/1e6:.2f} MB"
    )
    fp_model.eval()
    for p in fp_model.parameters():
        p.requires_grad_(False)

    res_hlut = train_phase(
        "H-LUT-Base",
        hlut_model,
        train_tokens,
        val_tokens,
        steps=2000,
        batch_size=16,
        lr_fp=5.0e-4,
        lr_lut=2.8e-2,
        teacher=fp_model,
        kd_alpha=0.55,
        kd_temp=2.0,
    )
    torch.save(hlut_model.state_dict(), "/content/griffith_jp_base_hlut.pt")

    log("\n--- H-LUT-Base (HARD Boolean LUTs, 0 FFN Muls!) Japanese Wiki Samples ---")
    hlut_samples = []
    for p in ["量子コンピュータとは、", "日本の東京は", "人工知能の歴史において"]:
        g = generate_text(hlut_model, tok, p, max_new=56, temp=0.55, hard=True)
        hlut_samples.append((p, g))
        log(f"  HLUT | {p} >>> {g}")

    # Free teacher before Phase 3 SFT
    del fp_model
    torch.cuda.empty_cache()

    # 4. Phase 3: Instant Specialization Proof on jpllm_corpus.txt (350 steps!)
    res_sft = None
    sft_samples = []
    if os.path.exists("/content/jpllm_corpus.txt"):
        log("\n=== PHASE 3: 2-Min Downstream Specialization (H-LUT-Base -> 4-mode Japanese Chat) ===")
        res_sft = sft_specialize(hlut_model, tok, "/content/jpllm_corpus.txt", steps=350, batch_size=16)
        torch.save(hlut_model.state_dict(), "/content/griffith_jp_sft_hlut.pt")
        sft_prompts = [
            "[会話]\nU: おはよう\nA: ",
            "[会話]\nU: 量子コンピュータってすごい？\nA: ",
            "[会話]\nU: 徹夜でバグが直らない\nA: ",
            "[会話]\nU: 宇宙旅行行ってみたい\nA: ",
            "[X訳] 今日めっちゃ眠い\n=> ",
            "[X訳] 新しいAIモデルが公開されてワクワクする\n=> ",
            "[関西] このラーメンめっちゃ美味しい\n=> ",
            "[津軽] 今日すごく寒い\n=> ",
        ]
        log("\n--- H-LUT-SFT (HARD Boolean LUTs!) Specialized + Unseen-Topic Samples ---")
        for p in sft_prompts:
            g = generate_text(hlut_model, tok, p, max_new=56, temp=0.42, stop_nl=True, hard=True)
            sft_samples.append((p.replace("\n", " / "), g))
            log(f"  SFT  | {p.replace(chr(10), ' / ')} >>> {g}")

    summary = {
        "gpu": gpu_name,
        "vocab_size": 4096,
        "train_tokens": int(len(train_tokens)),
        "val_tokens": int(len(val_tokens)),
        "fp_base": {**res_fp, "f16_mb": round(fp_f16_b / 1e6, 2), "samples": fp_samples},
        "hlut_base": {
            **res_hlut,
            "f16_mb": round(hlut_f16_b / 1e6, 2),
            "int8_mb": round(hlut_i8_b / 1e6, 2),
            "ppl_gap_vs_fp": round(res_hlut["best_hard_ppl"] - res_fp["best_hard_ppl"], 2),
            "samples": hlut_samples,
        },
        "hlut_sft": {**(res_sft or {}), "samples": sft_samples},
    }
    with open("/content/griffith_base_poc_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("\n==================== ROUTE 2 FRONTIER SUMMARY ====================")
    log(
        f"FP-Base    (SwiGLU, {fp_f16_b/1e6:.1f} MB f16)             : "
        f"val_hard_ppl = {res_fp['best_hard_ppl']:.2f} ({res_fp['train_sec']:.0f}s)"
    )
    log(
        f"H-LUT-Base (0-mul FWHT + 49K N=4 LUTs, {hlut_f16_b/1e6:.1f} MB): "
        f"val_hard_ppl = {res_hlut['best_hard_ppl']:.2f} (gap = {res_hlut['best_hard_ppl']-res_fp['best_hard_ppl']:+.2f}, {res_hlut['train_sec']:.0f}s)"
    )
    if res_sft:
        log(
            f"H-LUT-SFT  (350 steps specialization)         : "
            f"pre_ppl = {res_sft['pre_sft_ppl']:.2f} -> post_ppl = {res_sft['best_hard_ppl']:.2f} ({res_sft['train_sec']:.0f}s)"
        )
    log("DONE!")


if __name__ == "__main__":
    main()
