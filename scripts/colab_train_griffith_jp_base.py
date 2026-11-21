#!/usr/bin/env python3
"""
Griffith-JP-Base v1 Production Trainer (Route 2 Frontier):
  - Pure Kizu-compatible Byte-Level BPE (vocab=8192, raw bytes 0..255 + 7936 merges, EOS=0)
  - Real ~45M-token Japanese corpus: Japanese Wikipedia + FineWeb-2 Japanese + code mix
  - Looped Griffith Transformer: d=512, GQA (8Q/2KV), RoPE, 6 unique x 2 loops = 12 effective layers
  - H-LUT-FFN (ZERO FFN matrix multiplications):
      FWHT(RMSNorm(x)) -> 2-bit Thermometer -> 98,304 N=4 WARP-LUTs (196,608 effective/tok) -> GroupSum -> FWHT
  - Phase 1: Converged Looped FP-Base Teacher (SwiGLU, 10,000 steps)
  - Phase 2: Warm-started & KL-distilled H-LUT-Base (7,000 steps, best EvalHard checkpoint saved)
  - Phase 3: Specialized 4-mode Japanese Chat & Style Transfer SFT (800 steps w/ 15% general replay)
  - Direct export of compact ~9.2 MB `.glgpt` v3 binaries (int8 emb+GQA + discrete N=4 LUTs)
    plus PyTorch checkpoints and a bit-exact verification fixture for Pure Kizu runtime tests!
"""

import json
import math
import os
import re
import struct
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def log(msg: str) -> None:
    print(msg, flush=True)


# ==============================================================================
# 1. Zero-Multiplication Fast Walsh-Hadamard Transform (compiled for speed)
# ==============================================================================
def _fwht_ortho_impl(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    h = x
    step = 1
    while step < n:
        h = h.reshape(*x.shape[:-1], -1, 2, step)
        u = h[..., 0, :]
        v = h[..., 1, :]
        h = torch.stack((u + v, u - v), dim=-2).reshape(*x.shape)
        step *= 2
    return h * (1.0 / math.sqrt(n))


def fwht_last_dim_raw(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized FWHT (for LUT truth-table logits: exact Kizu wh_theta_to_logits)."""
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


try:
    fwht_ortho = torch.compile(_fwht_ortho_impl)
except Exception:
    fwht_ortho = _fwht_ortho_impl


# ==============================================================================
# 2. Fast Fused N=4 WARP-LUT (4-level Horner factoring, zero recompiles)
# ==============================================================================
def _warp_n4_ste(
    x: torch.Tensor,
    idx0: torch.Tensor,
    idx1: torch.Tensor,
    idx2: torch.Tensor,
    idx3: torch.Tensor,
    theta: torch.Tensor,
    inv_tau: torch.Tensor,
    gumbel_scale: torch.Tensor,
) -> torch.Tensor:
    s0 = 1.0 - 2.0 * x.index_select(1, idx0)
    s1 = 1.0 - 2.0 * x.index_select(1, idx1)
    s2 = 1.0 - 2.0 * x.index_select(1, idx2)
    s3 = 1.0 - 2.0 * x.index_select(1, idx3)

    t = theta.to(x.dtype)
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
    z32 = (c0 + s3 * c1).float()

    u = torch.rand_like(z32).clamp_(1e-5, 1.0 - 1e-5)
    z32 = z32 + torch.log(u / (1.0 - u)) * gumbel_scale
    y_soft = torch.sigmoid(z32 * inv_tau).to(x.dtype)
    y_hard = (z32 > 0.0).to(x.dtype)
    return y_hard + (y_soft - y_soft.detach())


try:
    _warp_n4_ste_compiled = torch.compile(_warp_n4_ste)
except Exception:
    _warp_n4_ste_compiled = _warp_n4_ste


class FastWarpLutN4(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, seed: int = 42, use_compile: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_compile = use_compile
        self.tau = 1.0
        self.gumbel_scale = 0.18
        self.hard_eval = False

        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.empty((out_dim, 4), dtype=torch.long)
        for p in range(4):
            idx[:, p] = torch.randint(0, in_dim, (out_dim,), generator=g)
        for p in range(1, 4):
            for prev in range(p):
                coll = idx[:, p] == idx[:, prev]
                if coll.any():
                    idx[coll, p] = (idx[coll, p] + p * 7 + 1) % in_dim

        self.register_buffer("idx0", idx[:, 0].contiguous())
        self.register_buffer("idx1", idx[:, 1].contiguous())
        self.register_buffer("idx2", idx[:, 2].contiguous())
        self.register_buffer("idx3", idx[:, 3].contiguous())
        self.register_buffer("_inv_tau", torch.tensor(1.0, dtype=torch.float32), persistent=False)
        self.register_buffer("_gumbel_scale", torch.tensor(0.18, dtype=torch.float32), persistent=False)
        self.theta = nn.Parameter(torch.randn(out_dim, 16) * 0.25)

    def sync_anneal(self, tau: float, gumbel_scale: float) -> None:
        self.tau = tau
        self.gumbel_scale = gumbel_scale
        self._inv_tau.fill_(1.0 / max(1e-4, tau))
        self._gumbel_scale.fill_(gumbel_scale)

    def discretized_indices_and_tt(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (indices [out_dim, 4] u16, truth_tables [out_dim] u16) matching Kizu EvalHard."""
        idx = torch.stack((self.idx0, self.idx1, self.idx2, self.idx3), dim=1).cpu().numpy().astype(np.uint16)
        lg = fwht_last_dim_raw(self.theta.detach().float().cpu())  # [out_dim, 16]
        bits = (lg > 0.0).numpy().astype(np.uint16)
        shifts = (1 << np.arange(16, dtype=np.uint16)).reshape(1, 16)
        tt = (bits * shifts).sum(axis=1).astype(np.uint16)
        return idx, tt

    def discretized_bytes(self) -> int:
        return self.out_dim * 10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hard_eval or not self.training:
            return self.forward_hard(x)
        fn = _warp_n4_ste_compiled if self.use_compile else _warp_n4_ste
        return fn(x, self.idx0, self.idx1, self.idx2, self.idx3, self.theta, self._inv_tau, self._gumbel_scale)

    @torch.no_grad()
    def forward_hard(self, x: torch.Tensor) -> torch.Tensor:
        xb = (x >= 0.5).to(torch.long)
        addr = (
            xb.index_select(1, self.idx0)
            | (xb.index_select(1, self.idx1) << 1)
            | (xb.index_select(1, self.idx2) << 2)
            | (xb.index_select(1, self.idx3) << 3)
        )
        tt = (fwht_last_dim_raw(self.theta.float()) > 0.0).to(x.dtype)
        row = torch.arange(self.out_dim, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        return tt[row, addr]


class HLutFfn(nn.Module):
    def __init__(self, d_model: int, h1_gates: int = 8192, group_size: int = 16, seed: int = 42):
        super().__init__()
        self.d_model = d_model
        self.h1_gates = h1_gates
        self.group_size = group_size
        self.n_bits = 2
        self.temp = 0.45
        self.hard_eval = False

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
        self.lut1.sync_anneal(tau, gumbel_scale)
        self.lut2.sync_anneal(tau, gumbel_scale)

    def total_gates(self) -> int:
        return self.lut1.out_dim + self.lut2.out_dim

    def discretized_bytes(self) -> int:
        return self.lut1.discretized_bytes() + self.lut2.discretized_bytes() + self.d_model * 6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2 = x.reshape(-1, self.d_model)
        u = fwht_ortho(x2)
        diff = u.unsqueeze(-1) - self.thresh.to(u.dtype).unsqueeze(0)
        if self.hard_eval or not self.training:
            bits = (diff > 0.0).to(u.dtype).reshape(u.shape[0], -1)
        else:
            ys = torch.sigmoid(diff / self.temp)
            yh = (diff > 0.0).to(u.dtype)
            bits = (yh + (ys - ys.detach())).reshape(u.shape[0], -1)
        h1 = self.lut1(bits)
        h2 = self.lut2(h1)
        gs = h2.reshape(-1, self.d_model, self.group_size).mean(dim=-1) - 0.5
        y = fwht_ortho(gs) * self.out_scale.to(gs.dtype).unsqueeze(0)
        return y.reshape(orig_shape)


# ==============================================================================
# 3. Looped Griffith Transformer (6 unique x 2 loops = 12 effective layers)
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
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.rep > 1:
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).contiguous().reshape(B, T, D))


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
    def __init__(
        self,
        vocab_size: int = 8192,
        seq_len: int = 256,
        d_model: int = 512,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        n_unique_layers: int = 6,
        n_loops: int = 2,
        use_lut_ffn: bool = False,
        h1_gates: int = 8192,
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
        self.h1_gates = h1_gates
        self.group_size = group_size

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.loop_emb = nn.Parameter(torch.zeros(n_loops, d_model))
        self.blocks = nn.ModuleList(
            [
                GriffithBlock(
                    d_model, n_heads, n_kv_heads, use_lut_ffn, h1_gates, group_size, seed=101 + l * 31
                )
                for l in range(n_unique_layers)
            ]
        )
        self.final_ln = RMSNorm(d_model)
        self.register_buffer("rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)

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

    def warm_start_from_fp(self, fp_model: "LoopedGriffithLM") -> None:
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
        return F.linear(x, self.tok_emb.weight.to(x.dtype))


# ==============================================================================
# 4. 100% Kizu-Compatible Byte-Level BPE (raw bytes 0..255 + ordered merges!)
# ==============================================================================
def gpt2_bytes_to_unicode() -> tuple[dict[int, str], dict[str, int]]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    b2u = {b: chr(c) for b, c in zip(bs, cs)}
    u2b = {chr(c): b for b, c in zip(bs, cs)}
    return b2u, u2b


class KizuExactBpe:
    """
    100% token-for-token identical to Kizu src/data/bpe.kizu:
      - IDs 0..255 are raw UTF-8 bytes 0x00..0xFF (EOS = 0)
      - IDs 256..vocab_size-1 are (left_id, right_id) merges in exact rank order
      - Uses Rust HuggingFace `tokenizers` BPE with use_regex=False for fast training & batch encoding,
        then verifies exact parity against pure Kizu greedy-merge reference!
    """

    def __init__(self, merge_left: list[int], merge_right: list[int]):
        self.merge_left = merge_left
        self.merge_right = merge_right
        self.num_merges = len(merge_left)
        self.vocab_size = 256 + self.num_merges
        self.eos_id = 0

        # Build byte table for decoding
        self.tok_bytes: list[bytes] = [bytes([b]) for b in range(256)]
        for l, r in zip(merge_left, merge_right):
            self.tok_bytes.append(self.tok_bytes[l] + self.tok_bytes[r])
        self.pair_rank = {(l, r): rk for rk, (l, r) in enumerate(zip(merge_left, merge_right))}

        # Build fast Rust tokenizer matching our EXACT IDs 0..vocab_size-1!
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers

        b2u, _ = gpt2_bytes_to_unicode()
        id_to_sym = [b2u[b] for b in range(256)]
        merges_sym = []
        for l, r in zip(merge_left, merge_right):
            merges_sym.append((id_to_sym[l], id_to_sym[r]))
            id_to_sym.append(id_to_sym[l] + id_to_sym[r])
        vocab_map = {sym: idx for idx, sym in enumerate(id_to_sym)}

        tok = Tokenizer(models.BPE(vocab=vocab_map, merges=merges_sym))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        tok.decoder = decoders.ByteLevel()
        self.fast_tok = tok

    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 8192) -> "KizuExactBpe":
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers

        tb0 = time.time()
        # Pre-chunk into <=24-char pieces ONLY during BpeTrainer frequency counting so Rust
        # BpeTrainer deduplicates words and finishes in ~3s instead of stalling on 4KB docs.
        # Resulting merge table is loaded with use_regex=False for 100% Kizu parity!
        chunk_re = re.compile(r"\[[^\]]{1,8}\]|=> |U: |A: |[^\n。、！？\s]{1,24}|[\n。、！？\s]+")

        def chunk_iter():
            for t in texts:
                for m in chunk_re.findall(t):
                    yield m

        b2u, u2b = gpt2_bytes_to_unicode()
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=[],
            initial_alphabet=[b2u[b] for b in range(256)],
        )
        tok.train_from_iterator(chunk_iter(), trainer=trainer)
        spec = json.loads(tok.to_str())
        raw_merges = spec["model"]["merges"]
        log(f"BpeTrainer finished in {time.time()-tb0:.1f}s ({len(raw_merges)} raw merges)")

        # Map each symbol piece -> exact Kizu ID (0..255 raw bytes, 256+ merged)
        sym_to_id = {b2u[b]: b for b in range(256)}
        ml, mr = [], []
        for item in raw_merges:
            if isinstance(item, str):
                ls, rs = item.split(" ")
            else:
                ls, rs = item[0], item[1]
            if ls in sym_to_id and rs in sym_to_id:
                lid = sym_to_id[ls]
                rid = sym_to_id[rs]
                new_sym = ls + rs
                if new_sym not in sym_to_id:
                    sym_to_id[new_sym] = 256 + len(ml)
                    ml.append(lid)
                    mr.append(rid)
                    if 256 + len(ml) >= vocab_size:
                        break
        inst = cls(ml, mr)
        # Verify 100% parity on Japanese + English + punctuation!
        for probe in [
            "量子コンピュータとは、量子力学の原理を応用した計算機です。",
            "[会話]\nU: 今日めっちゃ眠い\nA: 俺はクソほど眠いんだよ。\n",
            "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
        ]:
            ref = inst.encode_ref(probe.encode("utf-8"))
            fast = inst.encode_fast(probe)
            assert ref == fast, f"BPE parity mismatch on {probe!r}: ref={ref[:12]} vs fast={fast[:12]}"
            assert inst.decode(ref) == probe, "BPE decode roundtrip mismatch!"
        log(f"KizuExactBpe verified 100% bit-exact parity (vocab={inst.vocab_size}, merges={inst.num_merges})!")
        return inst

    def encode_ref(self, raw_bytes: bytes) -> list[int]:
        """Exact mirror of Kizu src/data/bpe.kizu bpe_encode."""
        ids = list(raw_bytes)
        while len(ids) >= 2:
            best_rank = 1 << 30
            best_pos = -1
            for i in range(len(ids) - 1):
                rk = self.pair_rank.get((ids[i], ids[i + 1]))
                if rk is not None and rk < best_rank:
                    best_rank = rk
                    best_pos = i
            if best_pos < 0:
                break
            new_id = 256 + best_rank
            # Replace all non-overlapping occurrences of best pair
            l_t, r_t = ids[best_pos], ids[best_pos + 1]
            out = []
            i = 0
            n = len(ids)
            while i < n:
                if i + 1 < n and ids[i] == l_t and ids[i + 1] == r_t:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return ids

    def encode_fast(self, text: str) -> list[int]:
        return self.fast_tok.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.fast_tok.encode_batch(texts)]

    def decode(self, ids: list[int]) -> str:
        buf = bytearray()
        for tid in ids:
            if 0 <= tid < self.vocab_size and tid != self.eos_id:
                buf.extend(self.tok_bytes[tid])
        return buf.decode("utf-8", errors="replace")


# ==============================================================================
# 5. `.glgpt` v3 Binary Exporter (int8 per-row quantized emb+GQA + discrete LUTs)
# ==============================================================================
def quantize_rowwise_int8(w: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Returns (scales_f16 [R], q_i8 [R, C]) and verifies dequant fidelity."""
    wf = w.detach().float().cpu().numpy()
    max_abs = np.max(np.abs(wf), axis=1)
    scales = np.where(max_abs > 1e-8, max_abs / 127.0, 1.0).astype(np.float16)
    scales_f32 = scales.astype(np.float32)[:, None]
    q = np.clip(np.round(wf / scales_f32), -127, 127).astype(np.int8)
    return scales, q


def dequantize_rowwise_int8(scales_f16: np.ndarray, q_i8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(q_i8.astype(np.float32) * scales_f16.astype(np.float32)[:, None])


def export_glgpt_v3(
    model: LoopedGriffithLM,
    bpe: KizuExactBpe,
    out_path: str,
    soft_ppl: float,
    hard_ppl: float,
    train_steps: int,
) -> tuple[int, dict]:
    """Writes `.glgpt` v3 binary and also returns int8-simulated verification logits!"""
    assert model.use_lut_ffn, "export_glgpt_v3 expects H-LUT model"
    buf = bytearray()
    flags = 1 | 2 | 4  # int8_weights | use_lut_ffn | use_fwht
    buf.extend(b"GLGP")
    header = [
        3,  # version
        flags,
        bpe.vocab_size,
        model.seq_len,
        model.d_model,
        model.n_heads,
        model.n_kv_heads,
        model.n_unique_layers,
        model.n_loops,
        model.h1_gates,
        model.group_size,
        2,  # ffn_n_bits
        4,  # ffn_n_inputs
        1000000,  # rope_base * 100
        int(round(soft_ppl * 100.0)),
        int(round(hard_ppl * 100.0)),
        int(train_steps),
        bpe.eos_id,
    ]
    buf.extend(struct.pack("<" + "i" * len(header), *header))

    # BPE merges
    for l, r in zip(bpe.merge_left, bpe.merge_right):
        buf.extend(struct.pack("<HH", l, r))

    def push_f16(t: torch.Tensor) -> None:
        buf.extend(t.detach().float().cpu().numpy().astype(np.float16).tobytes())

    def push_q8(w: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        sc, q = quantize_rowwise_int8(w)
        buf.extend(sc.tobytes())
        buf.extend(q.tobytes())
        return sc, q

    # Shared weights
    emb_sc, emb_q = push_q8(model.tok_emb.weight)
    push_f16(model.loop_emb)
    push_f16(model.final_ln.weight)

    # Per unique layer
    q_blocks = []
    for blk in model.blocks:
        push_f16(blk.ln1.weight)
        push_f16(blk.ln2.weight)
        wq_s, wq_q = push_q8(blk.attn.wq.weight)
        wk_s, wk_q = push_q8(blk.attn.wk.weight)
        wv_s, wv_q = push_q8(blk.attn.wv.weight)
        wo_s, wo_q = push_q8(blk.attn.wo.weight)
        q_blocks.append((wq_s, wq_q, wk_s, wk_q, wv_s, wv_q, wo_s, wo_q))

        push_f16(blk.ffn.thresh)
        push_f16(blk.ffn.out_scale)
        for lut in (blk.ffn.lut1, blk.ffn.lut2):
            idx_u16, tt_u16 = lut.discretized_indices_and_tt()
            buf.extend(idx_u16.tobytes())
            buf.extend(tt_u16.tobytes())

    with open(out_path, "wb") as f:
        f.write(buf)

    # Build exact int8-dequantized verification fixture (so Pure Kizu can assert bit-close parity!)
    verify_prompt = "量子コンピュータとは、"
    verify_ids = bpe.encode_ref(verify_prompt.encode("utf-8"))
    with torch.no_grad():
        device = next(model.parameters()).device
        # Run exact quantized copy on CPU float32 for clean reference
        ref = LoopedGriffithLM(
            vocab_size=model.vocab_size,
            seq_len=model.seq_len,
            d_model=model.d_model,
            n_heads=model.n_heads,
            n_kv_heads=model.n_kv_heads,
            n_unique_layers=model.n_unique_layers,
            n_loops=model.n_loops,
            use_lut_ffn=True,
            h1_gates=model.h1_gates,
            group_size=model.group_size,
        )
        ref.load_state_dict(model.state_dict())
        ref.tok_emb.weight.copy_(dequantize_rowwise_int8(emb_sc, emb_q))
        ref.loop_emb.copy_(ref.loop_emb.half().float())
        ref.final_ln.weight.copy_(ref.final_ln.weight.half().float())
        for blk, qb in zip(ref.blocks, q_blocks):
            blk.ln1.weight.copy_(blk.ln1.weight.half().float())
            blk.ln2.weight.copy_(blk.ln2.weight.half().float())
            blk.attn.wq.weight.copy_(dequantize_rowwise_int8(qb[0], qb[1]))
            blk.attn.wk.weight.copy_(dequantize_rowwise_int8(qb[2], qb[3]))
            blk.attn.wv.weight.copy_(dequantize_rowwise_int8(qb[4], qb[5]))
            blk.attn.wo.weight.copy_(dequantize_rowwise_int8(qb[6], qb[7]))
            blk.ffn.thresh.copy_(blk.ffn.thresh.half().float())
            blk.ffn.out_scale.copy_(blk.ffn.out_scale.half().float())
        ref.eval()
        ref.set_hard_eval(True)
        logits = ref(torch.tensor([verify_ids], dtype=torch.long))[0, -1]
        topv, topi = torch.topk(logits, 10)
        verify_info = {
            "prompt": verify_prompt,
            "ids": verify_ids,
            "top10_ids": [int(i) for i in topi.tolist()],
            "top10_logits": [round(float(v), 4) for v in topv.tolist()],
            "first16_logits": [round(float(v), 4) for v in logits[:16].tolist()],
        }
        del ref
    return len(buf), verify_info


# ==============================================================================
# 6. Production Corpus Loader (~40M-50M real Japanese + conversational + code tokens)
# ==============================================================================
def load_production_corpus() -> tuple[list[str], list[str]]:
    from datasets import load_dataset

    docs = []
    # A. Japanese Wikipedia (14,000 rich articles -> ~28M-32M tokens)
    log("Streaming 14,000 Japanese Wikipedia articles...")
    ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
    w_cnt = 0
    for row in ds_wiki:
        t = row.get("text", "").strip()
        if len(t) >= 300:
            docs.append(t[:4500])
            w_cnt += 1
            if w_cnt >= 14000:
                break
    log(f"  Wikipedia JA: {w_cnt:,} docs ({sum(len(d) for d in docs)/1e6:.1f}M chars)")

    # B. Japanese FineWeb-2 (real Japanese web, blogs, everyday casual text, tech articles!)
    fw_cnt = 0
    try:
        log("Streaming 10,000 Japanese FineWeb-2 (jpn_Jpan) web/conversational docs...")
        ds_fw = load_dataset("HuggingFaceFW/fineweb-2", "jpn_Jpan", split="train", streaming=True)
        for row in ds_fw:
            t = row.get("text", "").strip()
            if 250 <= len(t) <= 6000:
                docs.append(t[:4000])
                fw_cnt += 1
                if fw_cnt >= 10000:
                    break
        log(f"  FineWeb-2 JA: {fw_cnt:,} docs")
    except Exception as e:
        log(f"  FineWeb-2 stream unavailable ({e}); adding 6,000 extra Wikipedia JA docs instead...")
        ds_wiki2 = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
        seen = 0
        for row in ds_wiki2:
            t = row.get("text", "").strip()
            if len(t) >= 300:
                seen += 1
                if seen > w_cnt:
                    docs.append(t[:4500])
                    fw_cnt += 1
                    if fw_cnt >= 6000:
                        break
        log(f"  Extra Wikipedia JA: {fw_cnt:,} docs")

    # C. English Wikipedia tech/science slice (800 docs) so English/tech tokens work cleanly
    en_cnt = 0
    try:
        ds_en = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        for row in ds_en:
            t = row.get("text", "").strip()
            if len(t) >= 300:
                docs.append(t[:3000])
                en_cnt += 1
                if en_cnt >= 800:
                    break
        log(f"  Wikipedia EN slice: {en_cnt:,} docs")
    except Exception:
        pass

    # Deterministic interleave & hold out 200 docs (100 wiki + 100 web) for validation
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(docs))
    shuffled = [docs[i] for i in perm]
    val_docs = shuffled[:200]
    train_docs = shuffled[200:]
    log(
        f"Total corpus: {len(train_docs):,} train docs, {len(val_docs):,} val docs "
        f"({sum(len(d) for d in train_docs)/1e6:.1f}M train chars)"
    )
    return train_docs, val_docs


# ==============================================================================
# 7. Training & Evaluation Helpers
# ==============================================================================
@torch.no_grad()
def eval_lm_ppl(
    model: LoopedGriffithLM,
    val_tokens: torch.Tensor,
    seq_len: int,
    batch_size: int = 16,
    n_batches: int = 40,
    hard: bool = True,
) -> float:
    model.eval()
    model.set_hard_eval(hard)
    device = next(model.parameters()).device
    total_loss = 0.0
    stride = max(1, (len(val_tokens) - seq_len - 1) // max(1, n_batches * batch_size))
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
    bpe: KizuExactBpe,
    prompt: str,
    max_new: int = 64,
    temp: float = 0.50,
    top_p: float = 0.86,
    rep_pen: float = 1.20,
    stop_nl: bool = False,
    hard: bool = True,
) -> str:
    model.eval()
    model.set_hard_eval(hard)
    device = next(model.parameters()).device
    ids = bpe.encode_ref(prompt.encode("utf-8"))
    gen = []
    for _ in range(max_new):
        ctx = ids[-model.seq_len :]
        xb = torch.tensor([ctx], device=device, dtype=torch.long)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xb)[0, -1].float()
        logits[bpe.eos_id] -= 0.0  # allow EOS naturally
        for tid in set(ids[-24:]):
            if tid >= 128:
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
            sp[cs - sp > top_p] = 0.0
            sp /= sp.sum()
            nxt = int(si[torch.multinomial(sp, 1).item()].item())
        if nxt == bpe.eos_id:
            break
        ids.append(nxt)
        gen.append(nxt)
        txt = bpe.decode(gen)
        if stop_nl and "\n" in txt:
            return txt.split("\n")[0].strip()
    return bpe.decode(gen).strip()


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
    save_path: str | None = None,
    eval_every: int = 1000,
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
    warmup_steps = min(300, max(50, steps // 15))

    for step in range(1, steps + 1):
        prog = (step - 1) / max(1, steps - 1)
        warmup = min(1.0, step / warmup_steps)
        lr_mult = warmup * (0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * prog)))
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
                loss = (1.0 - kd_alpha) * ce + kd_alpha * kl
            else:
                loss = ce

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step == 100 or step % eval_every == 0 or step == steps:
            dt = time.time() - t0
            tps = (step * batch_size * seq_len) / max(1e-3, dt)
            hppl = eval_lm_ppl(model, val_tokens, seq_len, batch_size=16, n_batches=40, hard=True)
            mark = ""
            if hppl < best_ppl:
                best_ppl = hppl
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if save_path:
                    torch.save(best_state, save_path)
                mark = " [SAVED BEST]"
            log(
                f"  [{name}] step={step:5d}/{steps} loss={loss.item():.4f} "
                f"val_hard_ppl={hppl:.2f}{mark} tok/s={tps:,.0f} ({dt/60:.1f}m)"
            )
        elif step % 250 == 0:
            dt = time.time() - t0
            tps = (step * batch_size * seq_len) / max(1e-3, dt)
            log(f"  [{name}] step={step:5d}/{steps} loss={loss.item():.4f} tok/s={tps:,.0f} ({dt/60:.1f}m)")

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


def sft_specialize_with_replay(
    model: LoopedGriffithLM,
    bpe: KizuExactBpe,
    corpus_path: str,
    general_train_tokens: torch.Tensor,
    steps: int = 800,
    batch_size: int = 16,
    lr_fp: float = 2.5e-4,
    lr_lut: float = 1.0e-2,
) -> dict:
    """
    Fine-tunes H-LUT-Base on 4-mode Japanese Chat & Style Transfer with:
      - Example-aligned packing + response-token weighting (1.0 on response, 0.15 on prompt)
      - 15% general Japanese pretrain replay (prevents catastrophic forgetting!)
    """
    with open(corpus_path, "r", encoding="utf-8") as f:
        raw = f.read()
    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
    train_blocks = blocks[:-250]
    val_blocks = blocks[-250:]

    def encode_sft_stream(blks: list[str]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        toks = []
        wts = []
        starts = []
        for b in blks:
            starts.append(len(toks))
            lines = b.split("\n")
            for li, line in enumerate(lines):
                piece = line + ("\n" if li < len(lines) - 1 else "")
                ids = bpe.encode_fast(piece)
                is_resp = line.startswith("=> ") or line.startswith("A: ")
                w = 1.0 if is_resp else 0.15
                toks.extend(ids)
                wts.extend([w] * len(ids))
            toks.append(bpe.eos_id)
            wts.append(1.0)
        return torch.tensor(toks, dtype=torch.int32), torch.tensor(wts, dtype=torch.float32), starts

    tr_toks, tr_wts, tr_starts = encode_sft_stream(train_blocks)
    va_toks, _, _ = encode_sft_stream(val_blocks)
    seq_len = model.seq_len
    valid_starts = [s for s in tr_starts if s + seq_len + 1 < len(tr_toks)]
    log(f"SFT loaded: {len(blocks)} examples -> train={len(tr_toks):,} tok ({len(valid_starts)} aligned starts), val={len(va_toks):,} tok")

    pre_ppl = eval_lm_ppl(model, va_toks, seq_len, batch_size=8, n_batches=16, hard=True)
    log(f"  [H-LUT-Chat] Pre-SFT val_hard_ppl = {pre_ppl:.2f}")

    device = next(model.parameters()).device
    lut_params = [p for n, p in model.named_parameters() if "theta" in n]
    fp_params = [p for n, p in model.named_parameters() if "theta" not in n]
    opt = torch.optim.AdamW(
        [
            {"params": fp_params, "lr": lr_fp, "weight_decay": 0.015},
            {"params": lut_params, "lr": lr_lut, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
    )

    gen_max = len(general_train_tokens) - seq_len - 2
    t0 = time.time()
    best_ppl = 1e9
    best_step = 0
    best_state = None

    for step in range(1, steps + 1):
        prog = (step - 1) / max(1, steps - 1)
        lr_mult = min(1.0, step / 40.0) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        opt.param_groups[0]["lr"] = lr_fp * lr_mult
        opt.param_groups[1]["lr"] = lr_lut * lr_mult
        model.set_anneal(0.80 - 0.25 * prog, 0.08 * max(0.0, 1.0 - prog / 0.85))
        model.train()
        model.set_hard_eval(False)

        # Mix 13 SFT sequences + 3 general Japanese replay sequences (~19% replay)
        n_replay = 3
        n_sft = batch_size - n_replay
        xs, ys, ws = [], [], []
        for _ in range(n_sft):
            st = valid_starts[int(torch.randint(0, len(valid_starts), (1,)).item())]
            xs.append(tr_toks[st : st + seq_len])
            ys.append(tr_toks[st + 1 : st + seq_len + 1])
            ws.append(tr_wts[st + 1 : st + seq_len + 1])
        for _ in range(n_replay):
            st = int(torch.randint(0, gen_max, (1,)).item())
            xs.append(general_train_tokens[st : st + seq_len])
            ys.append(general_train_tokens[st + 1 : st + seq_len + 1])
            ws.append(torch.full((seq_len,), 0.40, dtype=torch.float32))

        xb = torch.stack(xs).to(device, dtype=torch.long)
        yb = torch.stack(ys).to(device, dtype=torch.long)
        wb = torch.stack(ws).to(device, dtype=torch.float32)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xb)
            per_tok = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), yb.reshape(-1), reduction="none")
            loss = (per_tok * wb.reshape(-1)).sum() / wb.sum().clamp_min(1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step in (1, 200, 400, 600, steps):
            hppl = eval_lm_ppl(model, va_toks, seq_len, batch_size=8, n_batches=16, hard=True)
            dt = time.time() - t0
            mark = ""
            if hppl < best_ppl:
                best_ppl = hppl
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                mark = " [BEST]"
            log(f"  [H-LUT-Chat] step={step:4d}/{steps} loss={loss.item():.4f} val_hard_ppl={hppl:.2f}{mark} ({dt:.1f}s)")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    model.set_hard_eval(True)
    return {"pre_sft_ppl": pre_ppl, "best_hard_ppl": best_ppl, "best_step": best_step, "train_sec": time.time() - t0}


def main() -> None:
    assert torch.cuda.is_available(), "CUDA GPU required!"
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    log(f"=== Griffith-JP-Base v1 PRODUCTION RUN on {gpu_name} ({vram_gb:.1f} GB VRAM) ===")

    try:
        import datasets  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets", "tokenizers"])

    # 1. Stream production Japanese corpus & train 100% Kizu-exact BPE 8192
    t0 = time.time()
    train_docs, val_docs = load_production_corpus()
    bpe_fit = train_docs[:3500]
    if os.path.exists("/content/jpllm_corpus.txt"):
        with open("/content/jpllm_corpus.txt", "r", encoding="utf-8") as f:
            jpllm_lines = [ln for ln in f.read().splitlines() if ln.strip()]
        bpe_fit = bpe_fit + jpllm_lines * 2
    log(f"Training 100% Kizu-exact BPE 8192 on {len(bpe_fit):,} texts...")
    bpe = KizuExactBpe.train(bpe_fit, vocab_size=8192)

    def encode_all(docs: list[str], label: str) -> torch.Tensor:
        te0 = time.time()
        out = []
        # Chunk encode_batch to keep peak RAM crisp
        for i in range(0, len(docs), 2000):
            for ids in bpe.encode_batch(docs[i : i + 2000]):
                out.extend(ids)
                out.append(bpe.eos_id)
        log(f"  Encoded {label}: {len(docs):,} docs -> {len(out):,} tokens in {time.time()-te0:.1f}s")
        return torch.tensor(out, dtype=torch.int32)

    train_tokens = encode_all(train_docs, "train")
    val_tokens = encode_all(val_docs, "val")
    log(
        f"Production corpus tokenized in {time.time()-t0:.1f}s | "
        f"vocab={bpe.vocab_size}, train={len(train_tokens):,} tok, val={len(val_tokens):,} tok"
    )

    # 2. Phase 1: Converged FP-Base Teacher (6 unique x 2 loops = 12 effective layers, d=512)
    log("\n=== PHASE 1: Production FP-Base Teacher (d=512, 6x2=12 eff layers, V=8192) ===")
    fp_model = LoopedGriffithLM(
        vocab_size=bpe.vocab_size,
        seq_len=256,
        d_model=512,
        n_heads=8,
        n_kv_heads=2,
        n_unique_layers=6,
        n_loops=2,
        use_lut_ffn=False,
    ).to(device)
    log(f"FP-Base params = {sum(p.numel() for p in fp_model.parameters()):,}")
    res_fp = train_phase(
        "FP-Base-v1",
        fp_model,
        train_tokens,
        val_tokens,
        steps=9000,
        batch_size=16,
        lr_fp=1.3e-3,
        save_path="/content/fp_base_v1.pt",
        eval_every=1500,
    )
    fp_samples = []
    log("\n--- FP-Base v1 Samples ---")
    for p in ["量子コンピュータとは、", "日本の東京は", "人工知能の発展により、", "Pythonで関数を定義するには、"]:
        g = generate_text(fp_model, bpe, p, max_new=64, temp=0.50)
        fp_samples.append((p, g))
        log(f"  FP   | {p} >>> {g}")

    # 3. Phase 2: Production H-LUT-Base (0-mul FWHT + 98,304 N=4 WARP-LUTs, 12 effective layers!)
    log("\n=== PHASE 2: Production H-LUT-Base (0-mul FWHT + 98,304 N=4 LUTs, 196K eff/tok) ===")
    hlut_model = LoopedGriffithLM(
        vocab_size=bpe.vocab_size,
        seq_len=256,
        d_model=512,
        n_heads=8,
        n_kv_heads=2,
        n_unique_layers=6,
        n_loops=2,
        use_lut_ffn=True,
        h1_gates=8192,
        group_size=16,
    ).to(device)
    hlut_model.warm_start_from_fp(fp_model)
    fp_model.eval()
    for p in fp_model.parameters():
        p.requires_grad_(False)

    res_hlut = train_phase(
        "H-LUT-Base-v1",
        hlut_model,
        train_tokens,
        val_tokens,
        steps=6500,
        batch_size=16,
        lr_fp=4.5e-4,
        lr_lut=2.8e-2,
        teacher=fp_model,
        kd_alpha=0.55,
        kd_temp=2.0,
        save_path="/content/griffith_jp_base_v1.pt",
        eval_every=1000,
    )

    base_bytes, base_verify = export_glgpt_v3(
        hlut_model,
        bpe,
        "/content/griffith_jp_base_v1.glgpt",
        soft_ppl=res_fp["best_hard_ppl"],
        hard_ppl=res_hlut["best_hard_ppl"],
        train_steps=res_hlut["best_step"],
    )
    log(f"Exported /content/griffith_jp_base_v1.glgpt ({base_bytes/1e6:.2f} MB, 98,304 N=4 LUTs, 0 FFN muls!)")

    hlut_samples = []
    log("\n--- H-LUT-Base v1 (HARD Boolean LUTs, 0 FFN Muls!) Samples ---")
    for p in ["量子コンピュータとは、", "日本の東京は", "人工知能の発展により、", "Pythonで関数を定義するには、"]:
        g = generate_text(hlut_model, bpe, p, max_new=64, temp=0.50, hard=True)
        hlut_samples.append((p, g))
        log(f"  HLUT | {p} >>> {g}")

    del fp_model
    torch.cuda.empty_cache()

    # 4. Phase 3: Specialized 4-Mode Japanese Chat + General Replay (.glgpt v3!)
    res_sft = None
    chat_bytes = 0
    chat_verify = None
    sft_samples = []
    if os.path.exists("/content/jpllm_corpus.txt"):
        log("\n=== PHASE 3: Downstream Specialization (H-LUT-Base v1 -> 4-mode Japanese Chat w/ Replay) ===")
        res_sft = sft_specialize_with_replay(
            hlut_model, bpe, "/content/jpllm_corpus.txt", train_tokens, steps=800, batch_size=16
        )
        # Measure retained general Japanese Wikipedia/Web PPL after replay-protected SFT!
        gen_ppl_after = eval_lm_ppl(hlut_model, val_tokens, 256, batch_size=16, n_batches=40, hard=True)
        res_sft["general_val_ppl_after_sft"] = gen_ppl_after
        log(f"  [H-LUT-Chat] General JA Val PPL retained after SFT = {gen_ppl_after:.2f} (base was {res_hlut['best_hard_ppl']:.2f})")

        torch.save(hlut_model.state_dict(), "/content/griffith_jp_chat_v1.pt")
        chat_bytes, chat_verify = export_glgpt_v3(
            hlut_model,
            bpe,
            "/content/griffith_jp_chat_v1.glgpt",
            soft_ppl=res_hlut["best_hard_ppl"],
            hard_ppl=res_sft["best_hard_ppl"],
            train_steps=res_sft["best_step"],
        )
        log(f"Exported /content/griffith_jp_chat_v1.glgpt ({chat_bytes/1e6:.2f} MB)")

        sft_prompts = [
            "[会話]\nU: おはよう\nA: ",
            "[会話]\nU: あなたは誰？\nA: ",
            "[会話]\nU: 量子コンピュータってすごい？\nA: ",
            "[会話]\nU: 徹夜でバグが直らない\nA: ",
            "[会話]\nU: ラーメン食べたい\nA: ",
            "[会話]\nU: 宇宙旅行行ってみたい\nA: ",
            "[X訳] 今日めっちゃ眠い\n=> ",
            "[X訳] 新しいAIモデルが公開されてワクワクする\n=> ",
            "[関西] このラーメンめっちゃ美味しい\n=> ",
            "[関西] 仕事終わったから帰る\n=> ",
            "[津軽] 今日すごく寒い\n=> ",
            "[津軽] このラーメンめっちゃ美味しい\n=> ",
        ]
        log("\n--- H-LUT-Chat v1 (HARD Boolean LUTs, 9.2 MB!) Live Samples ---")
        for p in sft_prompts:
            g = generate_text(hlut_model, bpe, p, max_new=60, temp=0.40, stop_nl=True, hard=True)
            sft_samples.append((p.replace("\n", " / "), g))
            log(f"  CHAT | {p.replace(chr(10), ' / ')} >>> {g}")

    summary = {
        "gpu": gpu_name,
        "vocab_size": bpe.vocab_size,
        "train_tokens": int(len(train_tokens)),
        "val_tokens": int(len(val_tokens)),
        "fp_base": {**res_fp, "samples": fp_samples},
        "hlut_base": {
            **res_hlut,
            "glgpt_bytes": base_bytes,
            "ppl_gap_vs_fp": round(res_hlut["best_hard_ppl"] - res_fp["best_hard_ppl"], 2),
            "samples": hlut_samples,
            "verify": base_verify,
        },
        "hlut_chat": {
            **(res_sft or {}),
            "glgpt_bytes": chat_bytes,
            "samples": sft_samples,
            "verify": chat_verify,
        },
    }
    with open("/content/griffith_jp_base_v1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open("/content/griffith_jp_bpe8192_merges.json", "w", encoding="utf-8") as f:
        json.dump({"merge_left": bpe.merge_left, "merge_right": bpe.merge_right}, f)

    log("\n==================== GRIFFITH-JP-BASE v1 SUMMARY ====================")
    log(f"Corpus      : {len(train_tokens):,} train tok / {len(val_tokens):,} val tok (V={bpe.vocab_size})")
    log(f"FP-Base v1  : val_hard_ppl = {res_fp['best_hard_ppl']:.2f} ({res_fp['train_sec']/60:.1f}m)")
    log(
        f"H-LUT-Base  : val_hard_ppl = {res_hlut['best_hard_ppl']:.2f} "
        f"(gap={res_hlut['best_hard_ppl']-res_fp['best_hard_ppl']:+.2f}, {base_bytes/1e6:.2f} MB, 98,304 N=4 LUTs, {res_hlut['train_sec']/60:.1f}m)"
    )
    if res_sft:
        log(
            f"H-LUT-Chat  : sft_ppl = {res_sft['best_hard_ppl']:.2f}, general_ppl = {res_sft['general_val_ppl_after_sft']:.2f} "
            f"({chat_bytes/1e6:.2f} MB, {res_sft['train_sec']/60:.1f}m)"
        )
    log("ALL PRODUCTION ARTIFACTS EXPORTED!")


if __name__ == "__main__":
    main()
