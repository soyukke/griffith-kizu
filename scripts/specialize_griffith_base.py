#!/usr/bin/env python3
"""
Griffith-JP-Base v1 Downstream Specializer:
Takes a pretrained `griffith_jp_base_v1.pt` + `bpe8192_merges.json` and specializes it
onto ANY custom Japanese corpus (X自動翻訳, 関西弁, ゲーム NPC, カスタムキャラ, etc.)
in ~2-5 minutes on CUDA/MPS, exporting a compact ~9.2 MB `.glgpt` v3 binary ready for
Pure Kizu native CLI and browser WASM!

Usage:
  python3 scripts/specialize_griffith_base.py \
      --base data/griffith_jp_base_v1/griffith_jp_base_v1.pt \
      --bpe  data/griffith_jp_base_v1/bpe8192_merges.json \
      --corpus data/jpllm_corpus.txt \
      --out web/jpllm/models/v3_chat.glgpt \
      --steps 600
"""

import argparse
import json
import math
import os
import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colab_train_griffith_jp_base import (
    KizuExactBpe,
    LoopedGriffithLM,
    eval_lm_ppl,
    export_glgpt_v3,
    generate_text,
)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description="Specialize Griffith-JP-Base v1 into a custom .glgpt v3 model")
    ap.add_argument("--base", required=True, help="Path to pretrained griffith_jp_base_v1.pt")
    ap.add_argument("--bpe", required=True, help="Path to bpe8192_merges.json")
    ap.add_argument("--corpus", required=True, help="Double-newline separated training examples (.txt)")
    ap.add_argument("--out", required=True, help="Output .glgpt v3 binary path")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr-fp", type=float, default=2.5e-4)
    ap.add_argument("--lr-lut", type=float, default=1.0e-2)
    args = ap.parse_args()

    device = pick_device()
    print(f"Specializing Griffith-JP-Base v1 on {device}...", flush=True)

    with open(args.bpe, "r", encoding="utf-8") as f:
        bpe_data = json.load(f)
    bpe = KizuExactBpe(bpe_data["merge_left"], bpe_data["merge_right"])

    model = LoopedGriffithLM(
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
    )
    state = torch.load(args.base, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    if device != "cuda":
        # Disable torch.compile on MPS/CPU for clean compatibility
        for blk in model.blocks:
            blk.ffn.lut1.use_compile = False
            blk.ffn.lut2.use_compile = False

    with open(args.corpus, "r", encoding="utf-8") as f:
        blocks = [b.strip() for b in f.read().split("\n\n") if b.strip()]
    val_n = min(200, max(20, len(blocks) // 10))
    train_blocks, val_blocks = blocks[:-val_n], blocks[-val_n:]

    def encode_stream(blks: list[str]):
        toks, wts, starts = [], [], []
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

    tr_toks, tr_wts, tr_starts = encode_stream(train_blocks)
    va_toks, _, _ = encode_stream(val_blocks)
    seq_len = model.seq_len
    valid_starts = [s for s in tr_starts if s + seq_len + 1 < len(tr_toks)]
    assert valid_starts, "Corpus too short for seq_len=256"

    lut_params = [p for n, p in model.named_parameters() if "theta" in n]
    fp_params = [p for n, p in model.named_parameters() if "theta" not in n]
    opt = torch.optim.AdamW(
        [
            {"params": fp_params, "lr": args.lr_fp, "weight_decay": 0.015},
            {"params": lut_params, "lr": args.lr_lut, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
    )

    t0 = time.time()
    best_ppl = 1e9
    best_step = 0
    best_state = None
    eval_every = max(100, args.steps // 4)

    for step in range(1, args.steps + 1):
        prog = (step - 1) / max(1, args.steps - 1)
        lr_mult = min(1.0, step / 30.0) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        opt.param_groups[0]["lr"] = args.lr_fp * lr_mult
        opt.param_groups[1]["lr"] = args.lr_lut * lr_mult
        model.set_anneal(0.80 - 0.25 * prog, 0.08 * max(0.0, 1.0 - prog / 0.85))
        model.train()
        model.set_hard_eval(False)

        xs, ys, ws = [], [], []
        for _ in range(args.batch):
            st = valid_starts[int(torch.randint(0, len(valid_starts), (1,)).item())]
            xs.append(tr_toks[st : st + seq_len])
            ys.append(tr_toks[st + 1 : st + seq_len + 1])
            ws.append(tr_wts[st + 1 : st + seq_len + 1])
        xb = torch.stack(xs).to(device, dtype=torch.long)
        yb = torch.stack(ys).to(device, dtype=torch.long)
        wb = torch.stack(ws).to(device, dtype=torch.float32)

        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        per_tok = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), yb.reshape(-1), reduction="none")
        loss = (per_tok * wb.reshape(-1)).sum() / wb.sum().clamp_min(1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % eval_every == 0 or step == args.steps:
            model.eval()
            model.set_hard_eval(True)
            with torch.no_grad():
                v_loss = 0.0
                cnt = 0
                stride = max(1, (len(va_toks) - seq_len - 1) // 32)
                for p in range(0, len(va_toks) - seq_len - 1, stride):
                    vx = va_toks[p : p + seq_len].unsqueeze(0).to(device, dtype=torch.long)
                    vy = va_toks[p + 1 : p + seq_len + 1].unsqueeze(0).to(device, dtype=torch.long)
                    vl = model(vx)
                    v_loss += F.cross_entropy(vl.float().reshape(-1, vl.size(-1)), vy.reshape(-1)).item()
                    cnt += 1
                    if cnt >= 24:
                        break
                hppl = math.exp(v_loss / max(1, cnt))
            mark = ""
            if hppl < best_ppl:
                best_ppl = hppl
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                mark = " [BEST]"
            print(f"step={step:4d}/{args.steps} loss={loss.item():.4f} val_hard_ppl={hppl:.2f}{mark} ({time.time()-t0:.1f}s)", flush=True)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    model.set_hard_eval(True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    nbytes, _ = export_glgpt_v3(model, bpe, args.out, soft_ppl=best_ppl, hard_ppl=best_ppl, train_steps=best_step)
    print(f"Exported specialized .glgpt v3 -> {args.out} ({nbytes/1e6:.2f} MB, best_hard_ppl={best_ppl:.2f})", flush=True)


if __name__ == "__main__":
    main()
