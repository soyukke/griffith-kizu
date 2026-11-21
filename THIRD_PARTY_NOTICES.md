# Third-Party Notices

This repository is an unofficial Kizu port and experimental application of Griffith.
It is not affiliated with or endorsed by the authors of the projects listed below.

## Griffith
- Source: https://github.com/tsukushi-gate/Griffith (by `@Tebasaki_lab`)
- Used as the architectural and algorithmic reference for differentiable logic gate networks (WARP-LUTs, Gumbel-ST, `LogicLadderGPT`, XNOR-popcount + argmax MUX Attention, and JSON IR).

## Kizu
- Source: https://github.com/kizu-lang/kizu
- Compiler, standard library, GPU codegen (`metal` / `spirv`), and `wasm32-browser` runtime.

## kizukani
- Source: https://github.com/soyukke/kizukani
- License: Apache License 2.0 (Copyright 2026 soyukke)
- `src/gpu/metal.kizu` is adapted from `kizukani/src/metal/metal.kizu`.

## Martian Mono Font
- Included in `web/fonts/martian-mono-latin-wght-normal.woff2`
- License: SIL Open Font License 1.1 (see `web/fonts/MartianMono-OFL.txt`).

## Distillation Teacher Model (`LAB-04 jpllm/`)
- Synthetic Japanese style-transfer and chat corpus (`data/jpllm_corpus.txt`, generated locally and excluded from git) was generated using `Qwen3-4B-Instruct` (Apache License 2.0) to train the compact BPE student checkpoints in `web/jpllm/models/`.
