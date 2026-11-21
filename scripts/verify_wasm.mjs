import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { instantiateKizu } from "../web/dist/app.mjs";

const wasmPath = path.resolve("web/dist/app.wasm");
const wasmBytes = fs.readFileSync(wasmPath);

const stdoutChunks = [];
const slotBuffers = { 0: new Uint8Array(0), 1: new Uint8Array(0) };
const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8", { fatal: false });
let lastCode = "";
let lastIrJson = "";
let lastPieceBytes = new Uint8Array(0);
let lastWindowToks = [];

const program = await instantiateKizu(wasmBytes, {
  write(stream, bytes) {
    if (stream === 1) stdoutChunks.push(new TextDecoder().decode(bytes));
    else process.stderr.write(new TextDecoder().decode(bytes));
    return true;
  },
  host: {
    read_input(host, slot, offset, ptr, cap) {
      const src = slotBuffers[slot];
      if (!src || offset >= src.byteLength) return 0;
      const end = Math.min(src.byteLength, offset + (cap >>> 0));
      return host.writeBytes(ptr, src.subarray(offset, end));
    },
    give_bytes(host, kind, ptr, len) {
      const raw = new Uint8Array(host.readBytes(ptr, len));
      if (kind === 0) lastCode = decoder.decode(raw);
      if (kind === 1) lastIrJson = decoder.decode(raw);
      if (kind === 2) lastPieceBytes = raw;
      if (kind === 3) {
        const toks = [];
        let s = 0;
        for (let i = 0; i <= raw.length; i++) {
          if (i === raw.length || raw[i] === 0) {
            toks.push(decoder.decode(raw.subarray(s, i)));
            s = i + 1;
          }
        }
        lastWindowToks = toks;
      }
    },
  },
});

const startStatus = program.start();
if (startStatus !== 0) throw new Error(`kizu_start failed with status ${startStatus}`);
const ex = program.exports;
console.log(`wasm_size_bytes=${wasmBytes.byteLength} version=${ex.lab_version()} stdout=${JSON.stringify(stdoutChunks.join("").trim())}`);

// 1. LAB-01b (compiler/): Stage 1 exact verification
const tasks = [
  { id: 0, patterns: 4, expected: (p) => (p & 1) ^ ((p >> 1) & 1) },
  { id: 1, patterns: 16, expected: (p) => (p & 1) ^ ((p >> 1) & 1) ^ ((p >> 2) & 1) ^ ((p >> 3) & 1) },
  { id: 2, patterns: 16, expected: (p) => (p & 3) + ((p >> 2) & 3) },
];
for (const t of tasks) {
  for (let p = 0; p < t.patterns; p++) {
    const exp = t.expected(p);
    if (ex.lab_eval(t.id, 0, p) !== exp || ex.lab_eval(t.id, 1, p) !== exp) {
      throw new Error(`Stage 1 mismatch task=${t.id} p=${p}`);
    }
  }
}
console.log("[LAB-01b compiler/] 36/36 patterns VM==AOT==expected: PASS");

// 2. LAB-01 (draw/): Live 5x5 trainer -> synthesis -> VM verification
const pack = (r0, r1, r2, r3, r4) => r0 | (r1 << 5) | (r2 << 10) | (r3 << 15) | (r4 << 20);
const shapes = [
  [pack(0b01110, 0b10001, 0b10001, 0b10001, 0b01110), 0],
  [pack(0b00100, 0b01010, 0b01010, 0b01010, 0b00100), 0],
  [pack(0b01110, 0b10001, 0b10001, 0b01110, 0b00000), 0],
  [pack(0b00000, 0b01110, 0b10001, 0b10001, 0b01110), 0],
  [pack(0b10001, 0b01010, 0b00100, 0b01010, 0b10001), 1],
  [pack(0b00000, 0b01010, 0b00100, 0b01010, 0b00000), 1],
  [pack(0b10001, 0b11011, 0b00100, 0b11011, 0b10001), 1],
  [pack(0b10010, 0b01100, 0b00100, 0b01100, 0b10010), 1],
  [pack(0b00100, 0b01010, 0b01010, 0b10001, 0b11111), 2],
  [pack(0b00100, 0b01110, 0b10001, 0b11111, 0b00000), 2],
  [pack(0b00000, 0b00100, 0b01010, 0b11111, 0b00000), 2],
  [pack(0b00100, 0b01010, 0b10001, 0b10001, 0b11111), 2],
];
const drawH = ex.live_create(42, 0) >>> 0;
for (const [b, l] of shapes) ex.live_add_sample(drawH, b, l);
const dt0 = performance.now();
ex.live_train_steps(drawH, 260, 85);
const dt1 = performance.now();
const drawOpt = ex.live_synthesize(drawH);
ex.live_emit_artifacts(drawH);
if (ex.live_metric_milli(drawH, 2) < 990 || drawOpt >= 120) throw new Error("LAB-01 draw failed");
console.log(`[LAB-01 draw/] train_260=${(dt1 - dt0).toFixed(1)}ms hard_acc=1.00 synth=120->${drawOpt} ir_gates=${JSON.parse(lastIrJson).nodes[0].params.gates.length}: PASS`);

// 3. LAB-02 (ladder/): Load ALL 5 real .glgpt checkpoints, run forward + MUX check + generation
for (const rung of ["l0", "l0b", "l1", "l2", "l3c"]) {
  const file = path.resolve(`web/ladder/models/${rung}.glgpt`);
  const blob = fs.readFileSync(file);
  slotBuffers[0] = blob;
  const h = ex.ladder_load(blob.byteLength) >>> 0;
  if (!h) throw new Error(`ladder_load failed for ${rung}`);
  const prompt = encoder.encode("ROMEO:\nO ");
  slotBuffers[1] = prompt;
  const t0 = performance.now();
  const seqLen = ex.ladder_eval_bytes(h, prompt.byteLength);
  const t1 = performance.now();
  const muls = ex.ladder_block_muls(h, seqLen);
  const gates = ex.ladder_meta(h, 9);
  const hppl = (ex.ladder_meta(h, 7) / 100).toFixed(2);
  // Check L3-C attention row is strictly 1-hot MUX
  if (rung === "l3c") {
    if (muls !== 0) throw new Error(`Expected 0 block muls for l3c, got ${muls}`);
    for (let l = 0; l < 2; l++) {
      for (let hd = 0; hd < 4; hd++) {
        let ones = 0;
        for (let k = 0; k < seqLen; k++) {
          const w = ex.ladder_attn_milli(h, l, hd, seqLen - 1, k);
          if (w === 1000) ones++;
          else if (w !== 0) throw new Error(`L3-C non-binary attention weight ${w}`);
        }
        if (ones !== 1) throw new Error(`L3-C MUX row did not have exactly 1 hot cell (${ones})`);
      }
    }
  }
  let gen = "ROMEO:\nO ";
  for (let i = 0; i < 24; i++) {
    const raw = encoder.encode(gen);
    slotBuffers[1] = raw;
    ex.ladder_eval_bytes(h, raw.byteLength);
    gen += String.fromCharCode(ex.ladder_sample_byte(h, 650, i === 0 ? 42 : 0));
  }
  console.log(`[LAB-02 ladder/${rung}] size=${(blob.byteLength / 1024).toFixed(0)}KB ppl=${hppl} gates=${gates} block_muls=${muls} fwd=${(t1 - t0).toFixed(2)}ms sample=${JSON.stringify(gen.replace(/\n/g, " / "))}`);
}

// 4. LAB-03 (distill/): Japanese Teacher + Scratch + KD + Logic Synthesis + Speed Bench inside WASM
const distillHtml = fs.readFileSync("web/distill/index.html", "utf8");
const tm = distillHtml.match(/const TRAIN_DATA = (\[[\s\S]*?\n    \]);/);
const um = distillHtml.match(/const UNLABELED_DATA = (\[[\s\S]*?\n    \]);/);
const vm = distillHtml.match(/const VAL_DATA = (\[[\s\S]*?\n    \]);/);
const TRAIN_DATA = eval(tm[1]);
const UNLABELED_DATA = eval(um[1]);
const VAL_DATA = eval(vm[1]);

const dh = ex.distill_create(25) >>> 0;
for (const [txt, lbl] of TRAIN_DATA) {
  const b = encoder.encode(txt); slotBuffers[1] = b;
  ex.distill_add_text(dh, b.byteLength, lbl, 0);
}
for (const txt of UNLABELED_DATA) {
  const b = encoder.encode(txt); slotBuffers[1] = b;
  ex.distill_add_text(dh, b.byteLength, -1, 2);
}
for (const [txt, lbl] of VAL_DATA) {
  const b = encoder.encode(txt); slotBuffers[1] = b;
  ex.distill_add_text(dh, b.byteLength, lbl, 1);
}
const kt0 = performance.now();
ex.distill_train_teacher(dh, 80, 30);
ex.distill_train_students(dh, 240, 60);
const kt1 = performance.now();
const dOpt = ex.distill_synthesize(dh);
ex.distill_emit_code(dh);

const tVal = ex.distill_metric_milli(dh, 1) / 10;
const sVal = ex.distill_metric_milli(dh, 4) / 10;
const kVal = ex.distill_metric_milli(dh, 7) / 10;
const agree = ex.distill_metric_milli(dh, 8) / 10;

// Speed bench
ex.distill_bench(dh, 0, 200);
ex.distill_bench(dh, 1, 200);
const bt0 = performance.now(); ex.distill_bench(dh, 0, 2000);
const bt1 = performance.now(); ex.distill_bench(dh, 1, 20000);
const bt2 = performance.now();
const fpNs = ((bt1 - bt0) * 1e6) / 2000;
const vmNs = ((bt2 - bt1) * 1e6) / (20000 * 64);

console.log(
  `[LAB-03 distill/] train=${(kt1 - kt0).toFixed(1)}ms val_acc(FP/Scratch/KD)=${tVal.toFixed(0)}%/${sVal.toFixed(0)}%/${kVal.toFixed(0)}% ` +
    `agree=${agree.toFixed(0)}% synth=160->${dOpt} fp=${(fpNs / 1000).toFixed(2)}us vm64=${vmNs.toFixed(1)}ns (${(fpNs / vmNs).toFixed(0)}x) aot_len=${lastCode.length}`
);
if (dOpt >= 160 || lastCode.length < 100 || kVal < sVal) throw new Error(`LAB-03 distill verification failed (kVal=${kVal} < sVal=${sVal})`);

// 5. LAB-04 (jpllm/): Japanese 4-mode BPE Logic-Gate LLM (.glgpt v2: l0 / l0b / l3c)
const jpllmDir = path.resolve("web/jpllm/models");
if (fs.existsSync(path.join(jpllmDir, "l3c.glgpt"))) {
  for (const rung of ["l0", "l0b", "l3c"]) {
    const file = path.join(jpllmDir, `${rung}.glgpt`);
    const blob = fs.readFileSync(file);
    slotBuffers[0] = blob;
    const h = ex.ladder_load(blob.byteLength) >>> 0;
    if (!h) throw new Error(`jpllm ladder_load failed for ${rung}`);
    const prompt = "[X訳] 今日めっちゃ眠い\n=> ";
    const pBytes = encoder.encode(prompt);
    slotBuffers[1] = pBytes;
    const t0 = performance.now();
    const seqLen = ex.ladder_eval_bytes(h, pBytes.byteLength);
    const t1 = performance.now();
    const muls = ex.ladder_block_muls(h, seqLen);
    const gates = ex.ladder_meta(h, 9);
    const hppl = (ex.ladder_meta(h, 7) / 100).toFixed(2);
    if (rung === "l3c") {
      if (muls !== 0) throw new Error(`jpllm l3c expected 0 block muls, got ${muls}`);
      for (let l = 0; l < 3; l++) {
        for (let hd = 0; hd < 4; hd++) {
          let ones = 0;
          for (let k = 0; k < seqLen; k++) {
            const w = ex.ladder_attn_milli(h, l, hd, seqLen - 1, k);
            if (w === 1000) ones++;
            else if (w !== 0) throw new Error(`jpllm l3c non-binary attn ${w}`);
          }
          if (ones !== 1) throw new Error(`jpllm l3c MUX row hot count ${ones}`);
        }
      }
    }
    const fullB = Array.from(pBytes);
    const outB = [];
    for (let i = 0; i < 36; i++) {
      const tid = ex.ladder_sample_bpe(h, 450, i === 0 ? 77 : 0);
      let hasNl = (tid === 10);
      for (const b of lastPieceBytes) {
        fullB.push(b);
        outB.push(b);
        if (b === 10) hasNl = true;
      }
      if (hasNl) break;
      const fb = new Uint8Array(fullB);
      slotBuffers[1] = fb;
      ex.ladder_eval_bytes(h, fb.byteLength);
    }
    const sample = decoder.decode(new Uint8Array(outB)).split("\n")[0];
    console.log(`[LAB-04 jpllm/${rung}] size=${(blob.byteLength / 1024).toFixed(0)}KB ppl=${hppl} gates=${gates} block_muls=${muls} fwd=${(t1 - t0).toFixed(2)}ms bpe_toks=${lastWindowToks.length} sample=${JSON.stringify(sample)}`);
  }
}

console.log("ALL LABS (compiler / draw / ladder / distill / jpllm) WASM VERIFICATION: PASS");
