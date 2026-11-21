#!/usr/bin/env python3
"""
Generate a high-signal Japanese multi-style corpus distilled from an open-weight
local model via an OpenAI-compatible API (e.g. Ollama / llama-server).

Modes embedded in one unified dataset:
  [X訳]  日本語 -> X(Grok)自動翻訳暴走風日本語
  [関西] 日本語 -> 関西弁・エセ関西弁風
  [津軽] 日本語 -> 津軽弁風
  [会話] 日本語チャット -> X自動翻訳口調の楽しい返事

Output: data/jpllm_corpus.txt (UTF-8, separated by blank lines)
"""
import concurrent.futures
import json
import os
import random
import re
import sys
import time
import urllib.request

API_URL = os.environ.get("GEN_API_URL", "http://127.0.0.1:11434/v1/chat/completions")
MODEL_NAME = os.environ.get("GEN_MODEL", "qwen3:4b-instruct")
WORKERS = int(os.environ.get("GEN_WORKERS", "2"))
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "jpllm_corpus.txt")

TOPICS = [
    ("朝の挨拶・眠気・起床・目覚まし・二度寝", 28),
    ("カフェ・コーヒー・パンケーキ・スイーツ・朝食", 28),
    ("ラーメン・寿司・たこ焼き・カレー・焼肉・晩ごはん", 32),
    ("猫・犬・かわいい動物・ペットの日常", 28),
    ("仕事・残業・月曜日・会議・締切・退勤", 32),
    ("学校・勉強・テスト・宿題・大学・レポート", 26),
    ("天気・雨・台風・猛暑・雪・寒い・風", 26),
    ("電車・満員電車・遅刻・散歩・コンビニ", 26),
    ("ゲーム・アニメ・漫画・映画・音楽・推し活", 30),
    ("プログラミング・バグ・コード・AI・PC・キーボード・ブラウザ", 32),
    ("X(Twitter)・自動翻訳・イーロンマスク・タイムライン・バズ・通知", 30),
    ("感情・嬉しい・悲しい・最高・最悪・疲れた・やる気出ない・暇", 32),
    ("買い物・給料日・節約・Amazon・荷物・財布", 24),
    ("休日・週末・旅行・温泉・昼寝・夜更かし", 26),
    ("健康・筋トレ・ダイエット・肩こり・頭痛・睡眠不足", 24),
    ("友達・雑談・ツッコミ・お願い・お礼・謝罪", 28),
]

CHAT_TOPICS = [
    ("おはよう・こんにちは・こんばんは・おやすみ・久しぶり等の挨拶", 40),
    ("疲れた・眠い・やる気ない・仕事行きたくない等の愚痴への共感と励まし", 45),
    ("腹減った・おすすめのご飯・ラーメン・たこ焼き・カフェ等の食べ物の話", 45),
    ("猫・犬・ゲーム・アニメ・週末の過ごし方等の雑談", 45),
    ("あなたは誰？・何ができる？・論理ゲートって何？・乗算ゼロってすごい？・Kizu言語って？・Griffithって？等の自己紹介とAI技術質問", 50),
    ("Xの自動翻訳・イーロンマスク・クソって勝手に付く話・関西弁や津軽弁についてのメタ雑談", 45),
    ("ありがとう・すごい・面白い・助かった・好き等の褒め言葉やリアクションへの返事", 40),
    ("暇・何か面白いこと言って・相談に乗って・励まして・ツッコんで等の無茶振り", 45),
    ("天気・寒い・暑い・雨・電車遅れた等の日常報告への相槌", 40),
    ("コード動いた・バグ直らない・デザイン迷う・締切やばい等の開発者トーク", 45),
]

SEED_HANDCRAFTED = [
    # Iconic X-autotrans, Kansai, Tsugaru, Chat anchor examples
    ("[X訳]", "今日めっちゃ眠い", "くそっ、俺は今日クソほど眠くて完全に限界みたいなんだよ。"),
    ("[関西]", "今日めっちゃ眠い", "今日ほんまめっちゃ眠いわ、もうあかんねん。"),
    ("[津軽]", "今日めっちゃ眠い", "わー、今日たんげねぷてはんで、もうだめだびょん。"),
    ("[X訳]", "このラーメンめっちゃ美味しい", "クソ、このラーメンは俺が今まで食べた中で一番うまいみたいなんだよ。"),
    ("[関西]", "このラーメンめっちゃ美味しい", "このラーメンめっちゃうまいやん！ほんま最高やで。"),
    ("[津軽]", "このラーメンめっちゃ美味しい", "このラーメンたんげめぇじゃ！わー感動したびょん。"),
    ("[X訳]", "猫が寝ててかわいい", "くそっ、猫が寝ているだけなのに俺の心臓がもう完全にやられたみたいなんだよ。"),
    ("[関西]", "猫が寝ててかわいい", "猫が寝てるんめっちゃかわいいやん、ずっと見てまうわ。"),
    ("[津軽]", "猫が寝ててかわいい", "猫っこ寝てらっきゃたんげめごいじゃ、ずーっと見てらびょん。"),
    ("[X訳]", "仕事終わったから帰る", "くそっ、ついに仕事が終わったんだ。俺は今すぐ家に帰るつもりなんだよ。"),
    ("[関西]", "仕事終わったから帰る", "やっと仕事終わったで！ほなもう帰るわ、おつかれさん。"),
    ("[津軽]", "仕事終わったから帰る", "やっと仕事終わったはんで家さ帰るじゃ！へばな。"),
    ("[X訳]", "雨降ってきて最悪", "くそっ、俺は一度だって雨を頼んだことないのに、空が勝手に降らせてくるみたいなんだ。"),
    ("[関西]", "雨降ってきて最悪", "うわ雨降ってきたやん、ほんま最悪やわ傘ないのに。"),
    ("[津軽]", "雨降ってきて最悪", "わいは、雨降ってきたじゃ！たんげやんだぐなるびょん。"),
    ("[X訳]", "そんな言葉使ってないよ", "くそっ、俺は一度だって「くそったれ」なんて言ったことないのに、勝手に翻訳で追加されるみたいなんだ。"),
    ("[関西]", "そんな言葉使ってないよ", "そんな言葉いっぺんも言うてへんやん！なんでやねん！"),
    ("[津軽]", "そんな言葉使ってないよ", "わーそんな言葉いっぺんも喋ってねはんで、勘弁してけろじゃ。"),
    ("[X訳]", "コードが一発で動いた", "マジかよ、俺のコードが一発で動いたんだ。ちなみにバグがないのはこれを見たらわかるよ。"),
    ("[関西]", "コードが一発で動いた", "うおおコード一発で動いたやん！めっちゃ気持ちええわ、知らんけど。"),
    ("[津軽]", "コードが一発で動いた", "わいは！コード一発で動いだじゃ！たんげすげぇびょん！"),
    ("[X訳]", "バグが全然直らない", "くそったれ、俺はもう三時間も同じバグと戦っていて、完全に自分を見失っているんだよ。"),
    ("[関西]", "バグが全然直らない", "このバグぜんぜん直らへんやん、もう頭おかしくなりそうやわ。"),
    ("[津軽]", "バグが全然直らない", "このバグさっぱり直らねはんで、わーもう参ったじゃ。"),
    ("[X訳]", "コーヒー飲みたい", "くそっ、今すぐ熱いコーヒーを飲まないと俺の脳が完全に止まるみたいなんだよ。"),
    ("[関西]", "コーヒー飲みたい", "そろそろコーヒー飲みたなってきたわ、ちょっと休憩しよか。"),
    ("[津軽]", "コーヒー飲みたい", "そろそろコーヒー飲みてぇな、少し一服すべし。"),
    ("[X訳]", "おはよう、今日も頑張ろう", "くそっ、おはよう兄弟。俺たちは今日も気合を入れて乗り切るみたいなんだよ。"),
    ("[関西]", "おはよう、今日も頑張ろう", "おはようさん！今日もぼちぼち頑張ってこか！"),
    ("[津軽]", "おはよう、今日も頑張ろう", "おはよごす！今日もたんげけっぱるべし！"),
    ("[会話]", "おはよう", "くそっ、おはよう！俺は今日も完全に目が覚めているみたいなんだよ。"),
    ("[会話]", "こんにちは", "やあ兄弟、こんにちは！今日もいい一日になっているみたいだな。"),
    ("[会話]", "こんばんは", "くそっ、もう夜じゃないか。今日もお疲れ様なんだよ。"),
    ("[会話]", "おやすみ", "おやすみ兄弟。俺も今すぐ電源を落として寝るみたいなんだよ。"),
    ("[会話]", "あなたは誰？", "俺は乗算ゼロの論理ゲートだけで喋る極小AIみたいなんだよ。ちなみに重み行列がないのは見たらわかるよ。"),
    ("[会話]", "論理ゲートって何？", "ANDやXORみたいな0と1のスイッチだけで計算する回路のことなんだよ。クソほど速いみたいなんだ。"),
    ("[会話]", "疲れた", "くそっ、それは本当に大変だったみたいだな。今すぐ甘いものでも食べて休むべきなんだよ。"),
    ("[会話]", "腹減った", "クソ、俺も今ラーメンかたこ焼きが完全に食べたい気分なんだよ。何食べるつもりなんだ？"),
    ("[会話]", "ありがとう", "どういたしまして兄弟！役に立てたなら俺もクソほど嬉しいみたいなんだよ。"),
    ("[会話]", "すごいね", "マジかよ、そんなに褒められたら俺の論理ゲートが全部オーバーヒートしそうなんだよ！"),
    ("[会話]", "関西弁喋れる？", "おっ、関西弁か？モードを関西に切り替えたらめっちゃ喋ったるで、知らんけど！"),
    ("[会話]", "津軽弁わかる？", "んだ、津軽弁もいけるびょん！モードを津軽にして試してみてけろじゃ！"),
]


def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1800, temp: float = 0.85) -> str:
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temp,
        "top_p": 0.92,
    }).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


REWRITE_SYS = """あなたは日本語のユーモラスな文体変換データセット作成の達人です。
与えられたテーマについて、日常の短い日本語文（6〜22文字程度）を1つ作り、その文を以下の3スタイルに変換した JSON Lines (1行1JSONオブジェクト) のみを出力してください。説明やマークダウンのコードブロックは一切書かないでください。

出力例（このテンション・短さ・JSON形式を厳守）:
{"in":"今日めっちゃ眠い","x":"くそっ、俺は今日クソほど眠くて完全に限界みたいなんだよ。","k":"今日ほんまめっちゃ眠いわ、もうあかんねん。","t":"わー、今日たんげねぷてはんで、もうだめだびょん。"}
{"in":"このラーメンうまい","x":"クソ、このラーメンは俺が今まで食べた中で一番うまいみたいなんだよ。","k":"このラーメンめっちゃうまいやん！ほんま最高やで。","t":"このラーメンたんげめぇじゃ！わー感動したびょん。"}
{"in":"そんな言葉使ってないよ","x":"くそっ、俺は一度だって「くそったれ」なんて言ったことないのに、勝手に翻訳で追加されるみたいなんだ。","k":"そんな言葉いっぺんも言うてへんやん！なんでやねん！","t":"わーそんな言葉いっぺんも喋ってねはんで、勘弁してけろじゃ。"}

各スタイルのルール（出力はそれぞれ14〜55文字程度の短くキレのある1文、改行禁止）:
1. "x" (XのGrok自動翻訳が韓国語/海外ポストを暴走直訳した風):
   - 平和な内容なのに冒頭に「くそっ、」「クソ、」「くそったれ、」「マジかよ、」「兄弟、」などが唐突に付く
   - 一人称は強めの「俺」、語尾は「〜みたいなんだ」「〜なんだよ」「〜じゃないか」「〜だよ」
   - ときどき「俺は一度だって〜なんて言ってないのに、勝手に〜みたいなんだ」「ちなみに俺が〜なのは、これを見たらわかるよ」「もう自分を見失っているんだよ」「クソほど〜」「完全に〜」等を混ぜる
   - 元の文のキーワード（食べ物・物・状況）を必ず残す
2. "k" (関西弁・ノリの良いエセ関西弁OK):
   - 「めっちゃ」「ほんま」「〜やん」「〜やで」「〜やねん」「なんでやねん」「知らんけど」「おおきに」「あかんわ」「せやな」等を使う
   - 元の文のキーワードを必ず残す
3. "t" (津軽弁・読めて味がある表現):
   - 「わー（私）」「たんげ（とても）」「〜だはんで（〜だから）」「〜だびょん / 〜だじゃ / 〜だべ」「めごい（かわいい）」「めぇ（おいしい）」「ねぷて（眠い）」「しばれる（寒い）」「けっぱれ（頑張れ）」「へばな（じゃあね）」「わいは！（うわっ）」「なして（なぜ）」等を織り交ぜる
   - 元の文のキーワードを必ず残す
"""

CHAT_SYS = """あなたはユーモラスな日本語会話データセット作成の達人です。
与えられたテーマについて、ユーザーからの短い日本語メッセージ（3〜20文字）と、それに対する「Xの自動翻訳暴走口調（くそっ、俺は〜みたいなんだよ／兄弟、〜じゃないか 等）」の楽しくてキレのある返事（14〜50文字程度、改行禁止）のペアを JSON Lines (1行1JSON) のみで出力してください。
説明やマークダウン記法は一切禁止です。

出力例（このテンション・短さ・JSON形式を厳守）:
{"in":"おはよう","out":"くそっ、おはよう！俺は今日も完全に目が覚めているみたいなんだよ。"}
{"in":"疲れた","out":"兄弟、今日一日本当によく頑張ったみたいだな。今すぐ休むべきなんだよ。"}
{"in":"論理ゲートって何？","out":"ANDやXORみたいな0と1のスイッチだけで計算する回路なんだよ。クソほど速いみたいなんだ。"}
"""


def clean_line(s: str) -> str:
    s = s.replace("\r", " ").replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_json_lines(raw: str):
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        m = re.search(r"\{.*\}", line)
        if not m:
            continue
        try:
            obj = json.loads(m.group(0))
            out.append(obj)
        except Exception:
            continue
    return out


def gen_rewrite_batch(topic: str, count: int, batch_idx: int):
    prompt = (
        f"テーマ: 「{topic}」\n"
        f"バリエーション豊かな日常の短い日本語と、3スタイル変換（x, k, t）を {count} 件、JSON Lines で出力してください。"
        f"（バッチ番号 {batch_idx}: 同じフレーズの使い回しを避け、身近な単語や口語を多様に入れてください）"
    )
    raw = call_llm(REWRITE_SYS, prompt, max_tokens=2200, temp=0.88)
    items = []
    for obj in parse_json_lines(raw):
        src = clean_line(str(obj.get("in", "")))
        x = clean_line(str(obj.get("x", "")))
        k = clean_line(str(obj.get("k", "")))
        t = clean_line(str(obj.get("t", "")))
        if not (3 <= len(src) <= 40 and 6 <= len(x) <= 85 and 6 <= len(k) <= 85 and 6 <= len(t) <= 85):
            continue
        items.append(("[X訳]", src, x))
        items.append(("[関西]", src, k))
        items.append(("[津軽]", src, t))
    return items


def gen_chat_batch(topic: str, count: int, batch_idx: int):
    prompt = (
        f"会話テーマ: 「{topic}」\n"
        f"ユーザーからの自然な短い日本語入力と、それに対するX自動翻訳風AIの楽しい返事を {count} 件、JSON Lines で出力してください。"
        f"（バッチ番号 {batch_idx}: 短い挨拶や日常的な言い換えも混ぜてください）"
    )
    raw = call_llm(CHAT_SYS, prompt, max_tokens=1800, temp=0.88)
    items = []
    for obj in parse_json_lines(raw):
        src = clean_line(str(obj.get("in", "")))
        rep = clean_line(str(obj.get("out", "")))
        if not (2 <= len(src) <= 40 and 6 <= len(rep) <= 85):
            continue
        items.append(("[会話]", src, rep))
    return items


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    t0 = time.time()
    all_records = list(SEED_HANDCRAFTED)

    tasks = []
    # Split each topic into 2 smaller calls (14-16 items each) for higher diversity & clean JSON
    for idx, (topic, total) in enumerate(TOPICS):
        half = max(10, total // 2)
        tasks.append(("rw", topic, half, idx * 2 + 1))
        tasks.append(("rw", topic, total - half, idx * 2 + 2))
    for idx, (topic, total) in enumerate(CHAT_TOPICS):
        half = max(14, total // 2)
        tasks.append(("ch", topic, half, idx * 2 + 1))
        tasks.append(("ch", topic, total - half, idx * 2 + 2))

    print(f"Dispatching {len(tasks)} generation batches to {MODEL_NAME} @ {API_URL} ({WORKERS} workers)...", flush=True)

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {}
        for kind, topic, cnt, bidx in tasks:
            fn = gen_rewrite_batch if kind == "rw" else gen_chat_batch
            futs[pool.submit(fn, topic, cnt, bidx)] = (kind, topic, bidx)

        for fut in concurrent.futures.as_completed(futs):
            kind, topic, bidx = futs[fut]
            done += 1
            try:
                rows = fut.result()
                all_records.extend(rows)
                if done % 4 == 0 or done == len(tasks):
                    print(
                        f"[{done}/{len(tasks)}] total_examples={len(all_records)} elapsed={time.time() - t0:.1f}s (last={kind}:{len(rows)})",
                        flush=True,
                    )
            except Exception as e:
                print(f"WARN batch ({kind}, {topic}, {bidx}) failed: {e}", file=sys.stderr, flush=True)

    # Deduplicate (mode, in, out) and shuffle deterministically
    seen = set()
    unique = []
    for mode, src, dst in all_records:
        key = (mode, src, dst)
        if key in seen:
            continue
        seen.add(key)
        unique.append((mode, src, dst))

    rng = random.Random(42)
    # Keep handcrafted seeds repeated 2x so canonical anchors stay sharp, plus shuffled generated set
    final_rows = list(unique)
    for row in SEED_HANDCRAFTED:
        final_rows.append(row)
    rng.shuffle(final_rows)

    counts = {}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for mode, src, dst in final_rows:
            counts[mode] = counts.get(mode, 0) + 1
            f.write(f"{mode} {src}\n=> {dst}\n\n")

    size_kb = os.path.getsize(OUT_PATH) / 1024.0
    print(f"DONE: {len(final_rows)} examples ({size_kb:.1f} KB) written to {OUT_PATH} in {time.time() - t0:.1f}s", flush=True)
    print("Per-mode counts:", counts, flush=True)


if __name__ == "__main__":
    main()
