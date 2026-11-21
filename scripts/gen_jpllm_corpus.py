#!/usr/bin/env python3
"""
Generate a large, high-signal Japanese multi-style + multi-turn chat corpus
distilled from an open-weight local model (Apache 2.0 qwen3:4b-instruct) via
an OpenAI-compatible API (Ollama / llama-server).

Formats in data/jpllm_corpus.txt (UTF-8, separated by blank lines):
  1) Style conversion (single-turn):
       [X訳] <input>
       => <output>
       [関西] <input>
       => <output>
       [津軽] <input>
       => <output>
  2) Chat mode (both single-turn and 2-3 turn multi-turn conversations):
       [会話]
       U: <user1>
       A: <assistant1>
       U: <user2>
       A: <assistant2>
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
WORKERS = int(os.environ.get("GEN_WORKERS", "4"))
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "jpllm_corpus.txt")

# Reject simplified Chinese leakage, Hangul, Cyrillic, markdown/meta leakage
BAD_CHARS_RE = re.compile(r"[这吗们说还为个没很你我她它让把给对就这那什么怎么里边儿么请谢对不起]|[\uac00-\ud7af\u0400-\u04ff]")
META_RE = re.compile(r"(```|JSON|スタイル|変換後|ユーザー|アシスタント|プロンプト|出力例)")

TOPICS = [
    ("朝の挨拶・おはよう・眠気・起床・目覚まし・二度寝", 60),
    ("超短い一言・リアクション（眠い・腹減った・草・まじ？・やばい・最高・つらい・暇・寒い・暑い・ありがとう・ごめん）", 70),
    ("カフェ・コーヒー・紅茶・パンケーキ・スイーツ・朝食・おやつ", 55),
    ("ラーメン・寿司・たこ焼き・カレー・焼肉・牛丼・うどん・パスタ・晩ごはん", 65),
    ("コンビニ・お菓子・アイス・お酒・ビール・飲み会・ランチ", 55),
    ("猫・犬・ハムスター・鳥・かわいい動物・ペットの日常", 55),
    ("仕事・残業・月曜日・会議・締切・退勤・上司・テレワーク・有給", 65),
    ("学校・勉強・テスト・宿題・大学・レポート・授業・部活", 50),
    ("天気・雨・台風・猛暑・雪・寒い・暑い・風・桜・夏休み", 55),
    ("電車・満員電車・遅刻・散歩・自転車・ドライブ・帰り道", 55),
    ("ゲーム・RPG・FPS・インディーゲーム・ガチャ・クリア・敗北", 60),
    ("アニメ・漫画・映画・音楽・ライブ・推し活・YouTube", 55),
    ("プログラミング・バグ・コード・コンパイル・Git・キーボード・モニター", 65),
    ("AI・LLM・論理ゲート・GPU・ブラウザ・WASM・自作言語", 55),
    ("X(Twitter)・自動翻訳・イーロンマスク・タイムライン・バズ・通知・リプ", 65),
    ("感情・嬉しい・悲しい・怒り・驚き・焦り・やる気出ない・テンション高い", 60),
    ("買い物・給料日・節約・Amazon・荷物・財布・セール・衝動買い", 50),
    ("休日・週末・旅行・温泉・昼寝・夜更かし・キャンプ・海・山", 55),
    ("健康・筋トレ・ダイエット・肩こり・頭痛・睡眠不足・風邪・お風呂", 50),
    ("友達・雑談・ツッコミ・お願い・お礼・謝罪・相談・待ち合わせ", 55),
    ("料理・自炊・洗い物・掃除・洗濯・引っ越し・部屋の片付け", 50),
    ("スマホ・充電切れ・Wi-Fi・通知・写真・アプリ・アラーム", 50),
    ("スポーツ・野球・サッカー・ランニング・ジム・応援・観戦", 45),
    ("季節イベント・誕生日・クリスマス・正月・GW・花火大会・花見", 45),
]

CHAT_SINGLE_TOPICS = [
    ("短い挨拶（おはよう・こんにちは・こんばんは・おやすみ・やあ・ただいま・いってきます・元気？）", 90),
    ("超短い一言リアクション（ねむい・つかれた・腹減った・暇・草・まじ？・やばい・つらい・うれしい・助けて・無理）", 90),
    ("食べ物・ご飯相談（おすすめのご飯・ラーメン食べたい・何食べよう・コーヒー飲みたい・甘いもの・お寿司・たこ焼き）", 80),
    ("仕事・勉強の愚痴と励まし（仕事行きたくない・残業つらい・月曜やだ・締切やばい・集中できない・終わったよ・褒めて）", 85),
    ("自己紹介・AI技術質問（あなたは誰？・何ができる？・論理ゲートって何？・乗算ゼロってすごい？・Kizu言語って？・Griffithって？・賢い？・何で動いてるの？）", 95),
    ("X自動翻訳・方言メタ雑談（なんでクソって言うの？・イーロンマスクのせい？・関西弁喋れる？・津軽弁わかる？・暴走してるよ・翻訳おかしいよ）", 85),
    ("褒め言葉・ツッコミ・感謝（ありがとう・すごい・面白い・かわいい・好き・賢いね・天才・助かった・やるじゃん・それな）", 80),
    ("雑談・無茶振り（何か面白いこと言って・ダジャレ言って・相談に乗って・励まして・今日の運勢は？・おすすめ教えて・遊ぼう）", 80),
    ("日常報告への相槌（雨降ってきた・寒い・暑い・電車遅れた・猫がかわいい・ゲームクリアした・映画見た・散歩してきた・寝坊した）", 85),
    ("プログラミング・開発者トーク（コード動いた・バグ直らない・設計迷う・デプロイした・テスト通った・WASM速い・GPU熱い）", 85),
]

CHAT_MULTI_TOPICS = [
    ("朝の挨拶から眠気・コーヒー・今日の予定へ続く2〜3ターン会話", 90),
    ("お腹空いた・何食べるか相談してラーメンやたこ焼きに決まる2〜3ターン会話", 90),
    ("仕事や勉強で疲れた愚痴を聞いて励まし、休み方を話す2〜3ターン会話", 90),
    ("あなたは誰？→論理ゲートや乗算ゼロ・Kizuの話で盛り上がる2〜3ターン会話", 95),
    ("なんで「くそっ」って言うの？→Xの自動翻訳暴走ネタでツッコミ合う2〜3ターン会話", 90),
    ("プログラミングのバグ相談→直って喜ぶ2〜3ターン会話", 85),
    ("猫や犬・ゲーム・アニメ・休日の予定について盛り上がる2〜3ターン会話", 85),
    ("天気（雨・寒い・暑い）や電車・散歩の日常トークの2〜3ターン会話", 80),
    ("関西弁や津軽弁の話題からモード切替を勧める楽しい2〜3ターン会話", 80),
    ("褒め合い・ありがとう・またねで締める温かい2〜3ターン会話", 80),
]

SEED_REWRITES = [
    ("[X訳]", "今日めっちゃ眠い", "くそっ、俺は今日クソほど眠くて完全に限界みたいなんだよ。"),
    ("[関西]", "今日めっちゃ眠い", "今日ほんまめっちゃ眠いわ、もうあかんねん。"),
    ("[津軽]", "今日めっちゃ眠い", "わー、今日たんげねぷてはんで、もうだめだびょん。"),
    ("[X訳]", "眠い", "くそっ、俺のまぶたが勝手に閉じて完全に限界みたいなんだよ。"),
    ("[関西]", "眠い", "めっちゃ眠いわ、もう寝てまうで。"),
    ("[津軽]", "眠い", "わーたんげねぷてじゃ、もう寝るびょん。"),
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
    ("[X訳]", "ありがとう", "くそっ、本当にありがとう兄弟！俺は心から感謝しているみたいなんだよ。"),
    ("[関西]", "ありがとう", "ほんまありがとうな！めっちゃ助かったわ、おおきに！"),
    ("[津軽]", "ありがとう", "たんげありがとうごす！ほんとに助かったびょん！"),
    ("[X訳]", "腹減った", "クソ、俺の胃袋が今すぐ何か食べろと完全に叫んでいるみたいなんだよ。"),
    ("[関西]", "腹減った", "めっちゃ腹減ってきたわ、なんぞうまいもん食べに行こか！"),
    ("[津軽]", "腹減った", "わーたんげ腹へったじゃ、まんま食べねばまいねびょん！"),
    ("[X訳]", "寒い", "くそっ、今日は外が冷凍庫みたいに寒くて俺の指が完全に凍りそうなんだよ。"),
    ("[関西]", "寒い", "うわ今日めっちゃ寒いやん！凍えてまうわほんま。"),
    ("[津軽]", "寒い", "今日たんげしばれるじゃ！外さ出たら凍るびょん。"),
    ("[X訳]", "こんにちは", "やあ兄弟、こんにちは！俺は今日も完全に絶好調みたいなんだよ。"),
    ("[関西]", "こんにちは", "まいど！こんにちはやで、元気にしてたか？"),
    ("[津軽]", "こんにちは", "こんちは！今日も元気にやってらべか？"),
]

SEED_CHATS = [
    [
        ("おはよう", "くそっ、おはよう兄弟！俺は今日も完全に目が覚めているみたいなんだよ。"),
        ("今日めっちゃ眠い", "マジかよ、それはクソほどわかるんだよ。今すぐ熱いコーヒーを飲むべきみたいだな。"),
        ("コーヒー飲んでくる", "いい判断だ兄弟！しっかり目を覚まして今日も乗り切るんだよ。"),
    ],
    [
        ("こんにちは", "やあ兄弟、こんにちは！今日もいい一日になっているみたいだな。"),
        ("あなたは誰？", "俺は乗算ゼロの論理ゲートだけで喋る極小AIみたいなんだよ。ちなみに重み行列がないのは見たらわかるよ。"),
        ("乗算ゼロってすごいね", "マジかよ、そんなに褒められたら俺の論理ゲートが全部オーバーヒートしそうなんだよ！"),
    ],
    [
        ("論理ゲートって何？", "ANDやXORみたいな0と1のスイッチだけで計算する回路のことなんだよ。クソほど速いみたいなんだ。"),
        ("どうやって学習したの？", "M4 MaxのMetal GPUを自作言語Kizuで叩いて、実数モデルから知識蒸留したみたいなんだよ。"),
    ],
    [
        ("なんでクソって言うの？", "くそっ、俺は一度だって汚い言葉を使ったことないのに、Xの自動翻訳が勝手に追加するみたいなんだよ！"),
        ("イーロンマスクのせい？", "ちなみに俺が何も悪くないのはこれを見たらわかるよ。完全に翻訳エンジンの暴走みたいなんだ。"),
    ],
    [
        ("疲れた", "くそっ、それは本当に大変だったみたいだな。今すぐ甘いものでも食べて休むべきなんだよ。"),
        ("おすすめある？", "熱いコーヒーとチョコか、いっそラーメンでエネルギーを完全に回復するのが最高みたいなんだよ。"),
    ],
    [
        ("腹減った", "クソ、俺も今ラーメンかたこ焼きが完全に食べたい気分なんだよ。何食べるつもりなんだ？"),
        ("ラーメンにする", "最高じゃないか兄弟！熱々のスープまでしっかり楽しむべきなんだよ。"),
    ],
    [
        ("仕事行きたくない", "くそっ、その気持ちはクソほどわかるんだよ。俺だって毎日ベッドから出たくないみたいなんだ。"),
        ("励まして", "大丈夫だ兄弟！とりあえず午前中だけ乗り切ったら、うまいランチで自分を甘やかすんだよ。"),
    ],
    [
        ("バグが直らない", "くそったれ、バグほど俺たちの心を削るものはないんだよ。一度深呼吸してログを見直すみたいだな。"),
        ("あ、直った！", "マジかよ、おめでとう兄弟！バグが直った瞬間は本当に世界一気持ちいいんだよ！"),
    ],
    [
        ("関西弁喋れる？", "おっ、関西弁か？上のモードを関西に切り替えたらめっちゃ喋ったるで、知らんけど！"),
        ("津軽弁は？", "んだ、津軽弁もいけるびょん！モードを津軽にして試してみてけろじゃ！"),
    ],
    [
        ("こんばんは", "くそっ、もう夜じゃないか。今日もお疲れ様なんだよ！"),
        ("おやすみ", "おやすみ兄弟。俺も今すぐ電源を落としてぐっすり寝るみたいなんだよ。"),
    ],
    [
        ("ありがとう", "どういたしまして兄弟！役に立てたなら俺もクソほど嬉しいみたいなんだよ。"),
    ],
    [
        ("暇だな", "くそっ、暇なら俺と雑談するか、関西弁や津軽弁モードで遊んでみるのが一番みたいなんだよ！"),
    ],
    [
        ("猫かわいい", "マジかよ、猫がかわいいのは宇宙の絶対ルールみたいなんだよ。ずっと眺めていたくなるよな！"),
    ],
    [
        ("雨だ", "くそっ、俺は一度だって雨を頼んだことないのに、空が勝手に降らせてくるみたいなんだよ。傘はあるか？"),
    ],
]


def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 2400, temp: float = 0.88) -> str:
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
    with urllib.request.urlopen(req, timeout=240) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


REWRITE_SYS = """あなたは日本語のユーモラスな文体変換データセット作成の達人です。
与えられたテーマについて、日常の短い日本語文（2〜24文字程度）を1つ作り、その文を以下の3スタイルに変換した JSON Lines (1行1JSONオブジェクト) のみを出力してください。説明やマークダウンは禁止。純粋な日本語のみ使用（中国語漢字禁止）。

出力例:
{"in":"今日めっちゃ眠い","x":"くそっ、俺は今日クソほど眠くて完全に限界みたいなんだよ。","k":"今日ほんまめっちゃ眠いわ、もうあかんねん。","t":"わー、今日たんげねぷてはんで、もうだめだびょん。"}
{"in":"このラーメンうまい","x":"クソ、このラーメンは俺が今まで食べた中で一番うまいみたいなんだよ。","k":"このラーメンめっちゃうまいやん！ほんま最高やで。","t":"このラーメンたんげめぇじゃ！わー感動したびょん。"}

各スタイルのルール（出力は12〜56文字程度のキレのある1文、改行禁止、元の文のキーワードを残す）:
1. "x" (Xの自動翻訳が海外ポストを暴走直訳した風):
   - 平和な内容なのに「くそっ、」「クソ、」「くそったれ、」「マジかよ、」「兄弟、」などが唐突に付く
   - 一人称「俺」、語尾「〜みたいなんだ」「〜なんだよ」「〜じゃないか」
   - ときどき「俺は一度だって〜なんて言ってないのに〜」「ちなみに俺が〜なのはこれを見たらわかるよ」「もう自分を見失っているんだよ」「クソほど〜」を混ぜる
2. "k" (関西弁・エセ関西弁):
   - 「めっちゃ」「ほんま」「〜やん」「〜やで」「〜やねん」「なんでやねん」「知らんけど」「あかんわ」「せやな」等
3. "t" (津軽弁・読めて味がある):
   - 「わー（私）」「たんげ（とても）」「〜だはんで（〜だから）」「〜だびょん / 〜だじゃ / 〜だべ」「めごい」「めぇ」「ねぷて」「しばれる」「けっぱれ」「へばな」「わいは！」等
"""

CHAT_SINGLE_SYS = """あなたはユーモラスな日本語チャットデータセット作成の達人です。
与えられたテーマについて、ユーザーからの自然な短い日本語メッセージ（2〜22文字）と、それに対する「Xの自動翻訳暴走口調（くそっ、俺は〜みたいなんだよ／兄弟、〜じゃないか／マジかよ 等）」の楽しくて的確な返事（12〜55文字、改行禁止）のペアを JSON Lines (1行1JSON) のみで出力してください。
説明・マークダウン・中国語漢字は厳禁です。ユーザーの質問や発言内容にきちんと寄り添って返事してください。

出力例:
{"u":"おはよう","a":"くそっ、おはよう兄弟！俺は今日も完全に目が覚めているみたいなんだよ。"}
{"u":"疲れた","a":"兄弟、今日一日本当によく頑張ったみたいだな。今すぐ休むべきなんだよ。"}
{"u":"論理ゲートって何？","a":"ANDやXORみたいな0と1のスイッチだけで計算する回路なんだよ。クソほど速いみたいなんだ。"}
"""

CHAT_MULTI_SYS = """あなたはユーモラスな日本語マルチターン会話データセット作成の達人です。
与えられたテーマについて、ユーザー(u)と「Xの自動翻訳暴走口調AI(a)」の2〜3ターンの自然な掛け合い会話を JSON Lines (1行1JSON) のみで出力してください。
各行は {"turns":[["ユーザー1","AI返事1"],["ユーザー2","AI返事2"]]} または3ターンの形式。
各発話は短くテンポよく（uは2〜22文字、aは12〜52文字、改行禁止）。前のターンの文脈をちゃんと受けて返事してください。
説明・マークダウン・中国語漢字は厳禁です。

出力例:
{"turns":[["おはよう","くそっ、おはよう兄弟！今日も完全に目が覚めているみたいだな。"],["まだめっちゃ眠いよ","マジかよ、それはクソほどわかるんだよ。今すぐ熱いコーヒーを飲むべきなんだよ。"],["了解、淹れてくる","いい判断だ兄弟！しっかり温まって今日も乗り切るみたいだな。"]]}
{"turns":[["腹減った","クソ、俺も今ラーメンかカレーが完全に食べたい気分なんだよ。何食べるつもりだ？"],["カレーにする","最高じゃないか兄弟！スパイスで気合を完全に入れ直すんだよ。"]]}
"""


def clean_line(s: str) -> str:
    s = s.replace("\r", " ").replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def is_clean_jp(s: str, min_len: int, max_len: int) -> bool:
    if not (min_len <= len(s) <= max_len):
        return False
    if BAD_CHARS_RE.search(s) or META_RE.search(s):
        return False
    if "=>" in s or "[X訳]" in s or "[関西]" in s or "[津軽]" in s or "[会話]" in s:
        return False
    if s.startswith("U:") or s.startswith("A:"):
        return False
    return True


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
            out.append(json.loads(m.group(0)))
        except Exception:
            continue
    return out


def gen_rewrite_batch(topic: str, count: int, batch_idx: int):
    prompt = (
        f"テーマ: 「{topic}」\n"
        f"バリエーション豊かな短い日本語文と3スタイル変換（x, k, t）を {count} 件、JSON Lines で出力してください。"
        f"（バッチ {batch_idx}: 同じ言い回しを避け、身近な口語・短い一言・日常文を混ぜてください）"
    )
    raw = call_llm(REWRITE_SYS, prompt, max_tokens=2600, temp=0.90)
    items = []
    for obj in parse_json_lines(raw):
        src = clean_line(str(obj.get("in", "")))
        x = clean_line(str(obj.get("x", "")))
        k = clean_line(str(obj.get("k", "")))
        t = clean_line(str(obj.get("t", "")))
        if not (is_clean_jp(src, 2, 36) and is_clean_jp(x, 8, 76) and is_clean_jp(k, 8, 76) and is_clean_jp(t, 8, 76)):
            continue
        items.append(("rw", "[X訳]", src, x))
        items.append(("rw", "[関西]", src, k))
        items.append(("rw", "[津軽]", src, t))
    return items


def gen_chat_single_batch(topic: str, count: int, batch_idx: int):
    prompt = (
        f"チャットテーマ: 「{topic}」\n"
        f"ユーザーの短いメッセージ(u)と、X自動翻訳風AIの楽しい返事(a)を {count} 件、JSON Lines で出力してください。"
        f"（バッチ {batch_idx}: 話し言葉や挨拶・ツッコミを多様に入れてください）"
    )
    raw = call_llm(CHAT_SINGLE_SYS, prompt, max_tokens=2200, temp=0.90)
    items = []
    for obj in parse_json_lines(raw):
        u = clean_line(str(obj.get("u", obj.get("in", ""))))
        a = clean_line(str(obj.get("a", obj.get("out", ""))))
        if not (is_clean_jp(u, 2, 36) and is_clean_jp(a, 8, 76)):
            continue
        items.append(("chat", [(u, a)]))
    return items


def gen_chat_multi_batch(topic: str, count: int, batch_idx: int):
    prompt = (
        f"マルチターン会話テーマ: 「{topic}」\n"
        f"2〜3ターンの掛け合い会話を {count} 本、JSON Lines で出力してください。"
        f"（バッチ {batch_idx}: 文脈がつながる自然な掛け合いにしてください）"
    )
    raw = call_llm(CHAT_MULTI_SYS, prompt, max_tokens=2600, temp=0.90)
    items = []
    for obj in parse_json_lines(raw):
        turns = obj.get("turns", [])
        if not isinstance(turns, list) or not (2 <= len(turns) <= 3):
            continue
        cleaned = []
        ok = True
        for pair in turns:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                ok = False
                break
            u = clean_line(str(pair[0]))
            a = clean_line(str(pair[1]))
            if not (is_clean_jp(u, 2, 34) and is_clean_jp(a, 8, 72)):
                ok = False
                break
            cleaned.append((u, a))
        if ok and cleaned:
            items.append(("chat", cleaned))
    return items


def format_record(rec) -> str:
    if rec[0] == "rw":
        _, mode, src, dst = rec
        return f"{mode} {src}\n=> {dst}\n\n"
    _, turns = rec
    lines = ["[会話]"]
    for u, a in turns:
        lines.append(f"U: {u}")
        lines.append(f"A: {a}")
    return "\n".join(lines) + "\n\n"


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    t0 = time.time()

    records = []
    for m, s, d in SEED_REWRITES:
        records.append(("rw", m, s, d))
    for turns in SEED_CHATS:
        records.append(("chat", turns))

    tasks = []
    for idx, (topic, total) in enumerate(TOPICS):
        chunk = 18
        n_batches = max(2, (total + chunk - 1) // chunk)
        per = max(12, total // n_batches)
        for b in range(n_batches):
            tasks.append(("rw", topic, per, idx * 10 + b + 1))

    for idx, (topic, total) in enumerate(CHAT_SINGLE_TOPICS):
        chunk = 22
        n_batches = max(2, (total + chunk - 1) // chunk)
        per = max(16, total // n_batches)
        for b in range(n_batches):
            tasks.append(("cs", topic, per, idx * 10 + b + 1))

    for idx, (topic, total) in enumerate(CHAT_MULTI_TOPICS):
        chunk = 14
        n_batches = max(3, (total + chunk - 1) // chunk)
        per = max(10, total // n_batches)
        for b in range(n_batches):
            tasks.append(("cm", topic, per, idx * 10 + b + 1))

    print(f"Dispatching {len(tasks)} batches to {MODEL_NAME} @ {API_URL} ({WORKERS} workers)...", flush=True)

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {}
        for kind, topic, cnt, bidx in tasks:
            fn = gen_rewrite_batch if kind == "rw" else (gen_chat_single_batch if kind == "cs" else gen_chat_multi_batch)
            futs[pool.submit(fn, topic, cnt, bidx)] = (kind, topic, bidx)

        for fut in concurrent.futures.as_completed(futs):
            kind, topic, bidx = futs[fut]
            done += 1
            try:
                rows = fut.result()
                records.extend(rows)
                if done % 10 == 0 or done == len(tasks):
                    print(
                        f"[{done}/{len(tasks)}] total_examples={len(records)} elapsed={time.time() - t0:.1f}s (last={kind}:{len(rows)})",
                        flush=True,
                    )
            except Exception as e:
                print(f"WARN batch ({kind}, {bidx}) failed: {e}", file=sys.stderr, flush=True)

    # Deduplicate by formatted text
    seen = set()
    unique = []
    for r in records:
        txt = format_record(r)
        if txt in seen:
            continue
        seen.add(txt)
        unique.append((r, txt))

    # Boost handcrafted anchors 3x so core demo inputs stay razor-sharp
    final_items = [txt for _, txt in unique]
    for _ in range(2):
        for m, s, d in SEED_REWRITES:
            final_items.append(format_record(("rw", m, s, d)))
        for turns in SEED_CHATS:
            final_items.append(format_record(("chat", turns)))

    rng = random.Random(42)
    rng.shuffle(final_items)

    counts = {"[X訳]": 0, "[関西]": 0, "[津軽]": 0, "[会話]": 0}
    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as f:
        for txt in final_items:
            for k in counts:
                if txt.startswith(k):
                    counts[k] += 1
                    break
            f.write(txt)

    size_kb = os.path.getsize(OUT_PATH) / 1024.0
    print(f"DONE: {len(final_items)} examples ({size_kb:.1f} KB) -> {OUT_PATH} in {time.time() - t0:.1f}s", flush=True)
    print("Per-mode counts:", counts, flush=True)


if __name__ == "__main__":
    main()
