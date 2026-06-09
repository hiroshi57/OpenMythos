"""
quick_train.py — OpenMythos 広告コピー用クイック学習スクリプト
実行方法: python scripts/quick_train.py
所要時間: CPU で約 5〜15 分（dim=256 の小型モデル）
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from open_mythos.main import MythosConfig, OpenMythos

# ── 広告コピー学習データ ──────────────────────────────────────────
AD_CORPUS = [
    # 化粧品・美容
    "溶ける前に、全部食べた。夏の罪悪感ごと、おいしい。",
    "守られてる感ゼロ。でも、守られてる。",
    "塗って、忘れて、思い切り楽しむ。",
    "素肌のままで、全部やる。",
    "紫外線より、思い出を増やそう。",
    "肌を守る。それだけで、全部変わる。",
    "アウトドア派の、静かな自信。",
    "日差しも、人生も、全力で浴びる。",
    # ファッション
    "着るたびに、自分が好きになる。",
    "シンプルだから、あなたが引き立つ。",
    "今日の気分を、そのまま着てほしい。",
    "流行より、自分らしさを選んだ。",
    "一枚で、気持ちが変わる。",
    # 食品・飲料
    "一口で、旅に出る。",
    "毎朝の贅沢を、あたりまえに。",
    "忙しい日も、この一杯だけは丁寧に。",
    "体が喜ぶ、本物の味。",
    "素材の声を、そのまま届ける。",
    # デジタル・テクノロジー
    "考える前に、もう動いてる。",
    "あなたの時間を、もっと自分のために。",
    "難しいことを、簡単に。",
    "つながることで、もっと自由に。",
    "未来を、今日から始める。",
    # 健康・フィットネス
    "体を動かすたびに、人生が変わる。",
    "今日の一歩が、明日の自信になる。",
    "頑張らなくていい。続けるだけでいい。",
    "あなたのペースで、あなたらしく。",
    # 旅行・観光
    "知らない街で、本当の自分に会う。",
    "旅は、答えではなく問いをくれる。",
    "また来たいと思える場所が、ある。",
    "非日常が、日常をリセットしてくれる。",
]

def main():
    print("=" * 55)
    print("OpenMythos クイック学習スクリプト")
    print("=" * 55)

    # ── 設定 ────────────────────────────────────────────────────────
    CHECKPOINT_PATH = "models/ad_model.pt"
    STEPS = 400
    LR = 3e-4
    SEQ_LEN = 64
    BATCH = 4

    os.makedirs("models", exist_ok=True)

    device = torch.device("cpu")
    print(f"デバイス: {device}")
    print(f"学習ステップ数: {STEPS}")
    print(f"データ件数: {len(AD_CORPUS)} 文")
    print()

    # ── トークナイザー ────────────────────────────────────────────
    print("トークナイザーを読み込み中...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # ── モデル ───────────────────────────────────────────────────
    print("モデルを初期化中...")
    # ★ サーバーの _build_config と完全に一致させる
    cfg = MythosConfig(
        vocab_size=50257,
        dim=256,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=512,      # サーバーと同じ
        max_loop_iters=16,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=8,          # サーバーと同じ
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=128,
        act_threshold=0.99,
        lora_rank=8,          # サーバーと同じ
        kv_lora_rank=64,
        q_lora_rank=128,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
    )
    model = OpenMythos(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"パラメータ数: {n_params:,}")

    # ── データ準備 ─────────────────────────────────────────────────
    print("データを準備中...")
    all_ids = []
    for text in AD_CORPUS:
        ids = tokenizer.encode(text, add_special_tokens=True)
        all_ids.extend(ids)

    # シーケンスに分割
    sequences = []
    for i in range(0, len(all_ids) - SEQ_LEN, SEQ_LEN // 2):
        seq = all_ids[i:i + SEQ_LEN + 1]
        if len(seq) == SEQ_LEN + 1:
            sequences.append(seq)

    # データが少ない場合は繰り返し
    while len(sequences) < BATCH * 10:
        sequences = sequences * 3

    print(f"シーケンス数: {len(sequences)}")
    print()

    # ── 学習 ──────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()

    print("学習開始...")
    print("-" * 40)
    start = time.time()

    for step in range(1, STEPS + 1):
        # ランダムにバッチを選択
        indices = torch.randint(len(sequences), (BATCH,))
        batch = torch.tensor([sequences[i] for i in indices])
        x = batch[:, :-1].to(device)
        y = batch[:, 1:].to(device)

        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            y.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0:
            elapsed = time.time() - start
            eta = elapsed / step * (STEPS - step)
            print(f"  step {step:4d}/{STEPS}  loss={loss.item():.4f}  "
                  f"経過{elapsed:.0f}s  残り約{eta:.0f}s")

    print("-" * 40)
    total = time.time() - start
    print(f"\n学習完了！ 合計時間: {total:.0f}秒")

    # ── 保存 ──────────────────────────────────────────────────────
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"\nチェックポイント保存: {CHECKPOINT_PATH}")

    # ── .env 更新ガイド ────────────────────────────────────────────
    abs_path = os.path.abspath(CHECKPOINT_PATH)
    print()
    print("=" * 55)
    print("次のステップ:")
    print("=" * 55)
    print(f"1. .env に以下を追加してください:")
    print(f"   MODEL_CHECKPOINT={abs_path}")
    print(f"   MODEL_DIM=256")
    print(f"   MODEL_ATTN=gqa")
    print()
    print("2. サーバーを再起動してください:")
    print("   Ctrl+C でサーバーを止めて、再度:")
    print("   uvicorn serve.api:app --host 0.0.0.0 --port 8000 --reload")
    print()
    print("3. 生成テスト:")
    print("   python examples/ad_demo.py")
    print("=" * 55)

    # 簡単な生成テスト
    print("\n学習直後のテスト生成:")
    model.eval()
    prompt = "夏の広告"
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=30, n_loops=4)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"入力: {prompt}")
    print(f"出力: {text}")

if __name__ == "__main__":
    main()
