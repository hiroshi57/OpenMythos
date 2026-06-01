#!/usr/bin/env python3
"""
OpenMythos DPO (Direct Preference Optimization) Fine-tuning。

クロード系モデル (ClaudeMythosアーキテクチャ) の特徴である
Constitutional AI / RLHF に対応するオープン実装。

DPO は強化学習を使わずに「chosen (好まれる回答)」と「rejected (好まれない回答)」の
ペアデータから直接 preference を学習する手法 (Rafailov et al., 2023)。

損失関数:
    L_DPO = -E[log σ(β · (log π(chosen) - log π_ref(chosen))
                       - β · (log π(rejected) - log π_ref(rejected)))]

データ形式 (JSONL):
    {"prompt": "...", "chosen": "...", "rejected": "..."}

使い方::

    python scripts/train_dpo.py \\
        --data data/preference_pairs.jsonl \\
        --checkpoint models/mythos_finetune.pt \\
        --out-dir models/dpo \\
        --epochs 3 \\
        --beta 0.1

テスト用サンプルデータ生成::

    python scripts/train_dpo.py --generate-sample-data
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent.parent))

from open_mythos.main import MythosConfig, OpenMythos


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------


@dataclass
class DPOConfig:
    """DPO学習設定。"""

    beta: float = 0.1
    """DPO の β パラメータ。小さいほど参照モデルに近づく。"""

    learning_rate: float = 5e-6
    """学習率。DPO はSFTより低い lr が安定する。"""

    batch_size: int = 1
    """バッチサイズ。GPU メモリに合わせて調整。"""

    epochs: int = 3
    """エポック数。"""

    max_seq_len: int = 256
    """最大シーケンス長。"""

    weight_decay: float = 0.01
    """AdamW 正則化。"""

    warmup_steps: int = 10
    """線形 warmup ステップ数。"""

    label_smoothing: float = 0.0
    """ラベルスムージング (0.0 = 無効)。"""

    log_every: int = 5
    """ログ出力間隔 (ステップ数)。"""


@dataclass
class PreferencePair:
    """好み比較ペア。"""

    prompt: str
    chosen: str
    rejected: str


# ---------------------------------------------------------------------------
# DPO 損失
# ---------------------------------------------------------------------------


def compute_dpo_loss(
    policy_model: OpenMythos,
    reference_model: OpenMythos,
    batch: list[PreferencePair],
    beta: float,
    max_seq_len: int,
    device: str,
) -> tuple[torch.Tensor, dict]:
    """
    DPO 損失を計算する。

    Args:
        policy_model    -- 学習対象モデル
        reference_model -- 参照モデル (frozen)
        batch           -- preference pair のバッチ
        beta            -- DPO の温度パラメータ
        max_seq_len     -- 最大シーケンス長
        device          -- torch device

    Returns:
        (loss, metrics_dict)
    """
    vocab_size = policy_model.cfg.vocab_size

    def _encode(text: str) -> torch.Tensor:
        """テキストをトークンIDテンソルに変換する。"""
        ids = [ord(c) % vocab_size for c in text[:max_seq_len]]
        if not ids:
            ids = [0]
        return torch.tensor(ids, dtype=torch.long, device=device)

    def _log_prob(model: OpenMythos, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """モデルの対数確率を計算する。"""
        # prompt + response を concat してフォワード
        all_ids = torch.cat([input_ids, target_ids]).unsqueeze(0)
        T_prompt = input_ids.shape[0]
        T_total = all_ids.shape[1]

        logits = model(all_ids)  # (1, T_total, vocab_size)

        # response 部分の対数確率のみ計算
        # log_probs shape: (T_response, vocab_size)
        log_probs = F.log_softmax(logits[0, T_prompt - 1 : T_total - 1, :], dim=-1)
        # gather: dim=1 (vocab 次元) で各 target トークンの log prob を取得
        token_log_probs = log_probs.gather(
            1, target_ids.unsqueeze(1)
        ).squeeze(1)  # (T_response,)
        return token_log_probs.sum()

    policy_chosen_logps: list[torch.Tensor] = []
    policy_rejected_logps: list[torch.Tensor] = []
    ref_chosen_logps: list[torch.Tensor] = []
    ref_rejected_logps: list[torch.Tensor] = []

    for pair in batch:
        prompt_ids = _encode(pair.prompt)
        chosen_ids = _encode(pair.chosen)
        rejected_ids = _encode(pair.rejected)

        # Policy model の対数確率
        policy_chosen_logps.append(_log_prob(policy_model, prompt_ids, chosen_ids))
        policy_rejected_logps.append(_log_prob(policy_model, prompt_ids, rejected_ids))

        # Reference model の対数確率 (勾配不要)
        with torch.no_grad():
            ref_chosen_logps.append(_log_prob(reference_model, prompt_ids, chosen_ids))
            ref_rejected_logps.append(_log_prob(reference_model, prompt_ids, rejected_ids))

    # スタック
    policy_chosen = torch.stack(policy_chosen_logps)      # (B,)
    policy_rejected = torch.stack(policy_rejected_logps)  # (B,)
    ref_chosen = torch.stack(ref_chosen_logps)            # (B,)
    ref_rejected = torch.stack(ref_rejected_logps)        # (B,)

    # DPO 損失
    chosen_reward = beta * (policy_chosen - ref_chosen)
    rejected_reward = beta * (policy_rejected - ref_rejected)
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()

    # メトリクス
    with torch.no_grad():
        chosen_rewards_mean = float(chosen_reward.mean().item())
        rejected_rewards_mean = float(rejected_reward.mean().item())
        reward_margin = chosen_rewards_mean - rejected_rewards_mean
        accuracy = float((chosen_reward > rejected_reward).float().mean().item())

    metrics = {
        "loss": float(loss.item()),
        "chosen_reward": chosen_rewards_mean,
        "rejected_reward": rejected_rewards_mean,
        "reward_margin": reward_margin,
        "accuracy": accuracy,
    }

    return loss, metrics


# ---------------------------------------------------------------------------
# データローダー
# ---------------------------------------------------------------------------


def load_preference_data(path: Path) -> list[PreferencePair]:
    """JSONL 形式の preference データを読み込む。"""
    pairs: list[PreferencePair] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                pairs.append(PreferencePair(
                    prompt=record["prompt"],
                    chosen=record["chosen"],
                    rejected=record["rejected"],
                ))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: line {line_no} skipped ({e})")
    return pairs


def generate_sample_data(out_path: Path, n: int = 20) -> None:
    """テスト用サンプル preference データを生成する。"""
    templates = [
        {
            "prompt": "LLMOとSEOの違いを説明してください",
            "chosen": "LLMOは AI検索エンジンへの最適化、SEOは従来の検索エンジンへの最適化です。LLMOではエンティティ密度と回答直接性が重視されます。",
            "rejected": "SEOとLLMOはどちらも大切です。詳しくは専門家にお聞きください。",
        },
        {
            "prompt": "CTRを改善するには？",
            "chosen": "CTR改善には3つのアプローチが有効です: (1) タイトルにキーワードを含める (2) メタディスクリプションを30字以内に要約 (3) CTAボタンを目立たせる。",
            "rejected": "CTRはクリック率です。改善できると思います。",
        },
        {
            "prompt": "コンテンツのLLMOスコアを上げるには？",
            "chosen": "LLMOスコア向上には: entity_density を 15/100語以上、answer_directness を冒頭1文で確保、citability には統計データと出典を含めることが効果的です。",
            "rejected": "良いコンテンツを書けばLLMOスコアが上がります。",
        },
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(n):
            record = templates[i % len(templates)].copy()
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Generated {n} preference pairs → {out_path}")


# ---------------------------------------------------------------------------
# 学習ループ
# ---------------------------------------------------------------------------


def train_dpo(
    policy_model: OpenMythos,
    reference_model: OpenMythos,
    data: list[PreferencePair],
    cfg: DPOConfig,
    device: str,
    out_dir: Path,
) -> dict:
    """
    DPO学習を実行する。

    Args:
        policy_model    -- 学習対象モデル (勾配有効)
        reference_model -- 参照モデル (frozen)
        data            -- 学習データ
        cfg             -- DPO設定
        device          -- torch device
        out_dir         -- モデル保存ディレクトリ

    Returns:
        学習結果サマリ dict
    """
    policy_model.train()
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad_(False)

    optimizer = AdamW(
        policy_model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # 線形 warmup スケジューラ
    total_steps = math.ceil(len(data) / cfg.batch_size) * cfg.epochs

    def _lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(cfg.warmup_steps, 1)
        return max(0.0, 1.0 - (step - cfg.warmup_steps) / max(total_steps - cfg.warmup_steps, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    global_step = 0
    best_margin = float("-inf")
    t_start = time.perf_counter()

    print(f"DPO Training: {len(data)} pairs × {cfg.epochs} epochs")
    print(f"  β={cfg.beta}  lr={cfg.learning_rate}  batch={cfg.batch_size}")
    print(f"  device={device}  max_seq_len={cfg.max_seq_len}")
    print("-" * 50)

    for epoch in range(cfg.epochs):
        # データをシャッフル
        import random
        random.shuffle(data)

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        n_batches = 0

        for i in range(0, len(data), cfg.batch_size):
            batch = data[i : i + cfg.batch_size]

            optimizer.zero_grad()
            loss, metrics = compute_dpo_loss(
                policy_model=policy_model,
                reference_model=reference_model,
                batch=batch,
                beta=cfg.beta,
                max_seq_len=cfg.max_seq_len,
                device=device,
            )
            loss.backward()
            # 勾配クリッピング (DPO は勾配が大きくなりやすい)
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += metrics["loss"]
            epoch_accuracy += metrics["accuracy"]
            n_batches += 1
            global_step += 1

            if global_step % cfg.log_every == 0 or global_step == 1:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"step={global_step:4d} | epoch={epoch+1}/{cfg.epochs} | "
                    f"loss={metrics['loss']:.4f} | acc={metrics['accuracy']:.2%} | "
                    f"margin={metrics['reward_margin']:.4f} | lr={lr_now:.2e}"
                )
                history.append({
                    "step": global_step,
                    "epoch": epoch + 1,
                    **metrics,
                    "lr": lr_now,
                })

                if metrics["reward_margin"] > best_margin:
                    best_margin = metrics["reward_margin"]
                    ckpt_path = out_dir / "dpo_best.pt"
                    torch.save({
                        "model": policy_model.state_dict(),
                        "cfg": policy_model.cfg,
                        "step": global_step,
                        "reward_margin": best_margin,
                    }, ckpt_path)

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc = epoch_accuracy / max(n_batches, 1)
        print(f"  Epoch {epoch+1} done: avg_loss={avg_loss:.4f}  avg_acc={avg_acc:.2%}")

    elapsed = time.perf_counter() - t_start
    final_ckpt = out_dir / "dpo_final.pt"
    torch.save({
        "model": policy_model.state_dict(),
        "cfg": policy_model.cfg,
        "step": global_step,
        "history": history,
    }, final_ckpt)

    summary = {
        "total_steps": global_step,
        "elapsed_sec": round(elapsed, 2),
        "best_reward_margin": round(best_margin, 4),
        "final_checkpoint": str(final_ckpt),
        "best_checkpoint": str(out_dir / "dpo_best.pt"),
    }
    print("-" * 50)
    print(f"DPO Training done: {global_step} steps in {elapsed:.1f}s")
    print(f"  Best reward margin: {best_margin:.4f}")
    print(f"  Saved: {final_ckpt}")
    return summary


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _build_tiny_model(device: str) -> OpenMythos:
    """テスト用の最小モデルを構築する。"""
    cfg = MythosConfig(
        vocab_size=512,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=256,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=1,
        expert_dim=32,
        lora_rank=4,
    )
    return OpenMythos(cfg).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenMythos DPO Fine-tuning"
    )
    parser.add_argument(
        "--data",
        default="",
        help="Preference ペア JSONL ファイルパス",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="ベースモデルのチェックポイント (.pt)。省略時はランダム重み",
    )
    parser.add_argument(
        "--out-dir",
        default="models/dpo",
        help="保存先ディレクトリ",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument(
        "--generate-sample-data",
        action="store_true",
        help="テスト用サンプルデータを生成して終了",
    )
    parser.add_argument(
        "--sample-data-path",
        default="data/preference_pairs.jsonl",
        help="サンプルデータの出力パス",
    )
    args = parser.parse_args()

    if args.generate_sample_data:
        generate_sample_data(Path(args.sample_data_path))
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # モデル構築
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model_cfg: MythosConfig = ckpt["cfg"]
        policy_model = OpenMythos(model_cfg).to(device)
        policy_model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        policy_model = _build_tiny_model(device)
        print("Using random-weight model (no checkpoint)")

    # 参照モデルはポリシーモデルのコピー
    reference_model = copy.deepcopy(policy_model).to(device)

    # データ読み込み
    if args.data and os.path.exists(args.data):
        data = load_preference_data(Path(args.data))
        print(f"Loaded {len(data)} preference pairs from {args.data}")
    else:
        print("No data file specified — using auto-generated sample data")
        sample_path = Path("data/preference_pairs.jsonl")
        generate_sample_data(sample_path, n=20)
        data = load_preference_data(sample_path)

    # DPO 設定
    dpo_cfg = DPOConfig(
        beta=args.beta,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_seq_len=args.max_seq_len,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
    )

    # 学習実行
    train_dpo(
        policy_model=policy_model,
        reference_model=reference_model,
        data=data,
        cfg=dpo_cfg,
        device=device,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
