#!/usr/bin/env python3
"""
OpenMythos ファインチューニングスクリプト — マーケティング領域4タスク対応

Phase 2 で作った前処理済み JSONL（data/processed/）を使って
社内ラベルデータでモデルを特化させる。

単一GPU:
    python scripts/finetune.py --data-dir data/processed/all --task all

タスク別:
    python scripts/finetune.py --data-dir data/processed/content_quality --task content_quality
    python scripts/finetune.py --data-dir data/processed/ad_performance  --task ad_performance

GPU調達後はそのまま CUDA で動作する（--device cuda）。
CPU でも動作するが速度は遅い（動作確認・小規模実験用）。
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from open_mythos.main import MythosConfig, OpenMythos
from open_mythos.variants import mythos_1b

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MarketingDataset(Dataset):
    def __init__(self, path: Path, max_length: int = 512):
        self.records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                ids = rec["input_ids"][:max_length]
                lbl = rec["labels"][:max_length]
                self.records.append(
                    {
                        "input_ids": ids,
                        "labels": lbl,
                        "task": rec.get("task", "unknown"),
                        "weight": rec.get("weight", 1.0),
                    }
                )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def collate_fn(batch: list[dict], pad_id: int = 0) -> dict[str, torch.Tensor]:
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    weights = torch.ones(len(batch))
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        weights[i] = b["weight"]
    return {"input_ids": input_ids, "labels": labels, "weights": weights}


# ---------------------------------------------------------------------------
# 学習率・設定
# ---------------------------------------------------------------------------

FINETUNE_CONFIGS = {
    "content_quality": dict(lr=3e-4, epochs=5, n_loops=8, batch=4),
    "ad_performance": dict(lr=3e-4, epochs=5, n_loops=8, batch=4),
    "persona_segment": dict(lr=2e-4, epochs=4, n_loops=6, batch=8),
    "market_research": dict(lr=2e-4, epochs=4, n_loops=12, batch=2),
    "all": dict(lr=2e-4, epochs=6, n_loops=8, batch=4),
}


# ---------------------------------------------------------------------------
# 訓練ループ
# ---------------------------------------------------------------------------


def train(args):
    device = torch.device(args.device)
    task_cfg = FINETUNE_CONFIGS.get(args.task, FINETUNE_CONFIGS["all"])
    n_loops = args.n_loops or task_cfg["n_loops"]
    lr = args.lr or task_cfg["lr"]
    epochs = args.epochs or task_cfg["epochs"]
    batch = args.batch or task_cfg["batch"]

    print(
        f"[finetune] task={args.task} device={device} loops={n_loops} lr={lr} epochs={epochs}"
    )

    # --- データ ---
    data_dir = Path(args.data_dir)
    train_ds = MarketingDataset(data_dir / "train.jsonl", args.max_length)
    val_ds = MarketingDataset(data_dir / "val.jsonl", args.max_length)
    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True, collate_fn=lambda b: collate_fn(b)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch, shuffle=False, collate_fn=lambda b: collate_fn(b)
    )
    print(f"  train={len(train_ds)} val={len(val_ds)}")

    # --- モデル ---
    if args.checkpoint:
        print(f"  loading checkpoint: {args.checkpoint}")
        cfg = mythos_1b() if args.model_size == "1b" else _small_config()
        model = OpenMythos(cfg).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        print("  initialising from scratch (no checkpoint)")
        cfg = _small_config()
        model = OpenMythos(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters")

    # --- オプティマイザ・スケジューラ ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 20),
        num_training_steps=total_steps,
    )

    # --- 訓練 ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, batch_data in enumerate(train_loader):
            input_ids = batch_data["input_ids"].to(device)
            labels = batch_data["labels"].to(device)
            weights = batch_data["weights"].to(device)

            logits = model(input_ids, n_loops=n_loops)  # (B, T, vocab)
            # タスク重み付き cross-entropy（SEO・広告を高め）
            loss_per_token = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="none",
            )
            # バッチ内の重みを適用
            seq_len = labels.size(1)
            w_expanded = weights.unsqueeze(1).expand(-1, seq_len).reshape(-1)
            valid = labels.view(-1) != -100
            loss = (
                loss_per_token * w_expanded * valid.float()
            ).sum() / valid.float().sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if (step + 1) % 10 == 0:
                avg = total_loss / (step + 1)
                ppl = math.exp(min(avg, 20))
                print(
                    f"  epoch {epoch} step {step+1}/{len(train_loader)} "
                    f"loss={avg:.4f} ppl={ppl:.1f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        # --- バリデーション ---
        val_loss = _evaluate(model, val_loader, device, n_loops)
        val_ppl = math.exp(min(val_loss, 20))
        elapsed = time.time() - t0
        print(
            f"[epoch {epoch}] train_loss={total_loss/len(train_loader):.4f} "
            f"val_loss={val_loss:.4f} val_ppl={val_ppl:.1f} ({elapsed:.0f}s)"
        )

        # --- チェックポイント保存 ---
        ckpt_path = out_dir / f"ckpt_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = out_dir / "best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"  ** best model saved -> {best_path}")

    print(f"\n訓練完了。ベストモデル: {out_dir / 'best.pt'}")
    print(
        f"推論サーバーで使うには MODEL_CHECKPOINT={out_dir / 'best.pt'} を設定してください。"
    )


@torch.no_grad()
def _evaluate(model, loader, device, n_loops) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch_data in loader:
        input_ids = batch_data["input_ids"].to(device)
        labels = batch_data["labels"].to(device)
        logits = model(input_ids, n_loops=n_loops)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += (labels != -100).sum().item()
    model.train()
    return total_loss / max(total_tokens, 1)


def _small_config() -> MythosConfig:
    return MythosConfig(
        vocab_size=50257,
        dim=256,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=512,
        max_loop_iters=16,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=8,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=128,
        act_threshold=0.99,
        lora_rank=8,
        kv_lora_rank=64,
        q_lora_rank=128,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="all",
        choices=[
            "content_quality",
            "ad_performance",
            "persona_segment",
            "market_research",
            "all",
        ],
    )
    parser.add_argument(
        "--data-dir", required=True, help="前処理済みJSONLのディレクトリ"
    )
    parser.add_argument(
        "--checkpoint", default="", help="初期チェックポイント（省略=ランダム初期化）"
    )
    parser.add_argument("--model-size", default="small", choices=["small", "1b"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--n-loops", type=int, default=None, help="訓練時のループ数")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--out-dir", default="checkpoints/finetune")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
