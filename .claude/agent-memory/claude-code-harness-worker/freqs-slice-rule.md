---
name: freqs-slice-rule
description: GQA/MLA attention に渡す freqs_cis は必ず [:T] スライスする
metadata:
  type: feedback
---

`precompute_rope_freqs(dim, max_seq_len)` は形状 `(max_seq_len, dim//2)` のテンソルを返す。
`GQAttention` / `MLAttention` / `TransformerBlock` / `RecurrentBlock` は **現在処理中の系列長 T** にスライス済みの `(T, dim//2)` を期待する。

全長のまま渡すと `apply_rope` 内でブロードキャストエラー:
```
RuntimeError: The size of tensor a (8) must match tensor b (32) at non-singleton dimension 1
```

**Why:** `OpenMythos.forward` は内部で `freqs_cis[start_pos:start_pos+T]` とスライスしてから各レイヤーに渡しているが、
ユニットテストが直接 attention を呼ぶ場合は呼び出し元がスライスする責任を持つ。

**How to apply:**
- テストの `setup_method` で `self.freqs = precompute_rope_freqs(...)[:T]` とプリスライス
- 直接 block を呼ぶ場合は `block(x, freqs[:T])` のように必ず `[:T]` を付ける
- `OpenMythos.forward` 経由なら内部でスライスするため不要
