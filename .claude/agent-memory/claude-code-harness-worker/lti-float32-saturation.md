---
name: lti-float32-saturation
description: LTIInjection.get_A() が float32 で 1.0 に飽和する問題と対処
metadata:
  type: feedback
---

`get_A()` の実装:
```python
return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))
```

大きな学習率（lr=1e3 など）で `log_dt + log_A` が `-20` 以下にクランプされると:
- `exp(-20) ≈ 2e-9`（ほぼ 0）
- `-exp(-20) ≈ 0`
- `exp(0) = 1.0`（float32 で完全に 1.0 に丸まる）

これは `ρ(A) < 1` の理論的保証を float32 精度で破る。

**Why:** float32 の精度は約 7 桁。`exp(-2e-9) = 0.9999999979...` は float32 で `1.0` に丸まる。

**How to apply:** inner exp に `.clamp(min=1e-6)` を追加して修正済み:
```python
return torch.exp(
    -torch.exp((self.log_dt + self.log_A).clamp(-20, 20)).clamp(min=1e-6)
)
```
これにより `A = exp(-1e-6) ≈ 0.9999990` が保証され、float32 でも `< 1.0` が成立する。
