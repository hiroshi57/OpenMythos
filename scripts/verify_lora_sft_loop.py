#!/usr/bin/env python3
"""Sprint 37.4 -- SelfDistillLoop x LoRA SFT end-to-end verification"""
from open_mythos.self_distill import SelfDistillLoop, SelfDistillConfig

# 1. for_lora_sft() factory
loop = SelfDistillLoop.for_lora_sft(n_rounds=2, score_threshold=0.3, samples_per_round=3)
assert loop.lora_sft_enabled, "lora_sft_enabled should be True"
assert loop.cfg.sft_backend == "lora", "sft_backend should be lora"
assert loop._lora_trainer is not None, "LoraTrainer should be auto-created"
print("[OK] for_lora_sft() factory: lora_sft_enabled=True, LoraTrainer auto-created")

# 2. run() with sft_backend="lora" actually calls LoraTrainer.train()
result = loop.run(["SEO対策の基本とは？", "コンテンツ品質を高める方法"], n_iterations=1)
assert result.rounds_completed >= 1, "at least 1 round should complete"
r0 = result.round_results[0]
assert r0.sft_result is not None, "sft_result should not be None when samples pass filter"
print(f"[OK] LoRA SFT ran: round={r0.round_num}, "
      f"samples={r0.filtered}, "
      f"train_loss={r0.sft_result.train_loss:.4f}, "
      f"eval_score={r0.sft_result.eval_score:.4f}")

# 3. Direct cfg route: sft_backend="lora" without factory
cfg = SelfDistillConfig(n_rounds=1, score_threshold=0.1, sft_backend="lora", sft_simulate=False)
loop2 = SelfDistillLoop(cfg=cfg)
assert loop2.lora_sft_enabled, "direct cfg route should also enable lora"
print("[OK] Direct SelfDistillConfig(sft_backend='lora') route works")

# 4. sft_simulate=True path still works (backward compat)
cfg_sim = SelfDistillConfig(n_rounds=1, score_threshold=0.1, sft_backend="simulate", sft_simulate=True)
loop3 = SelfDistillLoop(cfg=cfg_sim)
assert not loop3.lora_sft_enabled, "simulate mode should not be lora"
res3 = loop3.run(["test"], n_iterations=1)
print("[OK] Simulate fallback path still works")

print("\nSprint 37.4 verification PASSED")
