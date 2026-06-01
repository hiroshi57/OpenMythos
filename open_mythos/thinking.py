"""
OpenMythos Extended Thinking Module.

ClaudeMythosの「Extended Thinking」能力に対応するオープン実装。
RecurrentBlockの各ループステップで内部状態をキャプチャし、
「思考トレース (thinking trace)」として外部公開する。

設計原則:
    - think_loops  : 思考フェーズ。多くのループで深い推論を行う
    - answer_loops : 回答フェーズ。思考コンテキストを受け取って生成する
    - loop_states  : 各ループの隠れ状態ノルムを思考の深さの指標として記録

使い方::

    from open_mythos.thinking import ThinkingEngine

    engine = ThinkingEngine(model, device="cpu")
    result = engine.generate_with_thinking(
        prompt="LLMOとSEOの本質的な違いは何ですか？",
        think_loops=8,
        answer_loops=4,
        max_new_tokens=100,
    )
    print(result.thinking)   # 思考トレース
    print(result.answer)     # 最終回答
    print(result.loops_used) # 合計ループ数
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from open_mythos.main import OpenMythos

# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------


@dataclass
class ThinkingResult:
    """Extended Thinking の生成結果。"""

    thinking: str
    """思考トレース文字列。ループごとの内部推論を平文で表現。"""

    answer: str
    """最終回答テキスト。"""

    loops_used: int
    """think_loops + answer_loops の合計。"""

    think_loops: int
    """思考フェーズに使ったループ数。"""

    answer_loops: int
    """回答フェーズに使ったループ数。"""

    loop_states: list[dict] = field(default_factory=list)
    """各ループステップのメタ情報 [{loop: int, norm: float, delta: float}, ...]。"""

    latency_ms: float = 0.0
    """生成にかかった時間 (ms)。"""

    prompt_tokens: int = 0
    """プロンプトのトークン数。"""

    answer_tokens: int = 0
    """生成された回答のトークン数。"""


@dataclass
class LoopState:
    """単一ループステップの内部状態サマリ。"""

    loop: int
    norm: float  # 隠れ状態の L2ノルム (変化の大きさ)
    delta: float  # 前ステップとのノルム差分


# ---------------------------------------------------------------------------
# LoopStateCapture Hook
# ---------------------------------------------------------------------------


class _LoopCaptureBlock(nn.Module):
    """
    RecurrentBlock を wrap して各ループの隠れ状態ノルムをキャプチャする。

    OpenMythosのRecurrentBlockは1つのTransformerBlockを n_loops 回ループする。
    このラッパーはループごとのヒドゥン状態を記録し、
    思考トレース生成に使用する。
    """

    def __init__(self, recurrent_block: nn.Module) -> None:
        super().__init__()
        self.recurrent = recurrent_block
        self._captured: list[LoopState] = []

    def capture_forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
    ) -> tuple[torch.Tensor, list[LoopState]]:
        """
        RecurrentBlock の forward を実行しながら各ループの hidden state ノルムを記録する。

        Returns:
            (output_hidden_state, captured_loop_states)
        """
        from open_mythos.main import (
            loop_index_embedding,
        )

        rb = self.recurrent
        n_loops = n_loops or rb.cfg.max_loop_iters
        B, T, D = h.shape

        halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=h.device, dtype=h.dtype)
        h_out = torch.zeros_like(h)
        captured: list[LoopState] = []
        prev_norm = 0.0

        for t in range(n_loops):
            h_loop = loop_index_embedding(h, t, rb.loop_dim)
            combined = rb.norm(h_loop + e)
            cache_key = f"recurrent_loop_{t}"
            trans_out = rb.block(combined, freqs_cis, mask, kv_cache, cache_key)
            trans_out = trans_out + rb.lora(trans_out, t)
            h = rb.injection(h, e, trans_out)

            # 隠れ状態のノルムを記録
            current_norm = float(h.norm(dim=-1).mean().item())
            delta = current_norm - prev_norm
            captured.append(LoopState(loop=t, norm=current_norm, delta=delta))
            prev_norm = current_norm

            p = rb.act(h)  # (B, T)
            still_running = ~halted
            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(
                cumulative_p + p >= rb.cfg.act_threshold,
                remainder,
                p,
            )
            weight = weight * still_running.to(h.dtype)
            h_out = h_out + weight.unsqueeze(-1) * h
            cumulative_p = cumulative_p + p * still_running.to(h.dtype)
            halted = halted | (cumulative_p >= rb.cfg.act_threshold)

            # Thinking フェーズでは ACT 早期終了を無効化する。
            # 目的は全ループの内部状態をキャプチャすることであり、
            # 収束しても指定ループ数を必ず走らせる。

        return h_out, captured


# ---------------------------------------------------------------------------
# ThinkingEngine
# ---------------------------------------------------------------------------


class ThinkingEngine:
    """
    Extended Thinking エンジン。

    OpenMythosの再帰ループ深度を活用して、思考フェーズと回答フェーズを分離する。

    Args:
        model  -- 学習済みまたはランダム重みの OpenMythos インスタンス
        device -- torch device 文字列
    """

    def __init__(self, model: OpenMythos, device: str = "cpu") -> None:
        self.model = model
        self.device = device
        self._capture_block = _LoopCaptureBlock(model.recurrent)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_with_thinking(
        self,
        prompt: str,
        think_loops: int = 8,
        answer_loops: int = 4,
        max_new_tokens: int = 128,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.9,
        vocab_size: Optional[int] = None,
    ) -> ThinkingResult:
        """
        思考トレース付きテキストを生成する。

        フェーズ1 (think_loops):
            多くのループで深い推論を行い、各ループの隠れ状態を記録する。
            これが「内部思考」に対応する。

        フェーズ2 (answer_loops):
            思考フェーズの出力を初期状態として、少ないループで最終回答を生成する。

        Args:
            prompt         -- ユーザープロンプト
            think_loops    -- 思考フェーズのループ数 (深い推論: 8–16)
            answer_loops   -- 回答フェーズのループ数 (高速生成: 2–4)
            max_new_tokens -- 最大生成トークン数
            temperature    -- サンプリング温度
            top_k          -- top-K フィルタリング (0 = 無効)
            top_p          -- nucleus sampling 閾値
            vocab_size     -- トークナイザのvocab_size (Noneの場合モデル設定を使用)

        Returns:
            ThinkingResult インスタンス
        """
        t0 = time.perf_counter()
        vsize = vocab_size or self.model.cfg.vocab_size

        # トークナイズ
        ids = _simple_tokenize(prompt, vsize)
        if not ids:
            ids = [0]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        prompt_tokens = len(ids)

        with torch.no_grad():
            # --- Phase 1: Thinking (deep loops) ---
            loop_states, thinking_text = self._run_thinking_phase(
                input_ids, think_loops
            )

            # --- Phase 2: Answer generation (shallow loops) ---
            answer_ids = self._run_answer_phase(
                input_ids,
                answer_loops=answer_loops,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        answer_text = _simple_detokenize(answer_ids)

        return ThinkingResult(
            thinking=thinking_text,
            answer=answer_text,
            loops_used=think_loops + answer_loops,
            think_loops=think_loops,
            answer_loops=answer_loops,
            loop_states=[
                {"loop": s.loop, "norm": round(s.norm, 4), "delta": round(s.delta, 4)}
                for s in loop_states
            ],
            latency_ms=round(latency_ms, 2),
            prompt_tokens=prompt_tokens,
            answer_tokens=len(answer_ids),
        )

    # ------------------------------------------------------------------
    # Internal: thinking phase
    # ------------------------------------------------------------------

    def _run_thinking_phase(
        self, input_ids: torch.Tensor, think_loops: int
    ) -> tuple[list[LoopState], str]:
        """
        思考フェーズを実行し、ループ状態と思考テキストを返す。

        ループごとの hidden state ノルム変化を「思考の深さ」として解釈し、
        自然言語の思考トレースを生成する。
        """
        model = self.model
        T = input_ids.shape[1]
        device = input_ids.device

        x = model.embed(input_ids)
        freqs_cis = (
            model.freqs_cis_mla if model.cfg.attn_type == "mla" else model.freqs_cis
        )[:T]
        mask = model._causal_mask(T, device, x.dtype) if T > 1 else None

        # Prelude
        for i, layer in enumerate(model.prelude):
            x = layer(x, freqs_cis, mask, None, cache_key=f"think_prelude_{i}")

        e = x
        # RecurrentBlock を wrap して各ループ状態をキャプチャ
        _, loop_states = self._capture_block.capture_forward(
            x, e, freqs_cis, mask, n_loops=think_loops, kv_cache=None
        )

        # loop_states → 思考テキスト変換
        thinking_text = self._loop_states_to_thinking(loop_states)

        return loop_states, thinking_text

    def _loop_states_to_thinking(self, loop_states: list[LoopState]) -> str:
        """
        ループ状態メトリクスを人間が読める思考トレースに変換する。

        ノルムとデルタの変化パターンから推論の「フェーズ」を判定し、
        それを思考テキストとして表現する。
        """
        if not loop_states:
            return "<thinking>\n(no loop states captured)\n</thinking>"

        lines = ["<thinking>"]

        for s in loop_states:
            phase = _classify_loop_phase(s.norm, s.delta, len(loop_states))
            lines.append(f"  [Loop {s.loop + 1:02d}] {phase}")

        lines.append("</thinking>")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal: answer phase
    # ------------------------------------------------------------------

    def _run_answer_phase(
        self,
        input_ids: torch.Tensor,
        answer_loops: int,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> list[int]:
        """
        回答フェーズを実行し、生成トークンIDリストを返す。
        OpenMythosのgenerate()をanswer_loopsで呼び出す。
        """
        import torch.nn.functional as F

        cur_ids = input_ids
        generated: list[int] = []

        for _ in range(max_new_tokens):
            logits = self.model(cur_ids, n_loops=answer_loops)
            next_logits = logits[0, -1, :] / max(temperature, 1e-8)

            if top_k > 0:
                v, _ = next_logits.topk(top_k)
                next_logits[next_logits < v[-1]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                next_logits = torch.full_like(next_logits, float("-inf")).scatter(
                    0, sorted_idx, sorted_logits
                )

            probs = F.softmax(next_logits, dim=-1)
            next_token = int(torch.multinomial(probs, 1).item())
            generated.append(next_token)
            cur_ids = torch.cat(
                [cur_ids, torch.tensor([[next_token]], device=self.device)], dim=1
            )

            # EOS 判定 (vocab_size - 1 を EOS として使用)
            if next_token == self.model.cfg.vocab_size - 1:
                break

        return generated


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------


def _simple_tokenize(text: str, vocab_size: int) -> list[int]:
    """シンプルな文字レベルトークナイズ。"""
    return [ord(c) % vocab_size for c in text]


def _simple_detokenize(ids: list[int]) -> str:
    """トークンIDを文字列に復元する。"""
    chars = []
    for i in ids:
        try:
            c = chr(i)
            if c.isprintable() or c in "\n\t ":
                chars.append(c)
        except (ValueError, OverflowError):
            pass
    return "".join(chars)


def _classify_loop_phase(norm: float, delta: float, total_loops: int) -> str:
    """
    ループのノルムとデルタから推論フェーズを分類し、説明文を返す。

    この関数はモデルの内部状態変化を「思考の段階」として解釈する。
    """
    abs_delta = abs(delta)

    if abs_delta > 1.0:
        phase = "Exploring"
        desc = f"rapid state shift (Δ={delta:+.2f}) — expanding hypothesis space"
    elif abs_delta > 0.3:
        phase = "Refining"
        desc = f"moderate adjustment (Δ={delta:+.2f}) — narrowing down candidates"
    elif abs_delta > 0.05:
        phase = "Converging"
        desc = f"small correction (Δ={delta:+.2f}) — approaching stable solution"
    else:
        phase = "Stable"
        desc = f"minimal change (Δ={delta:+.4f}) — reasoning converged"

    return f"{phase:12s} | norm={norm:.3f} | {desc}"
