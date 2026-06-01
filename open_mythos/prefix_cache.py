"""
OpenMythos Prompt KV Prefix Cache。

共通のシステムプロンプトや Few-shot 例を事前に prefill し、
その KV 状態をキャッシュして後続リクエストの推論を高速化する。

設計:
    PrefixCacheEntry  -- キャッシュエントリ (prefix_ids + cached_logits)
    PromptPrefixCache -- LRU キャッシュ管理 + ヒット率追跡
    CachedGenResult   -- キャッシュ使用時の生成結果

使い方::

    from open_mythos.prefix_cache import PromptPrefixCache

    model = OpenMythos(cfg)
    cache = PromptPrefixCache(model, max_entries=32)

    # 共通システムプロンプトをキャッシュ
    system_prompt = "あなたは優秀なマーケティングアシスタントです。"
    cache.cache_prefix(system_prompt)

    # キャッシュを活用して生成（以降のリクエストで prefix prefill をスキップ）
    result = cache.generate_with_cache(
        prompt=system_prompt + "\\nCTRを最大化する広告コピーを書いて",
        max_new_tokens=60,
    )
    print(result.text, f"  cache_hit={result.cache_hit}")
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from open_mythos.main import OpenMythos

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------


@dataclass
class PrefixCacheEntry:
    """キャッシュされたプレフィックスエントリ。"""

    prefix_text: str
    """キャッシュした元テキスト。"""

    prefix_ids: list[int]
    """トークン ID リスト。"""

    cached_logits: torch.Tensor
    """prefill 時の最終ロジット。shape: (vocab_size,)"""

    n_loops: int
    """キャッシュ時の推論ループ数。"""

    created_at: float = field(default_factory=time.perf_counter)
    hits: int = 0

    @property
    def prefix_len(self) -> int:
        return len(self.prefix_ids)

    @property
    def cache_key(self) -> str:
        return hashlib.md5(self.prefix_text.encode()).hexdigest()[:16]


@dataclass
class CachedGenResult:
    """キャッシュを活用した生成結果。"""

    text: str
    """生成テキスト (prefix 以降)。"""

    prompt_used: str
    """実際に使用したプロンプト。"""

    cache_hit: bool
    """キャッシュを使用したか。"""

    cache_key: str
    """使用/作成したキャッシュエントリのキー。"""

    prefix_len: int
    """キャッシュした prefix の長さ (文字数)。"""

    latency_ms: float = 0.0
    prefill_skipped_tokens: int = 0
    """キャッシュヒットにより省略した prefill トークン数。"""


# ---------------------------------------------------------------------------
# PromptPrefixCache
# ---------------------------------------------------------------------------


class PromptPrefixCache:
    """
    LRU プロンプトプレフィックスキャッシュ。

    システムプロンプトや共通プレフィックスを事前に prefill し、
    そのロジット状態を保持する。

    重複リクエストでは prefix の prefill を省略することで
    TTFT (Time To First Token) を削減できる。

    Args:
        model       -- OpenMythos モデルインスタンス
        max_entries -- LRU キャッシュの最大エントリ数 (デフォルト: 32)
        device      -- torch device
    """

    def __init__(
        self,
        model: "OpenMythos",
        max_entries: int = 32,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.max_entries = max_entries
        self.device = device
        self._cache: OrderedDict[str, PrefixCacheEntry] = OrderedDict()
        self._total_hits = 0
        self._total_misses = 0

    # ------------------------------------------------------------------
    # キャッシュ操作
    # ------------------------------------------------------------------

    def _make_key(self, prefix_text: str) -> str:
        return hashlib.md5(prefix_text.encode()).hexdigest()[:16]

    def cache_prefix(
        self,
        prefix_text: str,
        n_loops: int = 4,
        force: bool = False,
    ) -> PrefixCacheEntry:
        """
        プレフィックスを prefill してキャッシュに保存する。

        Args:
            prefix_text -- キャッシュするプレフィックステキスト
            n_loops     -- モデルの推論ループ数
            force       -- True の場合、既存キャッシュを上書き

        Returns:
            作成/更新した PrefixCacheEntry
        """
        key = self._make_key(prefix_text)

        if key in self._cache and not force:
            # 既存エントリを LRU 先頭に移動
            self._cache.move_to_end(key)
            return self._cache[key]

        # LRU eviction
        if len(self._cache) >= self.max_entries:
            self._cache.popitem(last=False)

        # prefill
        vsize = self.model.cfg.vocab_size
        prefix_ids = [ord(c) % vsize for c in prefix_text]
        if not prefix_ids:
            prefix_ids = [0]

        prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(prefix_tensor, n_loops=n_loops)
            last_logits = logits[0, -1, :].clone()  # (vocab_size,) — キャッシュ保存

        entry = PrefixCacheEntry(
            prefix_text=prefix_text,
            prefix_ids=prefix_ids,
            cached_logits=last_logits,
            n_loops=n_loops,
        )
        self._cache[key] = entry
        self._cache.move_to_end(key)
        return entry

    def get(self, prefix_text: str) -> Optional[PrefixCacheEntry]:
        """キャッシュエントリを取得する。存在しない場合は None を返す。"""
        key = self._make_key(prefix_text)
        entry = self._cache.get(key)
        if entry is not None:
            self._cache.move_to_end(key)
            entry.hits += 1
            self._total_hits += 1
        else:
            self._total_misses += 1
        return entry

    def clear(self) -> None:
        """キャッシュを全消去する。"""
        self._cache.clear()

    def evict(self, prefix_text: str) -> bool:
        """特定エントリをキャッシュから削除する。"""
        key = self._make_key(prefix_text)
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def __len__(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------------
    # 統計
    # ------------------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        """キャッシュヒット率 (0.0〜1.0)。"""
        total = self._total_hits + self._total_misses
        return self._total_hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict:
        return {
            "n_entries": len(self._cache),
            "max_entries": self.max_entries,
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "hit_rate": round(self.hit_rate, 4),
        }

    # ------------------------------------------------------------------
    # キャッシュを活用した生成
    # ------------------------------------------------------------------

    def generate_with_cache(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        loops: int = 4,
        cache_prefix_len: Optional[int] = None,
    ) -> CachedGenResult:
        """
        キャッシュを活用してテキストを生成する。

        プロンプトの先頭部分がキャッシュにあればそこから継続生成する。
        なければ通常の prefill を行い、キャッシュに追加する。

        Args:
            prompt           -- 完全なプロンプト
            max_new_tokens   -- 最大生成トークン数
            temperature      -- サンプリング温度
            loops            -- 推論ループ数
            cache_prefix_len -- キャッシュとして使う prefix の文字長
                                (None の場合はプロンプト全体を prefix として扱う)

        Returns:
            CachedGenResult
        """
        t0 = time.perf_counter()
        vsize = self.model.cfg.vocab_size

        # キャッシュ対象 prefix を決定
        if cache_prefix_len is not None:
            cache_prefix = prompt[:cache_prefix_len]
            suffix = prompt[cache_prefix_len:]
        else:
            # プロンプト全体を1つの prefix として扱い、生成のみ行う
            cache_prefix = prompt
            suffix = ""

        # キャッシュ検索
        entry = self.get(cache_prefix)
        cache_hit = entry is not None

        if not cache_hit:
            # ミス: prefix を prefill してキャッシュ
            entry = self.cache_prefix(cache_prefix, n_loops=loops)

        prefill_skipped = entry.prefix_len if cache_hit else 0

        # suffix をトークナイズして追加 prefill
        suffix_ids = [ord(c) % vsize for c in suffix] if suffix else []

        if suffix_ids:
            suffix_tensor = torch.tensor(
                [entry.prefix_ids + suffix_ids], dtype=torch.long, device=self.device
            )
            with torch.no_grad():
                logits = self.model(suffix_tensor, n_loops=loops)
                cur_logits = logits[0, -1, :]
            cur_ids = suffix_tensor
        else:
            cur_logits = entry.cached_logits
            cur_ids = torch.tensor([entry.prefix_ids], dtype=torch.long, device=self.device)

        # デコード
        generated: list[int] = []
        for _ in range(max_new_tokens):
            next_logits = cur_logits / max(temperature, 1e-8)
            probs = F.softmax(next_logits, dim=-1)
            next_tok = int(torch.multinomial(probs, 1).item())
            generated.append(next_tok)
            cur_ids = torch.cat([cur_ids, torch.tensor([[next_tok]], device=self.device)], dim=1)
            with torch.no_grad():
                logits = self.model(cur_ids[:, -1:], n_loops=loops)
                cur_logits = logits[0, -1, :]
            if next_tok == vsize - 1:
                break

        # デコード
        chars = []
        for i in generated:
            try:
                c = chr(i % 128)
                if c.isprintable() or c in "\n\t ":
                    chars.append(c)
            except (ValueError, OverflowError):
                pass
        text = "".join(chars)

        latency_ms = (time.perf_counter() - t0) * 1000

        return CachedGenResult(
            text=text,
            prompt_used=prompt,
            cache_hit=cache_hit,
            cache_key=entry.cache_key,
            prefix_len=len(cache_prefix),
            latency_ms=round(latency_ms, 2),
            prefill_skipped_tokens=prefill_skipped,
        )
