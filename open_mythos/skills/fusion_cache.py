"""
Sprint 67B — Fusion 結果キャッシュ

同一質問に対する FusionEngine の再計算を TTL 付きインメモリキャッシュで回避する。

オブジェクト:
  CacheKey       : キャッシュキー (question + system の正規化ハッシュ)
  CacheEntry     : キャッシュエントリ (結果 + 生成時刻 + TTL)
  FusionCache    : インメモリキャッシュ管理 (get/put/evict/stats)
  CachedFusionEngine : FusionEngine をラップし、キャッシュを透過的に適用

設計方針:
  - 外部依存なし (hashlib / time のみ)
  - TTL デフォルト 300 秒（設定可能）
  - キャッシュキー = SHA-256(question + "\x00" + (system or ""))
  - max_size 超過時は LRU (最古エントリ) を evict
  - run_stream はキャッシュ対象外（SSE の性質上、再生成が自然）
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from open_mythos.skills.fusion import FusionEngine, FusionResult


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# キャッシュキー / エントリ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_cache_key(question: str, system: Optional[str]) -> str:
    """question と system から SHA-256 キャッシュキーを生成する"""
    raw = question + "\x00" + (system or "")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class CacheEntry:
    """キャッシュの 1 エントリ"""
    key:        str
    result:     FusionResult
    created_at: float = field(default_factory=time.time)
    ttl:        float = 300.0   # 秒

    @property
    def expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key":        self.key,
            "created_at": self.created_at,
            "ttl":        self.ttl,
            "expired":    self.expired,
            "age_sec":    round(time.time() - self.created_at, 2),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FusionCache
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FusionCache:
    """
    FusionResult の TTL 付きインメモリキャッシュ。

    Usage:
        cache = FusionCache(ttl=300, max_size=128)
        entry = cache.get(question="Q", system=None)
        if entry is None:
            result = engine.run(question="Q")
            cache.put(question="Q", system=None, result=result)
    """

    def __init__(self, ttl: float = 300.0, max_size: int = 128) -> None:
        self.ttl      = ttl
        self.max_size = max_size
        self._store: Dict[str, CacheEntry] = {}   # key → entry (insertion order)
        self._hits   = 0
        self._misses = 0

    # ---- 基本操作 ----

    def get(self, question: str, system: Optional[str] = None) -> Optional[FusionResult]:
        """キャッシュヒット時は FusionResult を返す。TTL 切れ・未登録は None。"""
        key = _make_cache_key(question, system)
        entry = self._store.get(key)
        if entry is None or entry.expired:
            if entry is not None:
                del self._store[key]
            self._misses += 1
            return None
        self._hits += 1
        return entry.result

    def put(
        self, question: str, result: FusionResult,
        system: Optional[str] = None,
    ) -> CacheEntry:
        """結果をキャッシュに登録する。max_size 超過時は最古エントリを削除。"""
        key = _make_cache_key(question, system)
        entry = CacheEntry(key=key, result=result, ttl=self.ttl)
        if key not in self._store and len(self._store) >= self.max_size:
            # LRU: 最古 (dict insertion order の先頭) を削除
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
        self._store[key] = entry
        return entry

    def invalidate(self, question: str, system: Optional[str] = None) -> bool:
        """指定エントリを削除する。削除できれば True。"""
        key = _make_cache_key(question, system)
        if key in self._store:
            del self._store[key]
            return True
        return False

    def evict_expired(self) -> int:
        """期限切れエントリを全削除し、削除数を返す。"""
        expired_keys = [k for k, e in self._store.items() if e.expired]
        for k in expired_keys:
            del self._store[k]
        return len(expired_keys)

    def clear(self) -> int:
        """全エントリを削除し、削除数を返す。"""
        n = len(self._store)
        self._store.clear()
        return n

    # ---- 統計 ----

    def stats(self) -> Dict[str, Any]:
        """キャッシュ統計を返す。"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total else 0.0
        live = sum(1 for e in self._store.values() if not e.expired)
        return {
            "size":       len(self._store),
            "live":       live,
            "expired":    len(self._store) - live,
            "hits":       self._hits,
            "misses":     self._misses,
            "hit_rate":   round(hit_rate, 4),
            "ttl":        self.ttl,
            "max_size":   self.max_size,
        }

    def __len__(self) -> int:
        return len(self._store)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CachedFusionEngine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CachedFusionEngine:
    """
    FusionEngine をラップし、キャッシュを透過的に適用する。

    Usage:
        engine = FusionEngineFactory.rule_based()
        cached = CachedFusionEngine(engine, cache=FusionCache(ttl=300))
        result = cached.run("質問")   # 2 回目はキャッシュから返す
    """

    def __init__(
        self,
        engine: FusionEngine,
        cache: Optional[FusionCache] = None,
    ) -> None:
        self._engine = engine
        self.cache   = cache if cache is not None else FusionCache()

    def run(
        self,
        question: str,
        system: Optional[str] = None,
    ) -> FusionResult:
        """
        キャッシュヒット時はキャッシュから返す。
        ミス時は FusionEngine.run() を呼び、結果をキャッシュに登録する。
        """
        cached = self.cache.get(question, system)
        if cached is not None:
            return cached
        result = self._engine.run(question, system=system)
        self.cache.put(question, result, system=system)
        return result

    def run_stream(self, question: str, system: Optional[str] = None):
        """ストリーミングはキャッシュ非対象 — 直接 engine に委譲。"""
        yield from self._engine.run_stream(question, system=system)

    @property
    def cache_stats(self) -> Dict[str, Any]:
        return self.cache.stats()
