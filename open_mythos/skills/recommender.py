"""
Sprint 81H — レコメンドエンジン統合

参考: https://qiita.com/birdwatcher/items/b60822bdf9be267e1328
      「レコメンドアルゴリズム入門：基礎から応用まで実装に必要な知識を解説」

実装手法:
    1. 協調フィルタリング (CF)
       - ユーザーベース型 (UserBasedCF)   — ユーザー同士の類似度
       - アイテムベース型 (ItemBasedCF)   — アイテム同士の類似度
    2. コンテンツベースフィルタリング (ContentBasedFilter)
       — アイテム特徴ベクトルのコサイン類似度
    3. 行列分解 (MatrixFactorizer)
       — SGD ベース MF (pure-Python, numpy 不要)
    4. ハイブリッド (HybridRecommender)
       — CF + CBF の重み付きブレンド
    5. 評価指標
       — Precision@K / Recall@K / NDCG@K / MAP@K
    6. パイプライン (RecommenderPipeline)
       — 上記全手法を束ねるファサード

パイプライン:
    Interaction[] ──► UserBasedCF ──► recommend(user_id)
                 ──► ItemBasedCF ──► similar_items(item_id)
                 ──► MatrixFactorizer ──► predict(user, item)
    Item[]       ──► ContentBasedFilter ──► recommend(user_id)
                 ──► HybridRecommender ──► recommend(user_id)
                              └── EvaluationReport
"""
from __future__ import annotations

import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ─── 定数 ──────────────────────────────────────────────────────────

DEFAULT_K_NEIGHBORS = 10   # 近傍ユーザー/アイテム数
DEFAULT_N_FACTORS   = 10   # 潜在因子数 (Matrix Factorization)
DEFAULT_N_EPOCHS    = 20   # SGD エポック数
DEFAULT_LR          = 0.01  # 学習率
DEFAULT_REG         = 0.01  # 正則化係数
DEFAULT_TOP_N       = 10   # 推薦件数


# ─── Enums ────────────────────────────────────────────────────────

class FeedbackType(str, Enum):
    """フィードバックの種類 (明示的 / 暗黙的)。"""
    PURCHASE = "purchase"   # 購入 (明示的・強シグナル)
    RATING   = "rating"     # 評価 (明示的)
    CLICK    = "click"      # クリック (暗黙的)
    VIEW     = "view"       # 閲覧 (暗黙的・弱シグナル)
    SKIP     = "skip"       # スキップ (暗黙的・負シグナル)


class SimilarityMetric(str, Enum):
    """類似度計算方式。"""
    COSINE  = "cosine"
    JACCARD = "jaccard"
    PEARSON = "pearson"


class RecommendMethod(str, Enum):
    """推薦手法。"""
    USER_CF  = "user_cf"
    ITEM_CF  = "item_cf"
    CONTENT  = "content"
    MF       = "mf"
    HYBRID   = "hybrid"


# ─── Data Models ──────────────────────────────────────────────────

@dataclass
class Interaction:
    """ユーザー × アイテムのインタラクション 1 件。"""
    user_id:       str
    item_id:       str
    feedback_type: FeedbackType = FeedbackType.VIEW
    value:         float = 1.0          # 評価値 or 暗黙値
    timestamp:     float = field(default_factory=time.time)
    interaction_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict:
        return {
            "interaction_id": self.interaction_id,
            "user_id": self.user_id,
            "item_id": self.item_id,
            "feedback_type": self.feedback_type.value,
            "value": self.value,
            "timestamp": self.timestamp,
        }


@dataclass
class Item:
    """アイテム (コンテンツベースフィルタリング用の特徴ベクトルを持つ)。"""
    item_id:  str
    features: Dict[str, float] = field(default_factory=dict)  # 特徴量 (正規化済み推奨)
    metadata: Dict[str, Any]   = field(default_factory=dict)  # タイトル・カテゴリなど
    created_at: float          = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "features": self.features,
            "metadata": self.metadata,
        }


@dataclass
class RecommendationResult:
    """推薦結果 1 件。"""
    result_id:    str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_id:      str = ""
    items:        List[Tuple[str, float]] = field(default_factory=list)
    method:       RecommendMethod = RecommendMethod.USER_CF
    generated_at: float = field(default_factory=time.time)

    @property
    def item_ids(self) -> List[str]:
        return [item_id for item_id, _ in self.items]

    def to_dict(self) -> dict:
        return {
            "result_id": self.result_id,
            "user_id": self.user_id,
            "items": [{"item_id": iid, "score": round(sc, 6)} for iid, sc in self.items],
            "method": self.method.value,
            "generated_at": self.generated_at,
            "count": len(self.items),
        }


@dataclass
class EvaluationReport:
    """推薦システムの評価レポート。"""
    method:        RecommendMethod
    k:             int
    precision_at_k: float = 0.0
    recall_at_k:    float = 0.0
    ndcg_at_k:      float = 0.0
    map_at_k:       float = 0.0
    n_users:        int = 0
    evaluated_at:   float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "method": self.method.value,
            "k": self.k,
            "precision_at_k": round(self.precision_at_k, 4),
            "recall_at_k": round(self.recall_at_k, 4),
            "ndcg_at_k": round(self.ndcg_at_k, 4),
            "map_at_k": round(self.map_at_k, 4),
            "n_users": self.n_users,
        }


# ─── 類似度計算 ────────────────────────────────────────────────────

def _cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    """疎ベクトル間のコサイン類似度。O(|共通キー|)。"""
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b[k] for k in a if k in b)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _jaccard_similarity(a: set, b: set) -> float:
    """Jaccard 類似度。集合ベース。"""
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def _pearson_correlation(a: Dict[str, float], b: Dict[str, float]) -> float:
    """ピアソン相関係数。共通キーのみで計算。"""
    common = [k for k in a if k in b]
    if len(common) < 2:
        return 0.0
    va = [a[k] for k in common]
    vb = [b[k] for k in common]
    mean_a = sum(va) / len(va)
    mean_b = sum(vb) / len(vb)
    da = [x - mean_a for x in va]
    db = [x - mean_b for x in vb]
    num = sum(x * y for x, y in zip(da, db))
    den = math.sqrt(sum(x * x for x in da)) * math.sqrt(sum(y * y for y in db))
    return num / den if den > 0 else 0.0


def _vec_dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _vec_norm(a: List[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


# ─── 評価指標 ──────────────────────────────────────────────────────

def precision_at_k(recommended: List[str], relevant: set, k: int) -> float:
    """Precision@K — 推薦上位K件中の正解率。"""
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for iid in top_k if iid in relevant)
    return hits / k


def recall_at_k(recommended: List[str], relevant: set, k: int) -> float:
    """Recall@K — 正解アイテムの何割を上位K件に含めたか。"""
    if not relevant or k <= 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for iid in top_k if iid in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: List[str], relevant: set, k: int) -> float:
    """NDCG@K — 順位を考慮した正規化累積利得。"""
    if not relevant or k <= 0:
        return 0.0
    dcg = 0.0
    for i, iid in enumerate(recommended[:k], start=1):
        if iid in relevant:
            dcg += 1.0 / math.log2(i + 1)
    # ideal DCG
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended: List[str], relevant: set, k: int) -> float:
    """AP@K — ヒット時の Precision の平均 (MAP 計算用)。"""
    if not relevant or k <= 0:
        return 0.0
    hits = 0
    total_precision = 0.0
    for i, iid in enumerate(recommended[:k], start=1):
        if iid in relevant:
            hits += 1
            total_precision += hits / i
    return total_precision / min(len(relevant), k)


# ─── InteractionStore ─────────────────────────────────────────────

class InteractionStore:
    """Interaction の CRUD ストア。"""

    def __init__(self) -> None:
        self._data: List[Interaction] = []

    def add(self, interaction: Interaction) -> None:
        self._data.append(interaction)

    def list_all(self) -> List[Interaction]:
        return list(self._data)

    def by_user(self, user_id: str) -> List[Interaction]:
        return [i for i in self._data if i.user_id == user_id]

    def by_item(self, item_id: str) -> List[Interaction]:
        return [i for i in self._data if i.item_id == item_id]

    def user_ids(self) -> List[str]:
        return list({i.user_id for i in self._data})

    def item_ids(self) -> List[str]:
        return list({i.item_id for i in self._data})

    def count(self) -> int:
        return len(self._data)

    def build_user_item_matrix(self) -> Dict[str, Dict[str, float]]:
        """ユーザー × アイテム スコア行列を構築。"""
        matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        for inter in self._data:
            uid, iid = inter.user_id, inter.item_id
            # 同一ユーザー × アイテムペアは最大値を採用
            prev = matrix[uid].get(iid, 0.0)
            matrix[uid][iid] = max(prev, inter.value)
        return dict(matrix)


# ─── ItemStore ────────────────────────────────────────────────────

class ItemStore:
    """Item の CRUD ストア。"""

    def __init__(self) -> None:
        self._items: Dict[str, Item] = {}

    def add(self, item: Item) -> None:
        self._items[item.item_id] = item

    def get(self, item_id: str) -> Optional[Item]:
        return self._items.get(item_id)

    def list_all(self) -> List[Item]:
        return list(self._items.values())

    def count(self) -> int:
        return len(self._items)


# ─── UserBasedCF ──────────────────────────────────────────────────

class UserBasedCF:
    """
    ユーザーベース協調フィルタリング。
    「あなたと似たユーザーが購入したアイテムを推薦します」
    参考: Qiita 記事 §協調フィルタリング > ユーザーベース型
    """

    def __init__(
        self,
        k_neighbors: int = DEFAULT_K_NEIGHBORS,
        similarity: SimilarityMetric = SimilarityMetric.COSINE,
    ) -> None:
        self.k_neighbors = k_neighbors
        self.similarity = similarity
        self._matrix: Dict[str, Dict[str, float]] = {}
        self._fitted = False

    def fit(self, interactions: List[Interaction]) -> "UserBasedCF":
        store = InteractionStore()
        for i in interactions:
            store.add(i)
        self._matrix = store.build_user_item_matrix()
        self._fitted = True
        return self

    def _sim(self, u: str, v: str) -> float:
        a, b = self._matrix.get(u, {}), self._matrix.get(v, {})
        if self.similarity == SimilarityMetric.COSINE:
            return _cosine_similarity(a, b)
        elif self.similarity == SimilarityMetric.JACCARD:
            return _jaccard_similarity(set(a.keys()), set(b.keys()))
        else:
            return _pearson_correlation(a, b)

    def _neighbors(self, user_id: str) -> List[Tuple[str, float]]:
        """k 近傍ユーザー (類似度降順)。"""
        sims = [
            (uid, self._sim(user_id, uid))
            for uid in self._matrix
            if uid != user_id
        ]
        sims.sort(key=lambda x: -x[1])
        return sims[:self.k_neighbors]

    def recommend(
        self,
        user_id: str,
        n: int = DEFAULT_TOP_N,
        exclude_seen: bool = True,
    ) -> List[Tuple[str, float]]:
        """ユーザーへの推薦アイテムリスト (item_id, score) 降順。"""
        if not self._fitted:
            return []
        seen = set(self._matrix.get(user_id, {}).keys())
        neighbors = self._neighbors(user_id)

        scores: Dict[str, float] = defaultdict(float)
        sim_sum: Dict[str, float] = defaultdict(float)
        for nbr_id, sim in neighbors:
            if sim <= 0:
                continue
            for item_id, val in self._matrix.get(nbr_id, {}).items():
                if exclude_seen and item_id in seen:
                    continue
                scores[item_id] += sim * val
                sim_sum[item_id] += sim

        result = [
            (iid, scores[iid] / sim_sum[iid])
            for iid in scores
            if sim_sum[iid] > 0
        ]
        result.sort(key=lambda x: -x[1])
        return result[:n]

    def similar_users(self, user_id: str, n: int = DEFAULT_TOP_N) -> List[Tuple[str, float]]:
        """類似ユーザーリスト。"""
        if not self._fitted:
            return []
        return self._neighbors(user_id)[:n]


# ─── ItemBasedCF ──────────────────────────────────────────────────

class ItemBasedCF:
    """
    アイテムベース協調フィルタリング。
    「この商品を買ったユーザーはこちらも買っています」
    参考: Qiita 記事 §協調フィルタリング > アイテムベース型
    """

    def __init__(
        self,
        k_neighbors: int = DEFAULT_K_NEIGHBORS,
        similarity: SimilarityMetric = SimilarityMetric.COSINE,
    ) -> None:
        self.k_neighbors = k_neighbors
        self.similarity = similarity
        self._item_matrix: Dict[str, Dict[str, float]] = {}  # item → {user: val}
        self._user_matrix: Dict[str, Dict[str, float]] = {}  # user → {item: val}
        self._fitted = False

    def fit(self, interactions: List[Interaction]) -> "ItemBasedCF":
        store = InteractionStore()
        for i in interactions:
            store.add(i)
        user_item = store.build_user_item_matrix()
        self._user_matrix = user_item
        # 転置: item → user
        item_user: Dict[str, Dict[str, float]] = defaultdict(dict)
        for uid, items in user_item.items():
            for iid, val in items.items():
                item_user[iid][uid] = val
        self._item_matrix = dict(item_user)
        self._fitted = True
        return self

    def _sim(self, i: str, j: str) -> float:
        a, b = self._item_matrix.get(i, {}), self._item_matrix.get(j, {})
        if self.similarity == SimilarityMetric.COSINE:
            return _cosine_similarity(a, b)
        elif self.similarity == SimilarityMetric.JACCARD:
            return _jaccard_similarity(set(a.keys()), set(b.keys()))
        else:
            return _pearson_correlation(a, b)

    def similar_items(self, item_id: str, n: int = DEFAULT_TOP_N) -> List[Tuple[str, float]]:
        """類似アイテムリスト (item_id, score) 降順。"""
        if not self._fitted:
            return []
        sims = [
            (iid, self._sim(item_id, iid))
            for iid in self._item_matrix
            if iid != item_id
        ]
        sims.sort(key=lambda x: -x[1])
        return sims[:n]

    def recommend(
        self,
        user_id: str,
        n: int = DEFAULT_TOP_N,
        exclude_seen: bool = True,
    ) -> List[Tuple[str, float]]:
        """ユーザーの購入履歴から類似アイテムを推薦。"""
        if not self._fitted:
            return []
        seen = set(self._user_matrix.get(user_id, {}).keys())
        scores: Dict[str, float] = defaultdict(float)

        for seen_item, rating in self._user_matrix.get(user_id, {}).items():
            for nbr_item, sim in self.similar_items(seen_item, n=self.k_neighbors):
                if exclude_seen and nbr_item in seen:
                    continue
                scores[nbr_item] += sim * rating

        result = list(scores.items())
        result.sort(key=lambda x: -x[1])
        return result[:n]


# ─── ContentBasedFilter ───────────────────────────────────────────

class ContentBasedFilter:
    """
    コンテンツベースフィルタリング。
    アイテム特徴ベクトルのコサイン類似度で推薦。
    参考: Qiita 記事 §コンテンツベースフィルタリング
    """

    def __init__(self) -> None:
        self._items: Dict[str, Item] = {}
        self._fitted = False

    def fit(self, items: List[Item]) -> "ContentBasedFilter":
        self._items = {item.item_id: item for item in items}
        self._fitted = True
        return self

    def similar_items(self, item_id: str, n: int = DEFAULT_TOP_N) -> List[Tuple[str, float]]:
        """特徴ベクトルが似ているアイテムを返す。"""
        if not self._fitted or item_id not in self._items:
            return []
        base = self._items[item_id].features
        sims = [
            (iid, _cosine_similarity(base, item.features))
            for iid, item in self._items.items()
            if iid != item_id
        ]
        sims.sort(key=lambda x: -x[1])
        return sims[:n]

    def recommend(
        self,
        user_history: List[str],
        n: int = DEFAULT_TOP_N,
        exclude_seen: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        ユーザー閲覧履歴 (item_id リスト) の平均特徴ベクトルと
        コサイン類似度が高いアイテムを推薦。
        """
        if not self._fitted or not user_history:
            return []
        seen = set(user_history) if exclude_seen else set()

        # 平均特徴ベクトル
        all_keys: set = set()
        for iid in user_history:
            if iid in self._items:
                all_keys |= set(self._items[iid].features.keys())

        if not all_keys:
            return []

        avg: Dict[str, float] = {}
        count = 0
        for iid in user_history:
            if iid in self._items:
                for k, v in self._items[iid].features.items():
                    avg[k] = avg.get(k, 0.0) + v
                count += 1
        if count > 0:
            avg = {k: v / count for k, v in avg.items()}

        sims = [
            (iid, _cosine_similarity(avg, item.features))
            for iid, item in self._items.items()
            if iid not in seen
        ]
        sims.sort(key=lambda x: -x[1])
        return sims[:n]


# ─── MatrixFactorizer (SGD ベース MF) ────────────────────────────

class MatrixFactorizer:
    """
    行列分解 (Matrix Factorization)。
    SGD で潜在因子ベクトルを学習。numpy 不要の純 Python 実装。
    参考: Qiita 記事 §行列分解 (MF)
    """

    def __init__(
        self,
        n_factors: int = DEFAULT_N_FACTORS,
        n_epochs:  int = DEFAULT_N_EPOCHS,
        lr:        float = DEFAULT_LR,
        reg:       float = DEFAULT_REG,
        seed:      int = 42,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs  = n_epochs
        self.lr        = lr
        self.reg       = reg
        self.seed      = seed
        self._user_factors: Dict[str, List[float]] = {}
        self._item_factors: Dict[str, List[float]] = {}
        self._global_mean: float = 0.0
        self._fitted = False

    def _init_vector(self, rng_state: List[int]) -> List[float]:
        """簡易 LCG 乱数で因子ベクトルを初期化。"""
        a, c, m = 1664525, 1013904223, 2**32
        vec = []
        for _ in range(self.n_factors):
            rng_state[0] = (a * rng_state[0] + c) % m
            vec.append((rng_state[0] / m) * 0.1)
        return vec

    def fit(self, interactions: List[Interaction]) -> "MatrixFactorizer":
        if not interactions:
            return self
        rng = [self.seed]
        # 全ユーザー・アイテムを収集
        user_ids = list({i.user_id for i in interactions})
        item_ids = list({i.item_id for i in interactions})
        self._user_factors = {uid: self._init_vector(rng) for uid in user_ids}
        self._item_factors = {iid: self._init_vector(rng) for iid in item_ids}
        self._global_mean = sum(i.value for i in interactions) / len(interactions)

        # SGD
        for _ in range(self.n_epochs):
            for inter in interactions:
                uid, iid, r = inter.user_id, inter.item_id, inter.value
                if uid not in self._user_factors or iid not in self._item_factors:
                    continue
                pu = self._user_factors[uid]
                qi = self._item_factors[iid]
                pred = self._global_mean + _vec_dot(pu, qi)
                err = r - pred
                # 更新
                new_pu = [
                    pu[k] + self.lr * (err * qi[k] - self.reg * pu[k])
                    for k in range(self.n_factors)
                ]
                new_qi = [
                    qi[k] + self.lr * (err * pu[k] - self.reg * qi[k])
                    for k in range(self.n_factors)
                ]
                self._user_factors[uid] = new_pu
                self._item_factors[iid] = new_qi

        self._fitted = True
        return self

    def predict(self, user_id: str, item_id: str) -> float:
        """ユーザー × アイテムの予測スコア。"""
        if not self._fitted:
            return self._global_mean
        pu = self._user_factors.get(user_id)
        qi = self._item_factors.get(item_id)
        if pu is None or qi is None:
            return self._global_mean
        return self._global_mean + _vec_dot(pu, qi)

    def recommend(
        self,
        user_id: str,
        n: int = DEFAULT_TOP_N,
        exclude_seen: Optional[set] = None,
    ) -> List[Tuple[str, float]]:
        """ユーザーへの推薦アイテムリスト。"""
        if not self._fitted or user_id not in self._user_factors:
            return []
        seen = exclude_seen or set()
        scores = [
            (iid, self.predict(user_id, iid))
            for iid in self._item_factors
            if iid not in seen
        ]
        scores.sort(key=lambda x: -x[1])
        return scores[:n]


# ─── HybridRecommender ────────────────────────────────────────────

class HybridRecommender:
    """
    ハイブリッドレコメンダー。
    UserBasedCF と ContentBasedFilter の重み付きスコアをブレンド。
    """

    def __init__(
        self,
        cf_weight:  float = 0.5,
        cbf_weight: float = 0.5,
        k_neighbors: int = DEFAULT_K_NEIGHBORS,
    ) -> None:
        self.cf_weight  = cf_weight
        self.cbf_weight = cbf_weight
        self._cf  = UserBasedCF(k_neighbors=k_neighbors)
        self._cbf = ContentBasedFilter()
        self._user_history: Dict[str, List[str]] = defaultdict(list)
        self._fitted = False

    def fit(self, interactions: List[Interaction], items: List[Item]) -> "HybridRecommender":
        self._cf.fit(interactions)
        self._cbf.fit(items)
        for inter in interactions:
            if inter.item_id not in self._user_history[inter.user_id]:
                self._user_history[inter.user_id].append(inter.item_id)
        self._fitted = True
        return self

    def recommend(
        self,
        user_id: str,
        n: int = DEFAULT_TOP_N,
    ) -> List[Tuple[str, float]]:
        """CF + CBF のスコアをブレンドして推薦。"""
        if not self._fitted:
            return []
        cf_recs  = dict(self._cf.recommend(user_id, n=n * 2))
        cbf_recs = dict(self._cbf.recommend(self._user_history.get(user_id, []), n=n * 2))

        all_items = set(cf_recs) | set(cbf_recs)
        blended: Dict[str, float] = {}
        for iid in all_items:
            blended[iid] = (
                self.cf_weight  * cf_recs.get(iid, 0.0) +
                self.cbf_weight * cbf_recs.get(iid, 0.0)
            )

        result = sorted(blended.items(), key=lambda x: -x[1])
        return result[:n]


# ─── RecommenderPipeline ──────────────────────────────────────────

class RecommenderPipeline:
    """
    レコメンドパイプラインのファサード。
    インタラクション登録 → 学習 → 推薦 → 評価 を一貫して提供する。
    """

    def __init__(self) -> None:
        self._interactions = InteractionStore()
        self._items = ItemStore()
        self._user_cf:  Optional[UserBasedCF]      = None
        self._item_cf:  Optional[ItemBasedCF]       = None
        self._cbf:      Optional[ContentBasedFilter]= None
        self._mf:       Optional[MatrixFactorizer]  = None
        self._hybrid:   Optional[HybridRecommender] = None
        self._trained_methods: List[RecommendMethod] = []

    # ── データ登録 ─────────────────────────────────────────────────

    def add_interaction(
        self,
        user_id:       str,
        item_id:       str,
        feedback_type: FeedbackType = FeedbackType.VIEW,
        value:         float = 1.0,
    ) -> Interaction:
        inter = Interaction(user_id=user_id, item_id=item_id,
                            feedback_type=feedback_type, value=value)
        self._interactions.add(inter)
        return inter

    def add_item(
        self,
        item_id:  str,
        features: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Item:
        item = Item(item_id=item_id, features=features, metadata=metadata or {})
        self._items.add(item)
        return item

    # ── 学習 ──────────────────────────────────────────────────────

    def train(
        self,
        method: RecommendMethod = RecommendMethod.USER_CF,
        **kwargs: Any,
    ) -> None:
        """指定手法でモデルを学習する。"""
        inters = self._interactions.list_all()
        items  = self._items.list_all()

        if method == RecommendMethod.USER_CF:
            self._user_cf = UserBasedCF(**{k: v for k, v in kwargs.items()
                                           if k in ("k_neighbors", "similarity")})
            self._user_cf.fit(inters)

        elif method == RecommendMethod.ITEM_CF:
            self._item_cf = ItemBasedCF(**{k: v for k, v in kwargs.items()
                                           if k in ("k_neighbors", "similarity")})
            self._item_cf.fit(inters)

        elif method == RecommendMethod.CONTENT:
            self._cbf = ContentBasedFilter()
            self._cbf.fit(items)

        elif method == RecommendMethod.MF:
            self._mf = MatrixFactorizer(**{k: v for k, v in kwargs.items()
                                           if k in ("n_factors", "n_epochs", "lr", "reg", "seed")})
            self._mf.fit(inters)

        elif method == RecommendMethod.HYBRID:
            self._hybrid = HybridRecommender(**{k: v for k, v in kwargs.items()
                                                if k in ("cf_weight", "cbf_weight", "k_neighbors")})
            self._hybrid.fit(inters, items)

        if method not in self._trained_methods:
            self._trained_methods.append(method)

    # ── 推薦 ──────────────────────────────────────────────────────

    def recommend(
        self,
        user_id: str,
        n: int = DEFAULT_TOP_N,
        method: RecommendMethod = RecommendMethod.USER_CF,
    ) -> RecommendationResult:
        items: List[Tuple[str, float]] = []

        if method == RecommendMethod.USER_CF and self._user_cf:
            items = self._user_cf.recommend(user_id, n=n)
        elif method == RecommendMethod.ITEM_CF and self._item_cf:
            items = self._item_cf.recommend(user_id, n=n)
        elif method == RecommendMethod.CONTENT and self._cbf:
            hist = [i.item_id for i in self._interactions.by_user(user_id)]
            items = self._cbf.recommend(hist, n=n)
        elif method == RecommendMethod.MF and self._mf:
            seen = {i.item_id for i in self._interactions.by_user(user_id)}
            items = self._mf.recommend(user_id, n=n, exclude_seen=seen)
        elif method == RecommendMethod.HYBRID and self._hybrid:
            items = self._hybrid.recommend(user_id, n=n)

        return RecommendationResult(user_id=user_id, items=items, method=method)

    # ── 評価 ──────────────────────────────────────────────────────

    def evaluate(
        self,
        test_interactions: List[Interaction],
        k: int = DEFAULT_TOP_N,
        method: RecommendMethod = RecommendMethod.USER_CF,
    ) -> EvaluationReport:
        """
        テストデータを正解として評価指標を算出。
        ユーザーごとに推薦結果と正解セットを比較。
        """
        # test: ユーザー → 正解アイテムセット
        ground_truth: Dict[str, set] = defaultdict(set)
        for inter in test_interactions:
            ground_truth[inter.user_id].add(inter.item_id)

        p_list, r_list, ndcg_list, ap_list = [], [], [], []
        for uid, relevant in ground_truth.items():
            result = self.recommend(uid, n=k, method=method)
            rec_ids = result.item_ids
            p_list.append(precision_at_k(rec_ids, relevant, k))
            r_list.append(recall_at_k(rec_ids, relevant, k))
            ndcg_list.append(ndcg_at_k(rec_ids, relevant, k))
            ap_list.append(average_precision_at_k(rec_ids, relevant, k))

        n = len(p_list)
        return EvaluationReport(
            method=method,
            k=k,
            precision_at_k=sum(p_list) / n if n else 0.0,
            recall_at_k=sum(r_list) / n if n else 0.0,
            ndcg_at_k=sum(ndcg_list) / n if n else 0.0,
            map_at_k=sum(ap_list) / n if n else 0.0,
            n_users=n,
        )

    def status(self) -> dict:
        return {
            "interactions": self._interactions.count(),
            "items": self._items.count(),
            "trained_methods": [m.value for m in self._trained_methods],
            "users": len(self._interactions.user_ids()),
        }
