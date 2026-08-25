"""
Sprint 81H — レコメンドエンジン テスト
参考: https://qiita.com/birdwatcher/items/b60822bdf9be267e1328
累計: 4613 + ? → 目標 +70 PASS
"""
from __future__ import annotations

import math
from typing import List

import pytest

from open_mythos.skills.recommender import (
    # Enums
    FeedbackType, SimilarityMetric, RecommendMethod,
    # Data models
    Interaction, Item, RecommendationResult, EvaluationReport,
    # Similarity
    _cosine_similarity, _jaccard_similarity, _pearson_correlation,
    # Metrics
    precision_at_k, recall_at_k, ndcg_at_k, average_precision_at_k,
    # Stores
    InteractionStore, ItemStore,
    # Recommenders
    UserBasedCF, ItemBasedCF, ContentBasedFilter, MatrixFactorizer,
    HybridRecommender,
    # Pipeline
    RecommenderPipeline,
    # Constants
    DEFAULT_K_NEIGHBORS, DEFAULT_N_FACTORS, DEFAULT_TOP_N,
)


# ─── ヘルパー ─────────────────────────────────────────────────────

def make_interaction(user_id: str, item_id: str,
                     value: float = 1.0,
                     fb: FeedbackType = FeedbackType.PURCHASE) -> Interaction:
    return Interaction(user_id=user_id, item_id=item_id,
                       feedback_type=fb, value=value)


def make_item(item_id: str, **features: float) -> Item:
    return Item(item_id=item_id, features=dict(features))


def make_interactions_grid() -> List[Interaction]:
    """4ユーザー × 5アイテムの評価データ（一部欠損）。"""
    data = [
        ("u1", "i1", 5.0), ("u1", "i2", 3.0), ("u1", "i3", 4.0),
        ("u2", "i1", 4.0), ("u2", "i2", 2.0), ("u2", "i4", 3.0),
        ("u3", "i2", 5.0), ("u3", "i3", 4.0), ("u3", "i5", 3.0),
        ("u4", "i1", 3.0), ("u4", "i4", 4.0), ("u4", "i5", 5.0),
    ]
    return [make_interaction(u, i, v) for u, i, v in data]


def make_items_list() -> List[Item]:
    """5アイテムの特徴ベクトル。"""
    return [
        make_item("i1", action=1.0, sci_fi=0.0, romance=0.0),
        make_item("i2", action=0.5, sci_fi=0.5, romance=0.0),
        make_item("i3", action=0.0, sci_fi=1.0, romance=0.0),
        make_item("i4", action=0.0, sci_fi=0.5, romance=0.5),
        make_item("i5", action=0.0, sci_fi=0.0, romance=1.0),
    ]


# ─── 定数 ────────────────────────────────────────────────────────

class TestConstants:
    def test_default_k(self):
        assert DEFAULT_K_NEIGHBORS >= 5

    def test_default_n(self):
        assert DEFAULT_TOP_N >= 5

    def test_default_factors(self):
        assert DEFAULT_N_FACTORS >= 5


# ─── Enums ───────────────────────────────────────────────────────

class TestEnums:
    def test_feedback_types(self):
        assert FeedbackType.PURCHASE == "purchase"
        assert FeedbackType.CLICK    == "click"
        assert FeedbackType.VIEW     == "view"
        assert FeedbackType.RATING   == "rating"
        assert FeedbackType.SKIP     == "skip"
        assert len(FeedbackType) == 5

    def test_similarity_metrics(self):
        assert len(SimilarityMetric) == 3

    def test_recommend_methods(self):
        assert len(RecommendMethod) == 5


# ─── データモデル ─────────────────────────────────────────────────

class TestInteraction:
    def test_to_dict_keys(self):
        inter = make_interaction("u1", "i1", 3.0)
        d = inter.to_dict()
        for k in ("interaction_id", "user_id", "item_id",
                  "feedback_type", "value", "timestamp"):
            assert k in d

    def test_feedback_value_string(self):
        inter = make_interaction("u1", "i1", fb=FeedbackType.CLICK)
        assert inter.to_dict()["feedback_type"] == "click"

    def test_unique_ids(self):
        a = make_interaction("u1", "i1")
        b = make_interaction("u1", "i1")
        assert a.interaction_id != b.interaction_id


class TestItem:
    def test_to_dict(self):
        item = make_item("i1", action=1.0, sci_fi=0.5)
        d = item.to_dict()
        assert d["item_id"] == "i1"
        assert "features" in d
        assert d["features"]["action"] == 1.0

    def test_empty_features(self):
        item = Item(item_id="i0")
        assert item.features == {}


class TestRecommendationResult:
    def test_item_ids(self):
        result = RecommendationResult(
            user_id="u1",
            items=[("i1", 0.9), ("i2", 0.7)],
        )
        assert result.item_ids == ["i1", "i2"]

    def test_to_dict_structure(self):
        result = RecommendationResult(
            user_id="u1",
            items=[("i1", 0.8)],
            method=RecommendMethod.USER_CF,
        )
        d = result.to_dict()
        assert d["user_id"] == "u1"
        assert d["count"] == 1
        assert d["method"] == "user_cf"
        assert d["items"][0]["item_id"] == "i1"

    def test_empty_result(self):
        r = RecommendationResult(user_id="u99")
        assert r.item_ids == []
        assert r.to_dict()["count"] == 0


class TestEvaluationReport:
    def test_to_dict(self):
        rep = EvaluationReport(method=RecommendMethod.USER_CF, k=10,
                               precision_at_k=0.5, recall_at_k=0.3,
                               ndcg_at_k=0.4, map_at_k=0.35, n_users=5)
        d = rep.to_dict()
        assert d["k"] == 10
        assert d["precision_at_k"] == 0.5
        assert d["n_users"] == 5


# ─── 類似度 ──────────────────────────────────────────────────────

class TestSimilarity:
    def test_cosine_identical(self):
        a = {"x": 1.0, "y": 1.0}
        assert abs(_cosine_similarity(a, a) - 1.0) < 1e-9

    def test_cosine_orthogonal(self):
        a = {"x": 1.0}
        b = {"y": 1.0}
        assert _cosine_similarity(a, b) == 0.0

    def test_cosine_empty(self):
        assert _cosine_similarity({}, {"x": 1.0}) == 0.0

    def test_cosine_partial_overlap(self):
        a = {"x": 1.0, "y": 0.0}
        b = {"x": 1.0, "z": 0.0}
        # both have x=1 → cosine = 1.0/1.0 * 1.0/1.0 ... wait
        # norm_a = sqrt(1+0) = 1, norm_b = sqrt(1+0) = 1, dot = 1
        assert abs(_cosine_similarity(a, b) - 1.0) < 1e-9

    def test_jaccard_full_overlap(self):
        a = {"x", "y"}
        b = {"x", "y"}
        assert _jaccard_similarity(a, b) == 1.0

    def test_jaccard_no_overlap(self):
        assert _jaccard_similarity({"x"}, {"y"}) == 0.0

    def test_jaccard_partial(self):
        # |{x}| / |{x,y,z}| = 1/3
        sim = _jaccard_similarity({"x", "y"}, {"x", "z"})
        assert abs(sim - 1/3) < 1e-9

    def test_jaccard_empty(self):
        assert _jaccard_similarity(set(), set()) == 0.0

    def test_pearson_perfect(self):
        a = {"x": 1.0, "y": 2.0, "z": 3.0}
        b = {"x": 2.0, "y": 4.0, "z": 6.0}  # proportional
        assert abs(_pearson_correlation(a, b) - 1.0) < 1e-9

    def test_pearson_negative(self):
        a = {"x": 1.0, "y": 2.0, "z": 3.0}
        b = {"x": 3.0, "y": 2.0, "z": 1.0}  # inverse
        assert _pearson_correlation(a, b) < 0

    def test_pearson_insufficient_data(self):
        a = {"x": 1.0}
        b = {"x": 2.0}
        assert _pearson_correlation(a, b) == 0.0


# ─── 評価指標 ────────────────────────────────────────────────────

class TestMetrics:
    def test_precision_perfect(self):
        assert precision_at_k(["i1", "i2", "i3"], {"i1", "i2", "i3"}, 3) == 1.0

    def test_precision_zero(self):
        assert precision_at_k(["i1", "i2"], {"i3"}, 2) == 0.0

    def test_precision_partial(self):
        # 2 hits in top 4 → 0.5
        assert precision_at_k(["i1", "i2", "i3", "i4"], {"i1", "i3"}, 4) == 0.5

    def test_precision_k_zero(self):
        assert precision_at_k(["i1"], {"i1"}, 0) == 0.0

    def test_recall_perfect(self):
        assert recall_at_k(["i1", "i2"], {"i1", "i2"}, 2) == 1.0

    def test_recall_zero(self):
        assert recall_at_k(["i1"], {"i2"}, 1) == 0.0

    def test_recall_partial(self):
        # 1 hit out of 2 relevant → 0.5
        assert recall_at_k(["i1", "i3"], {"i1", "i2"}, 2) == 0.5

    def test_recall_empty_relevant(self):
        assert recall_at_k(["i1"], set(), 1) == 0.0

    def test_ndcg_perfect(self):
        assert abs(ndcg_at_k(["i1", "i2", "i3"], {"i1", "i2", "i3"}, 3) - 1.0) < 1e-9

    def test_ndcg_first_hit_higher(self):
        # hit at pos1 > hit at pos2
        ndcg1 = ndcg_at_k(["i1", "x"], {"i1"}, 2)
        ndcg2 = ndcg_at_k(["x", "i1"], {"i1"}, 2)
        assert ndcg1 > ndcg2

    def test_ndcg_zero(self):
        assert ndcg_at_k(["i1"], {"i2"}, 1) == 0.0

    def test_ndcg_k_zero(self):
        assert ndcg_at_k(["i1"], {"i1"}, 0) == 0.0

    def test_ap_perfect(self):
        assert abs(average_precision_at_k(["i1", "i2"], {"i1", "i2"}, 2) - 1.0) < 1e-9

    def test_ap_zero(self):
        assert average_precision_at_k(["i1"], {"i2"}, 1) == 0.0


# ─── InteractionStore ────────────────────────────────────────────

class TestInteractionStore:
    def _make(self) -> InteractionStore:
        store = InteractionStore()
        for uid, iid, v in [("u1","i1",1.0),("u1","i2",2.0),("u2","i1",3.0)]:
            store.add(make_interaction(uid, iid, v))
        return store

    def test_count(self):
        assert self._make().count() == 3

    def test_by_user(self):
        store = self._make()
        assert len(store.by_user("u1")) == 2
        assert len(store.by_user("u9")) == 0

    def test_by_item(self):
        store = self._make()
        assert len(store.by_item("i1")) == 2

    def test_user_ids(self):
        store = self._make()
        assert set(store.user_ids()) == {"u1", "u2"}

    def test_item_ids(self):
        store = self._make()
        assert set(store.item_ids()) == {"i1", "i2"}

    def test_matrix_max_value(self):
        store = InteractionStore()
        store.add(make_interaction("u1", "i1", 1.0))
        store.add(make_interaction("u1", "i1", 5.0))  # 重複 → 最大値
        m = store.build_user_item_matrix()
        assert m["u1"]["i1"] == 5.0

    def test_matrix_structure(self):
        store = self._make()
        m = store.build_user_item_matrix()
        assert "u1" in m
        assert "i1" in m["u1"]


# ─── UserBasedCF ─────────────────────────────────────────────────

class TestUserBasedCF:
    def _setup(self) -> UserBasedCF:
        cf = UserBasedCF(k_neighbors=3)
        cf.fit(make_interactions_grid())
        return cf

    def test_recommend_returns_list(self):
        cf = self._setup()
        recs = cf.recommend("u1", n=3)
        assert isinstance(recs, list)

    def test_recommend_excludes_seen(self):
        cf = self._setup()
        seen = {"i1", "i2", "i3"}
        recs = cf.recommend("u1", n=5, exclude_seen=True)
        rec_ids = {iid for iid, _ in recs}
        assert rec_ids.isdisjoint(seen)

    def test_recommend_score_positive(self):
        cf = self._setup()
        recs = cf.recommend("u1", n=3)
        for _, score in recs:
            assert score >= 0

    def test_recommend_sorted(self):
        cf = self._setup()
        recs = cf.recommend("u1", n=5)
        scores = [s for _, s in recs]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_unknown_user(self):
        cf = self._setup()
        assert cf.recommend("u_unknown") == []

    def test_similar_users(self):
        cf = self._setup()
        sims = cf.similar_users("u1", n=3)
        assert len(sims) <= 3
        for uid, score in sims:
            assert uid != "u1"
            assert -1.0 <= score <= 1.0

    def test_jaccard_similarity(self):
        cf = UserBasedCF(k_neighbors=3, similarity=SimilarityMetric.JACCARD)
        cf.fit(make_interactions_grid())
        recs = cf.recommend("u1", n=3)
        assert isinstance(recs, list)

    def test_pearson_similarity(self):
        cf = UserBasedCF(k_neighbors=3, similarity=SimilarityMetric.PEARSON)
        cf.fit(make_interactions_grid())
        recs = cf.recommend("u1", n=3)
        assert isinstance(recs, list)

    def test_not_fitted(self):
        cf = UserBasedCF()
        assert cf.recommend("u1") == []


# ─── ItemBasedCF ─────────────────────────────────────────────────

class TestItemBasedCF:
    def _setup(self) -> ItemBasedCF:
        cf = ItemBasedCF(k_neighbors=3)
        cf.fit(make_interactions_grid())
        return cf

    def test_similar_items(self):
        cf = self._setup()
        sims = cf.similar_items("i1", n=3)
        assert len(sims) <= 3
        for iid, score in sims:
            assert iid != "i1"

    def test_similar_items_sorted(self):
        cf = self._setup()
        sims = cf.similar_items("i1", n=5)
        scores = [s for _, s in sims]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_returns_list(self):
        cf = self._setup()
        recs = cf.recommend("u1", n=3)
        assert isinstance(recs, list)

    def test_recommend_excludes_seen(self):
        cf = self._setup()
        seen = {"i1", "i2", "i3"}
        recs = cf.recommend("u1", n=5, exclude_seen=True)
        for iid, _ in recs:
            assert iid not in seen

    def test_not_fitted(self):
        cf = ItemBasedCF()
        assert cf.recommend("u1") == []
        assert cf.similar_items("i1") == []


# ─── ContentBasedFilter ──────────────────────────────────────────

class TestContentBasedFilter:
    def _setup(self) -> ContentBasedFilter:
        cbf = ContentBasedFilter()
        cbf.fit(make_items_list())
        return cbf

    def test_similar_items(self):
        cbf = self._setup()
        sims = cbf.similar_items("i1", n=3)
        assert len(sims) == 3
        for iid, score in sims:
            assert iid != "i1"
            assert 0.0 <= score <= 1.0

    def test_similar_items_sorted(self):
        cbf = self._setup()
        sims = cbf.similar_items("i1", n=4)
        scores = [s for _, s in sims]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_from_history(self):
        cbf = self._setup()
        recs = cbf.recommend(["i1", "i2"], n=3)
        assert len(recs) <= 3
        rec_ids = [iid for iid, _ in recs]
        # seen i1, i2 → neither should appear
        assert "i1" not in rec_ids
        assert "i2" not in rec_ids

    def test_recommend_empty_history(self):
        cbf = self._setup()
        assert cbf.recommend([]) == []

    def test_unknown_item(self):
        cbf = self._setup()
        assert cbf.similar_items("i_unknown") == []

    def test_not_fitted(self):
        cbf = ContentBasedFilter()
        assert cbf.recommend(["i1"]) == []


# ─── MatrixFactorizer ────────────────────────────────────────────

class TestMatrixFactorizer:
    def _setup(self) -> MatrixFactorizer:
        mf = MatrixFactorizer(n_factors=5, n_epochs=30, lr=0.01, seed=42)
        mf.fit(make_interactions_grid())
        return mf

    def test_predict_returns_float(self):
        mf = self._setup()
        score = mf.predict("u1", "i4")
        assert isinstance(score, float)

    def test_predict_unknown_user(self):
        mf = self._setup()
        # returns global mean
        score = mf.predict("u_new", "i1")
        assert isinstance(score, float)

    def test_predict_known_pair(self):
        mf = self._setup()
        # 学習済みペアは positive
        score = mf.predict("u1", "i1")
        assert score > 0

    def test_recommend_returns_list(self):
        mf = self._setup()
        recs = mf.recommend("u1", n=3)
        assert isinstance(recs, list)
        assert len(recs) <= 3

    def test_recommend_sorted(self):
        mf = self._setup()
        recs = mf.recommend("u1", n=5)
        scores = [s for _, s in recs]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_exclude_seen(self):
        mf = self._setup()
        seen = {"i1", "i2", "i3"}
        recs = mf.recommend("u1", n=5, exclude_seen=seen)
        for iid, _ in recs:
            assert iid not in seen

    def test_empty_interactions(self):
        mf = MatrixFactorizer()
        mf.fit([])
        assert mf.recommend("u1") == []

    def test_not_fitted_predict(self):
        mf = MatrixFactorizer()
        # returns default global_mean (0.0 before fit)
        score = mf.predict("u1", "i1")
        assert isinstance(score, float)


# ─── HybridRecommender ───────────────────────────────────────────

class TestHybridRecommender:
    def _setup(self) -> HybridRecommender:
        hybrid = HybridRecommender(cf_weight=0.6, cbf_weight=0.4, k_neighbors=3)
        hybrid.fit(make_interactions_grid(), make_items_list())
        return hybrid

    def test_recommend_returns_list(self):
        hybrid = self._setup()
        recs = hybrid.recommend("u1", n=3)
        assert isinstance(recs, list)

    def test_recommend_sorted(self):
        hybrid = self._setup()
        recs = hybrid.recommend("u1", n=5)
        scores = [s for _, s in recs]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_unknown_user(self):
        hybrid = self._setup()
        # unknown user → empty or minimal
        recs = hybrid.recommend("u_new", n=3)
        assert isinstance(recs, list)

    def test_not_fitted(self):
        hybrid = HybridRecommender()
        assert hybrid.recommend("u1") == []


# ─── RecommenderPipeline ─────────────────────────────────────────

class TestRecommenderPipeline:
    def _setup(self) -> RecommenderPipeline:
        pipeline = RecommenderPipeline()
        for uid, iid, v in [
            ("u1","i1",5.0),("u1","i2",3.0),("u1","i3",4.0),
            ("u2","i1",4.0),("u2","i2",2.0),("u2","i4",3.0),
            ("u3","i2",5.0),("u3","i3",4.0),("u3","i5",3.0),
            ("u4","i1",3.0),("u4","i4",4.0),("u4","i5",5.0),
        ]:
            pipeline.add_interaction(uid, iid, FeedbackType.PURCHASE, v)
        for iid, feats in [
            ("i1", {"action":1.0}), ("i2", {"action":0.5,"sci_fi":0.5}),
            ("i3", {"sci_fi":1.0}), ("i4", {"sci_fi":0.5,"romance":0.5}),
            ("i5", {"romance":1.0}),
        ]:
            pipeline.add_item(iid, feats)
        return pipeline

    def test_status_before_train(self):
        p = self._setup()
        st = p.status()
        assert st["interactions"] == 12
        assert st["items"] == 5
        assert st["trained_methods"] == []

    def test_add_interaction(self):
        p = RecommenderPipeline()
        inter = p.add_interaction("u1", "i1", FeedbackType.CLICK, 1.0)
        assert inter.user_id == "u1"
        assert p.status()["interactions"] == 1

    def test_add_item(self):
        p = RecommenderPipeline()
        item = p.add_item("i1", {"x": 1.0})
        assert item.item_id == "i1"

    def test_train_user_cf(self):
        p = self._setup()
        p.train(RecommendMethod.USER_CF)
        assert RecommendMethod.USER_CF in p._trained_methods

    def test_recommend_user_cf(self):
        p = self._setup()
        p.train(RecommendMethod.USER_CF)
        result = p.recommend("u1", n=3, method=RecommendMethod.USER_CF)
        assert isinstance(result, RecommendationResult)
        assert result.user_id == "u1"
        assert result.method == RecommendMethod.USER_CF

    def test_train_item_cf(self):
        p = self._setup()
        p.train(RecommendMethod.ITEM_CF)
        result = p.recommend("u1", n=3, method=RecommendMethod.ITEM_CF)
        assert isinstance(result, RecommendationResult)

    def test_train_content(self):
        p = self._setup()
        p.train(RecommendMethod.CONTENT)
        result = p.recommend("u1", n=3, method=RecommendMethod.CONTENT)
        assert isinstance(result, RecommendationResult)

    def test_train_mf(self):
        p = self._setup()
        p.train(RecommendMethod.MF, n_factors=5, n_epochs=10)
        result = p.recommend("u1", n=3, method=RecommendMethod.MF)
        assert isinstance(result, RecommendationResult)

    def test_train_hybrid(self):
        p = self._setup()
        p.train(RecommendMethod.HYBRID, cf_weight=0.6, cbf_weight=0.4)
        result = p.recommend("u1", n=3, method=RecommendMethod.HYBRID)
        assert isinstance(result, RecommendationResult)

    def test_recommend_untrained_method(self):
        p = self._setup()
        # 学習していない手法 → 空リスト
        result = p.recommend("u1", n=3, method=RecommendMethod.USER_CF)
        assert result.items == []

    def test_evaluate_user_cf(self):
        p = self._setup()
        p.train(RecommendMethod.USER_CF, k_neighbors=3)
        test_ints = [make_interaction("u1", "i4"), make_interaction("u2", "i3")]
        report = p.evaluate(test_ints, k=3, method=RecommendMethod.USER_CF)
        assert isinstance(report, EvaluationReport)
        assert 0.0 <= report.precision_at_k <= 1.0
        assert 0.0 <= report.ndcg_at_k <= 1.0
        assert report.n_users == 2

    def test_evaluate_to_dict(self):
        p = self._setup()
        p.train(RecommendMethod.USER_CF)
        test_ints = [make_interaction("u1", "i4")]
        report = p.evaluate(test_ints, k=5)
        d = report.to_dict()
        assert "precision_at_k" in d
        assert "recall_at_k" in d
        assert "ndcg_at_k" in d
        assert "map_at_k" in d

    def test_status_after_train(self):
        p = self._setup()
        p.train(RecommendMethod.USER_CF)
        p.train(RecommendMethod.ITEM_CF)
        st = p.status()
        assert "user_cf" in st["trained_methods"]
        assert "item_cf" in st["trained_methods"]
        assert st["users"] == 4
