"""
Sprint 60-C — A/Bテスト基盤

複数の広告コピー案を並走させ、統計的有意性を検定して最良案を選定する
A/Bテストフレームワーク。

オブジェクト:
  VariantStatus  : バリアント状態 (ACTIVE / PAUSED / WINNER / LOSER)
  TestStatus     : テスト状態 (DRAFT / RUNNING / COMPLETED / STOPPED)
  StatMethod     : 統計検定手法 (T_TEST / CHI_SQUARE / BAYESIAN)
  Variant        : A/Bテストの1案（コピー + インプレッション/クリック/コンバージョン）
  ABTest         : テスト全体の定義と状態管理
  ABTestStore    : ABTest CRUD ストア
  StatEngine     : Welch t-test / χ²検定 / ベイズ推定（bernoulli）
  ABTestAnalyzer : バリアント分析・勝者判定・リフト計算
  ABTestRunner   : テストライフサイクル管理 (start/record/stop/analyze)
  ABReportEngine : Markdown / JSON レポート生成

設計方針:
  - scipy / sklearn 非依存（pure Python 統計実装）
  - Welch t-test は正規分布近似で p 値を計算
  - ベイズ推定は Beta 分布の期待値で CVR を推定し Thompson sampling で勝者判定
  - 全データはメモリ内保持（永続化は外部 DB に委ねる設計）
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enum 層
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VariantStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    WINNER = "winner"
    LOSER  = "loser"


class TestStatus(str, Enum):
    DRAFT     = "draft"
    RUNNING   = "running"
    COMPLETED = "completed"
    STOPPED   = "stopped"


class StatMethod(str, Enum):
    T_TEST     = "t_test"
    CHI_SQUARE = "chi_square"
    BAYESIAN   = "bayesian"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# バリアント / テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Variant:
    """A/Bテスト内の1バリアント（1案）"""
    id:          str
    name:        str
    copy:        str                      # 広告コピーテキスト
    status:      VariantStatus           = VariantStatus.ACTIVE
    impressions: int                     = 0
    clicks:      int                     = 0
    conversions: int                     = 0
    metadata:    Dict[str, Any]          = field(default_factory=dict)
    created_at:  int                     = field(default_factory=lambda: int(time.time()))

    @property
    def ctr(self) -> float:
        return self.clicks / self.impressions if self.impressions > 0 else 0.0

    @property
    def cvr(self) -> float:
        return self.conversions / self.clicks if self.clicks > 0 else 0.0

    @property
    def conversion_rate(self) -> float:
        """インプレッション基準のコンバージョン率"""
        return self.conversions / self.impressions if self.impressions > 0 else 0.0

    def record(self, impressions: int = 0, clicks: int = 0, conversions: int = 0) -> None:
        self.impressions += max(0, impressions)
        self.clicks      += max(0, clicks)
        self.conversions += max(0, conversions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":          self.id,
            "name":        self.name,
            "copy":        self.copy,
            "status":      self.status.value,
            "impressions": self.impressions,
            "clicks":      self.clicks,
            "conversions": self.conversions,
            "ctr":         round(self.ctr, 4),
            "cvr":         round(self.cvr, 4),
            "conversion_rate": round(self.conversion_rate, 4),
            "metadata":    self.metadata,
            "created_at":  self.created_at,
        }


@dataclass
class ABTest:
    """A/Bテスト定義"""
    id:          str
    name:        str
    variants:    List[Variant]           = field(default_factory=list)
    status:      TestStatus             = TestStatus.DRAFT
    method:      StatMethod             = StatMethod.CHI_SQUARE
    confidence:  float                  = 0.95     # 有意水準（1 - α）
    min_samples: int                    = 100      # 判定に必要な最低サンプル数
    description: str                    = ""
    winner_id:   Optional[str]          = None
    start_at:    Optional[int]          = None
    end_at:      Optional[int]          = None
    created_at:  int                    = field(default_factory=lambda: int(time.time()))

    def get_variant(self, variant_id: str) -> Optional[Variant]:
        return next((v for v in self.variants if v.id == variant_id), None)

    def add_variant(self, name: str, copy: str) -> Variant:
        v = Variant(id=str(uuid.uuid4()), name=name, copy=copy)
        self.variants.append(v)
        return v

    def is_ready(self) -> bool:
        """最低サンプル数を満たしているか"""
        return all(v.impressions >= self.min_samples for v in self.variants)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":          self.id,
            "name":        self.name,
            "variants":    [v.to_dict() for v in self.variants],
            "status":      self.status.value,
            "method":      self.method.value,
            "confidence":  self.confidence,
            "min_samples": self.min_samples,
            "description": self.description,
            "winner_id":   self.winner_id,
            "start_at":    self.start_at,
            "end_at":      self.end_at,
            "created_at":  self.created_at,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ストア
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ABTestStore:
    """A/Bテスト CRUD ストア（メモリ内）"""

    def __init__(self) -> None:
        self._tests: Dict[str, ABTest] = {}

    def create(self, name: str, description: str = "",
               method: StatMethod = StatMethod.CHI_SQUARE,
               confidence: float = 0.95, min_samples: int = 100) -> ABTest:
        tid = str(uuid.uuid4())
        test = ABTest(id=tid, name=name, description=description,
                      method=method, confidence=confidence, min_samples=min_samples)
        self._tests[tid] = test
        return test

    def get(self, test_id: str) -> Optional[ABTest]:
        return self._tests.get(test_id)

    def list_all(self, status: Optional[TestStatus] = None) -> List[ABTest]:
        tests = list(self._tests.values())
        if status is not None:
            tests = [t for t in tests if t.status == status]
        return tests

    def delete(self, test_id: str) -> bool:
        if test_id in self._tests:
            del self._tests[test_id]
            return True
        return False

    def __len__(self) -> int:
        return len(self._tests)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 統計エンジン
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class StatResult:
    """統計検定の結果"""
    method:      str
    p_value:     float
    significant: bool
    confidence:  float
    effect_size: float    # Cohen's h (proportion difference) or t-statistic
    winner_idx:  Optional[int]  = None   # variants リスト内のインデックス

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method":      self.method,
            "p_value":     round(self.p_value, 6),
            "significant": self.significant,
            "confidence":  self.confidence,
            "effect_size": round(self.effect_size, 4),
            "winner_idx":  self.winner_idx,
        }


class StatEngine:
    """Welch t-test / χ²検定 / ベイズ推定（pure Python）"""

    # ── χ²検定 ────────────────────────────────────────────────────

    def chi_square_test(self, a: Variant, b: Variant,
                        confidence: float = 0.95) -> StatResult:
        """2バリアントの CVR 差の χ²検定"""
        n_a = a.impressions
        n_b = b.impressions
        c_a = a.conversions
        c_b = b.conversions

        if n_a == 0 or n_b == 0:
            return StatResult("chi_square", 1.0, False, confidence, 0.0)

        total = n_a + n_b
        total_conv = c_a + c_b
        if total_conv == 0:
            return StatResult("chi_square", 1.0, False, confidence, 0.0)

        # 2×2 分割表: [[c_a, n_a-c_a], [c_b, n_b-c_b]]
        e_a = n_a * total_conv / total
        e_b = n_b * total_conv / total
        e_na = n_a * (total - total_conv) / total
        e_nb = n_b * (total - total_conv) / total

        def safe_chi(obs: float, exp: float) -> float:
            return (obs - exp) ** 2 / exp if exp > 0 else 0.0

        chi2 = (safe_chi(c_a, e_a) + safe_chi(n_a - c_a, e_na) +
                safe_chi(c_b, e_b) + safe_chi(n_b - c_b, e_nb))

        p_value = self._chi2_sf(chi2, df=1)
        alpha   = 1 - confidence
        significant = p_value < alpha

        # effect_size: Cohen's h
        p1 = c_a / n_a
        p2 = c_b / n_b
        effect = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))

        winner_idx = None
        if significant:
            winner_idx = 0 if a.conversion_rate >= b.conversion_rate else 1

        return StatResult("chi_square", p_value, significant, confidence,
                          abs(effect), winner_idx)

    # ── Welch t-test ─────────────────────────────────────────────

    def welch_t_test(self, a: Variant, b: Variant,
                     confidence: float = 0.95) -> StatResult:
        """2バリアントの CTR 差の Welch t-test（正規分布近似）"""
        if a.impressions < 2 or b.impressions < 2:
            return StatResult("t_test", 1.0, False, confidence, 0.0)

        p1, n1 = a.ctr, a.impressions
        p2, n2 = b.ctr, b.impressions
        var1 = p1 * (1 - p1) / n1
        var2 = p2 * (1 - p2) / n2
        se   = math.sqrt(var1 + var2)

        if se == 0:
            return StatResult("t_test", 1.0, False, confidence, 0.0)

        t_stat = (p1 - p2) / se
        # Welch-Satterthwaite 自由度
        if var1 == 0 and var2 == 0:
            df = 1.0
        else:
            df = (var1 + var2) ** 2 / (
                (var1 ** 2 / (n1 - 1) if n1 > 1 else 0) +
                (var2 ** 2 / (n2 - 1) if n2 > 1 else 0) + 1e-12
            )
        p_value = 2 * self._t_sf(abs(t_stat), df)
        alpha   = 1 - confidence
        significant = p_value < alpha

        winner_idx = None
        if significant:
            winner_idx = 0 if a.ctr >= b.ctr else 1

        return StatResult("t_test", p_value, significant, confidence,
                          abs(t_stat), winner_idx)

    # ── ベイズ推定 ────────────────────────────────────────────────

    def bayesian_test(self, a: Variant, b: Variant,
                      confidence: float = 0.95, n_samples: int = 10000) -> StatResult:
        """Beta 分布の期待値で CVR 推定、モンテカルロで P(A>B) を計算"""
        # Beta(alpha, beta) の期待値 = alpha / (alpha + beta)
        alpha_a = a.conversions + 1
        beta_a  = max(a.impressions - a.conversions, 0) + 1
        alpha_b = b.conversions + 1
        beta_b  = max(b.impressions - b.conversions, 0) + 1

        # 解析的近似: P(A > B) ≈ 正規分布近似
        mu_a  = alpha_a / (alpha_a + beta_a)
        mu_b  = alpha_b / (alpha_b + beta_b)
        var_a = alpha_a * beta_a / ((alpha_a + beta_a) ** 2 * (alpha_a + beta_a + 1))
        var_b = alpha_b * beta_b / ((alpha_b + beta_b) ** 2 * (alpha_b + beta_b + 1))

        se = math.sqrt(var_a + var_b)
        if se == 0:
            prob_a_wins = 0.5
        else:
            z = (mu_a - mu_b) / se
            prob_a_wins = self._normal_cdf(z)

        # p_value を「1 - max(P(A>B), P(B>A))」で近似
        p_value = 1.0 - max(prob_a_wins, 1 - prob_a_wins)
        significant = max(prob_a_wins, 1 - prob_a_wins) >= confidence
        effect = mu_a - mu_b

        winner_idx = None
        if significant:
            winner_idx = 0 if prob_a_wins >= 0.5 else 1

        return StatResult("bayesian", p_value, significant, confidence,
                          effect, winner_idx)

    # ── 累積分布関数近似 ─────────────────────────────────────────

    def _chi2_sf(self, x: float, df: int) -> float:
        """χ²分布の上側確率 (df=1 用 gamma 近似)"""
        if x <= 0:
            return 1.0
        # df=1: P(χ²>x) = 2*(1-Φ(√x))
        if df == 1:
            return 2 * (1 - self._normal_cdf(math.sqrt(x)))
        # df=2: P(χ²>x) = exp(-x/2)
        if df == 2:
            return math.exp(-x / 2)
        # 一般: Wilson-Hilferty 近似
        z = ((x / df) ** (1/3) - (1 - 2/(9*df))) / math.sqrt(2/(9*df))
        return 1 - self._normal_cdf(z)

    def _t_sf(self, t: float, df: float) -> float:
        """t 分布の上側確率（片側）: 正規分布近似（df>30 で精度良好）"""
        if df > 30:
            return 1 - self._normal_cdf(t)
        # 小 df: ベータ関数近似
        x = df / (df + t * t)
        return 0.5 * self._beta_inc(df / 2, 0.5, x)

    def _normal_cdf(self, z: float) -> float:
        """標準正規分布の CDF（Abramowitz & Stegun 近似）"""
        if z < -6:
            return 0.0
        if z > 6:
            return 1.0
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    def _beta_inc(self, a: float, b: float, x: float) -> float:
        """不完全ベータ関数 I_x(a,b) の連分数近似"""
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        # Lentz の連分数展開（最大50項）
        lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
        # 連分数係数
        def d_m(m: int) -> float:
            return m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m))
        def d_mp1(m: int) -> float:
            return -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))

        f = 1.0
        C = 1.0
        D = 1 - (a + b) * x / (a + 1)
        D = 1 / D if abs(D) > 1e-30 else 1e30
        C = 1.0
        f = D
        for m in range(1, 51):
            dm = d_m(m)
            D = 1 + dm * D
            C = 1 + dm / C
            D = 1 / D if abs(D) > 1e-30 else 1e30
            C = C if abs(C) > 1e-30 else 1e30
            delta = C * D
            f *= delta
            dm2 = d_mp1(m)
            D = 1 + dm2 * D
            C = 1 + dm2 / C
            D = 1 / D if abs(D) > 1e-30 else 1e30
            C = C if abs(C) > 1e-30 else 1e30
            delta = C * D
            f *= delta
            if abs(delta - 1) < 1e-8:
                break
        return front * f


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 分析エンジン
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class AnalysisResult:
    """A/Bテスト分析結果"""
    test_id:         str
    stat_result:     StatResult
    winner:          Optional[Variant]
    lift_ctr:        Optional[float]     # CTR のリフト（%）
    lift_cvr:        Optional[float]     # CVR のリフト（%）
    recommendation:  str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id":        self.test_id,
            "stat_result":    self.stat_result.to_dict(),
            "winner":         self.winner.to_dict() if self.winner else None,
            "lift_ctr":       round(self.lift_ctr, 4) if self.lift_ctr is not None else None,
            "lift_cvr":       round(self.lift_cvr, 4) if self.lift_cvr is not None else None,
            "recommendation": self.recommendation,
        }


class ABTestAnalyzer:
    """バリアント分析・勝者判定・リフト計算"""

    def __init__(self, engine: Optional[StatEngine] = None) -> None:
        self.engine = engine or StatEngine()

    def analyze(self, test: ABTest) -> AnalysisResult:
        """テスト全体を分析してベスト案を判定"""
        if len(test.variants) < 2:
            return AnalysisResult(
                test_id=test.id,
                stat_result=StatResult(test.method.value, 1.0, False, test.confidence, 0.0),
                winner=None, lift_ctr=None, lift_cvr=None,
                recommendation="バリアントが2つ以上必要です",
            )

        # control = variants[0], treatment = variants[1] (2バリアントを想定)
        a, b = test.variants[0], test.variants[1]
        stat = self._run_stat(test.method, a, b, test.confidence)

        winner: Optional[Variant] = None
        lift_ctr = lift_cvr = None

        if stat.significant and stat.winner_idx is not None:
            winner = test.variants[stat.winner_idx]
            loser  = test.variants[1 - stat.winner_idx]
            lift_ctr = self._lift(winner.ctr, loser.ctr)
            lift_cvr = self._lift(winner.cvr, loser.cvr)
            rec = (f"バリアント「{winner.name}」が有意に優れています "
                   f"(p={stat.p_value:.4f}, CTR lift={lift_ctr:+.1%})")
        elif not test.is_ready():
            rec = f"サンプル不足です（最低 {test.min_samples} インプレッション必要）"
        else:
            rec = "有意差なし — テストを継続するか、コピーを見直してください"

        return AnalysisResult(
            test_id=test.id,
            stat_result=stat,
            winner=winner,
            lift_ctr=lift_ctr,
            lift_cvr=lift_cvr,
            recommendation=rec,
        )

    def _run_stat(self, method: StatMethod, a: Variant, b: Variant,
                  confidence: float) -> StatResult:
        if method == StatMethod.T_TEST:
            return self.engine.welch_t_test(a, b, confidence)
        elif method == StatMethod.BAYESIAN:
            return self.engine.bayesian_test(a, b, confidence)
        else:
            return self.engine.chi_square_test(a, b, confidence)

    @staticmethod
    def _lift(winner_val: float, baseline_val: float) -> float:
        if baseline_val == 0:
            return 0.0
        return (winner_val - baseline_val) / baseline_val


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# テストランナー
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ABTestRunner:
    """テストライフサイクル管理 (start / record / stop / analyze)"""

    def __init__(self,
                 store:    Optional[ABTestStore]    = None,
                 analyzer: Optional[ABTestAnalyzer] = None) -> None:
        self.store    = store    if store    is not None else ABTestStore()
        self.analyzer = analyzer if analyzer is not None else ABTestAnalyzer()

    def start(self, test_id: str) -> Optional[ABTest]:
        test = self.store.get(test_id)
        if test is None:
            return None
        if len(test.variants) < 2:
            raise ValueError("A/Bテストにはバリアントが2つ以上必要です")
        test.status   = TestStatus.RUNNING
        test.start_at = int(time.time())
        return test

    def record(self, test_id: str, variant_id: str,
               impressions: int = 0, clicks: int = 0, conversions: int = 0) -> Optional[Variant]:
        test = self.store.get(test_id)
        if test is None:
            return None
        variant = test.get_variant(variant_id)
        if variant is None:
            return None
        variant.record(impressions=impressions, clicks=clicks, conversions=conversions)
        return variant

    def stop(self, test_id: str) -> Optional[ABTest]:
        test = self.store.get(test_id)
        if test is None:
            return None
        test.status = TestStatus.STOPPED
        test.end_at = int(time.time())
        return test

    def analyze(self, test_id: str) -> Optional[AnalysisResult]:
        test = self.store.get(test_id)
        if test is None:
            return None
        result = self.analyzer.analyze(test)

        # 勝者・敗者をバリアントに反映
        if result.winner is not None:
            for v in test.variants:
                v.status = VariantStatus.WINNER if v.id == result.winner.id else VariantStatus.LOSER
            test.winner_id = result.winner.id
            test.status    = TestStatus.COMPLETED
            test.end_at    = int(time.time())

        return result

    def auto_analyze_and_complete(self, test_id: str) -> Optional[AnalysisResult]:
        """サンプル数が揃っていれば自動解析・完了処理"""
        test = self.store.get(test_id)
        if test is None:
            return None
        if not test.is_ready():
            return None
        return self.analyze(test_id)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# レポートエンジン
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ABReportEngine:
    """A/Bテストレポート生成（Markdown / JSON）"""

    def to_markdown(self, test: ABTest,
                    analysis: Optional[AnalysisResult] = None) -> str:
        lines = [
            f"# A/Bテストレポート: {test.name}",
            f"",
            f"**ステータス**: {test.status.value}  |  **手法**: {test.method.value}  |  **有意水準**: {test.confidence:.0%}",
            f"",
            f"## バリアント比較",
            f"| バリアント | インプレッション | CTR | CVR | コンバージョン |",
            f"|----------|--------------|-----|-----|------------|",
        ]
        for v in test.variants:
            marker = " ⭐" if (test.winner_id and v.id == test.winner_id) else ""
            lines.append(
                f"| {v.name}{marker} | {v.impressions:,} | "
                f"{v.ctr:.2%} | {v.cvr:.2%} | {v.conversions:,} |"
            )
        lines.append("")

        if analysis:
            s = analysis.stat_result
            lines += [
                f"## 統計検定結果",
                f"- **p 値**: {s.p_value:.6f}",
                f"- **有意差**: {'あり ✅' if s.significant else 'なし ❌'}",
                f"- **効果量**: {s.effect_size:.4f}",
                f"",
                f"## 推奨",
                f"{analysis.recommendation}",
                f"",
            ]
            if analysis.lift_ctr is not None:
                lines.append(f"- CTR リフト: {analysis.lift_ctr:+.1%}")
            if analysis.lift_cvr is not None:
                lines.append(f"- CVR リフト: {analysis.lift_cvr:+.1%}")

        return "\n".join(lines)

    def to_json(self, test: ABTest,
                analysis: Optional[AnalysisResult] = None) -> Dict[str, Any]:
        return {
            "report_type": "ab_test",
            "test":        test.to_dict(),
            "analysis":    analysis.to_dict() if analysis else None,
        }
