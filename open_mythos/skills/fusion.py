"""
Sprint 65 — Fusion マルチモデル融合 (OpenRouter Fusion Server Tool 移植)

参照: https://openrouter.ai/docs/guides/features/server-tools/fusion
      https://openrouter.ai/blog/announcements/fusion-beats-frontier/

仕組み（3 段パイプライン）:
  1. 候補モデル群 (candidates) が同一プロンプトに各自回答する
  2. 審査モデル (judge) が各回答を読み、構造化された分析結果を生成する
     （強み・弱み・スコア・ランキング・合成方針）
  3. 呼び出しモデル (caller) が分析結果に基づき最終回答を合成する

オブジェクト:
  FusionRole       : モデルの役割 (Candidate/Judge/Caller)
  CandidateSpec    : 候補モデルの指定 (label/preferred_provider/temperature)
  CandidateResponse: 候補モデル 1 件の生回答
  CandidateAnalysis: 審査モデルによる候補 1 件の構造化分析
  FusionAnalysis   : 審査モデルの全体分析 (候補分析リスト + ランキング + 合成方針)
  FusionConfig     : Fusion 設定
  FusionResult     : 最終結果 (final_answer + analysis + candidates + メタデータ)
  FusionAnalysisParser: 審査モデル JSON レスポンスのパーサ
  JudgeAnalyzer    : 審査ステージ (候補群 → FusionAnalysis)
  FusionEngine     : 3 段パイプライン全体のオーケストレーター
  FusionEngineFactory: from_env / from_mock / rule_based ファクトリ

設計方針:
  - llm_providers.MultiProviderRouter / LLMRequest / LLMResponse を再利用
  - LLM 不在時はヒューリスティック (rule-based) にフォールバック
  - 審査の構造化出力は JSON、パース失敗時は正規表現/ヒューリスティック
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enum / 設定
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FusionRole(str, Enum):
    CANDIDATE = "candidate"   # 候補モデル
    JUDGE     = "judge"       # 審査モデル
    CALLER    = "caller"      # 呼び出し（合成）モデル


@dataclass
class CandidateSpec:
    """候補モデルの指定"""
    label:              str
    preferred_provider: Optional[str] = None   # "claude" / "openai" / "openmythos"
    temperature:        float         = 0.7
    max_tokens:         int           = 512


@dataclass
class FusionConfig:
    """Fusion 設定"""
    candidates:        List[CandidateSpec] = field(default_factory=list)
    judge_provider:    Optional[str] = None   # 審査モデルのプロバイダー
    caller_provider:   Optional[str] = None   # 合成モデルのプロバイダー
    judge_temperature: float = 0.2            # 審査は低温で安定化
    caller_temperature: float = 0.5
    max_tokens:        int   = 1024
    fallback_on_error: bool  = True

    @classmethod
    def default(cls) -> "FusionConfig":
        """デフォルト 3 候補構成"""
        return cls(candidates=[
            CandidateSpec(label="candidate-1", preferred_provider="claude", temperature=0.7),
            CandidateSpec(label="candidate-2", preferred_provider="openai", temperature=0.7),
            CandidateSpec(label="candidate-3", preferred_provider="openmythos", temperature=0.9),
        ])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# データモデル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class CandidateResponse:
    """候補モデル 1 件の生回答"""
    label:         str
    text:          str
    provider_used: str   = "unknown"
    model:         str   = "unknown"
    latency_ms:    float = 0.0
    error:         Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label":         self.label,
            "text":          self.text,
            "provider_used": self.provider_used,
            "model":         self.model,
            "latency_ms":    round(self.latency_ms, 2),
            "error":         self.error,
            "success":       self.success,
        }


@dataclass
class CandidateAnalysis:
    """審査モデルによる候補 1 件の構造化分析"""
    label:      str
    score:      float                  # 0.0〜1.0
    strengths:  List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    key_points: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label":      self.label,
            "score":      round(self.score, 4),
            "strengths":  self.strengths,
            "weaknesses": self.weaknesses,
            "key_points": self.key_points,
        }


@dataclass
class FusionAnalysis:
    """審査モデルの全体分析結果"""
    candidate_analyses: List[CandidateAnalysis]
    ranking:            List[str]               # label の優先順（高評価順）
    synthesis_guidance: str = ""                # 合成方針
    judge_provider:     str = "unknown"

    def best_label(self) -> Optional[str]:
        return self.ranking[0] if self.ranking else None

    def get_analysis(self, label: str) -> Optional[CandidateAnalysis]:
        for ca in self.candidate_analyses:
            if ca.label == label:
                return ca
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_analyses": [ca.to_dict() for ca in self.candidate_analyses],
            "ranking":            self.ranking,
            "synthesis_guidance": self.synthesis_guidance,
            "judge_provider":     self.judge_provider,
        }


@dataclass
class FusionResult:
    """Fusion 最終結果"""
    final_answer:   str
    analysis:       FusionAnalysis
    candidates:     List[CandidateResponse]
    caller_provider: str   = "unknown"
    fallback_used:  bool  = False
    total_latency_ms: float = 0.0
    timestamp:      float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_answer":     self.final_answer,
            "analysis":         self.analysis.to_dict(),
            "candidates":       [c.to_dict() for c in self.candidates],
            "caller_provider":  self.caller_provider,
            "fallback_used":    self.fallback_used,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "timestamp":        self.timestamp,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# プロンプト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_JUDGE_SYSTEM = """\
あなたは複数の AI モデルの回答を評価する審査員です。
各候補回答を読み、客観的に分析してください。

出力は必ず以下の JSON 形式のみで返してください:
{
  "analyses": [
    {
      "label": "候補ラベル",
      "score": 0.0〜1.0,
      "strengths": ["強み1", "強み2"],
      "weaknesses": ["弱み1"],
      "key_points": ["回答の要点1", "要点2"]
    }
  ],
  "ranking": ["最良の候補ラベル", "次点", ...],
  "synthesis_guidance": "最終回答を合成する際の方針"
}

JSON 以外のテキストは出力しないでください。"""

_CALLER_SYSTEM = """\
あなたは複数の候補回答と審査分析をもとに、最良の最終回答を合成する統合者です。
審査分析の synthesis_guidance とランキングを踏まえ、各候補の強みを統合した
高品質な最終回答を生成してください。最終回答の本文のみを出力してください。"""


def _build_judge_prompt(question: str, candidates: List[CandidateResponse]) -> str:
    lines = [f"# 質問\n{question}", "", "# 候補回答"]
    for c in candidates:
        lines.append(f"\n## 候補: {c.label}\n{c.text}")
    lines.append("\n上記の候補回答を分析し、指定の JSON 形式で評価してください。")
    return "\n".join(lines)


def _build_caller_prompt(
    question: str,
    candidates: List[CandidateResponse],
    analysis: FusionAnalysis,
) -> str:
    lines = [f"# 質問\n{question}", "", "# 候補回答"]
    for c in candidates:
        lines.append(f"\n## 候補: {c.label}\n{c.text}")
    lines += [
        "",
        "# 審査分析",
        f"ランキング: {' > '.join(analysis.ranking)}",
        f"合成方針: {analysis.synthesis_guidance}",
        "",
        "上記を踏まえて最終回答を合成してください。",
    ]
    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FusionAnalysisParser
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FusionAnalysisParser:
    """審査モデルの JSON レスポンスを FusionAnalysis にパースする"""

    def parse(
        self,
        raw: str,
        candidate_labels: List[str],
        judge_provider: str = "unknown",
    ) -> FusionAnalysis:
        data = self._try_json(raw) or self._try_code_block(raw)
        if data:
            return self._from_dict(data, candidate_labels, judge_provider)
        # フォールバック: 全候補を均等スコアにする
        return self._fallback(candidate_labels, judge_provider)

    def _try_json(self, text: str) -> Optional[Dict]:
        try:
            obj = json.loads(text.strip())
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None

    def _try_code_block(self, text: str) -> Optional[Dict]:
        m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if m:
            return self._try_json(m.group(1))
        return None

    def _from_dict(
        self, data: Dict, labels: List[str], judge_provider: str
    ) -> FusionAnalysis:
        analyses: List[CandidateAnalysis] = []
        for item in data.get("analyses", []):
            if not isinstance(item, dict):
                continue
            analyses.append(CandidateAnalysis(
                label=str(item.get("label", "")),
                score=self._clamp_score(item.get("score", 0.0)),
                strengths=self._str_list(item.get("strengths")),
                weaknesses=self._str_list(item.get("weaknesses")),
                key_points=self._str_list(item.get("key_points")),
            ))

        # analyses が空ならフォールバック
        if not analyses:
            return self._fallback(labels, judge_provider)

        ranking = data.get("ranking", [])
        if not isinstance(ranking, list) or not ranking:
            # スコア降順でランキング生成
            ranking = [a.label for a in sorted(analyses, key=lambda x: -x.score)]
        else:
            ranking = [str(r) for r in ranking]

        return FusionAnalysis(
            candidate_analyses=analyses,
            ranking=ranking,
            synthesis_guidance=str(data.get("synthesis_guidance", "")),
            judge_provider=judge_provider,
        )

    def _fallback(self, labels: List[str], judge_provider: str) -> FusionAnalysis:
        analyses = [
            CandidateAnalysis(label=lbl, score=0.5) for lbl in labels
        ]
        return FusionAnalysis(
            candidate_analyses=analyses,
            ranking=list(labels),
            synthesis_guidance="（審査パース失敗のため全候補を均等扱い）",
            judge_provider=judge_provider,
        )

    @staticmethod
    def _clamp_score(v: Any) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, f))

    @staticmethod
    def _str_list(v: Any) -> List[str]:
        if isinstance(v, list):
            return [str(x) for x in v]
        return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ヒューリスティック審査（LLM 不在時のフォールバック）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _heuristic_score(text: str) -> float:
    """
    回答テキストの簡易品質スコア (0.0〜1.0)。
    長さ・文の数・構造を簡易評価する。
    """
    if not text:
        return 0.0
    length = len(text)
    sentences = len(re.split(r"[。.!?！？\n]+", text.strip()))
    # 長さ正規化（300 文字で頭打ち）
    length_score = min(1.0, length / 300)
    # 文の数（5 文で頭打ち）
    structure_score = min(1.0, sentences / 5)
    return round(0.6 * length_score + 0.4 * structure_score, 4)


class JudgeAnalyzer:
    """
    審査ステージ。候補回答群を読み FusionAnalysis を生成する。

    LLM (router) があれば審査モデルを使い、なければヒューリスティック評価。
    """

    def __init__(
        self,
        router: Optional[Any] = None,
        parser: Optional[FusionAnalysisParser] = None,
    ) -> None:
        self._router = router
        self._parser = parser or FusionAnalysisParser()

    def analyze(
        self,
        question: str,
        candidates: List[CandidateResponse],
        config: FusionConfig,
    ) -> FusionAnalysis:
        successful = [c for c in candidates if c.success]
        labels = [c.label for c in successful] or [c.label for c in candidates]

        if self._router is not None and self._llm_available():
            try:
                return self._analyze_via_llm(question, successful, config, labels)
            except Exception:
                if not config.fallback_on_error:
                    raise
        return self._analyze_heuristic(successful or candidates)

    def _llm_available(self) -> bool:
        try:
            return len(self._router.available_providers()) > 0
        except Exception:
            return False

    def _analyze_via_llm(
        self,
        question: str,
        candidates: List[CandidateResponse],
        config: FusionConfig,
        labels: List[str],
    ) -> FusionAnalysis:
        from open_mythos.skills.llm_providers import LLMRequest
        req = LLMRequest(
            prompt=_build_judge_prompt(question, candidates),
            system=_JUDGE_SYSTEM,
            max_tokens=config.max_tokens,
            temperature=config.judge_temperature,
        )
        preferred = self._to_provider(config.judge_provider)
        resp = self._router.complete(req, preferred=preferred)
        return self._parser.parse(resp.text, labels, judge_provider=resp.provider_used)

    def _analyze_heuristic(
        self, candidates: List[CandidateResponse]
    ) -> FusionAnalysis:
        analyses: List[CandidateAnalysis] = []
        for c in candidates:
            score = _heuristic_score(c.text)
            analyses.append(CandidateAnalysis(
                label=c.label,
                score=score,
                strengths=["十分な情報量"] if score >= 0.6 else [],
                weaknesses=["情報が簡潔すぎる"] if score < 0.4 else [],
                key_points=[],
            ))
        ranking = [a.label for a in sorted(analyses, key=lambda x: -x.score)]
        return FusionAnalysis(
            candidate_analyses=analyses,
            ranking=ranking,
            synthesis_guidance="ヒューリスティック評価（長さ・構造ベース）",
            judge_provider="heuristic",
        )

    @staticmethod
    def _to_provider(name: Optional[str]):
        if not name:
            return None
        from open_mythos.skills.llm_providers import ProviderType
        try:
            return ProviderType(name)
        except ValueError:
            return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FusionEngine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FusionEngine:
    """
    Fusion 3 段パイプラインのオーケストレーター。

    1. fan-out: 候補モデル群に同一プロンプトを投げて回答収集
    2. judge:   審査モデルで構造化分析を生成
    3. synth:   呼び出しモデルで最終回答を合成

    Usage:
        engine = FusionEngineFactory.from_env()
        result = engine.run("Pythonでフィボナッチを書いて")
        print(result.final_answer)
    """

    def __init__(
        self,
        config: Optional[FusionConfig] = None,
        router: Optional[Any] = None,
        judge: Optional[JudgeAnalyzer] = None,
    ) -> None:
        self.config = config or FusionConfig.default()
        self._router = router
        self._judge = judge or JudgeAnalyzer(router=router)

    @property
    def has_llm(self) -> bool:
        if self._router is None:
            return False
        try:
            return len(self._router.available_providers()) > 0
        except Exception:
            return False

    def run(self, question: str, system: Optional[str] = None) -> FusionResult:
        """Fusion パイプラインを実行する"""
        t0 = time.time()

        # 1) 候補収集
        candidates = self._gather_candidates(question, system)

        # 2) 審査
        analysis = self._judge.analyze(question, candidates, self.config)

        # 3) 合成
        final_answer, caller_provider, fallback = self._synthesize(
            question, candidates, analysis
        )

        return FusionResult(
            final_answer=final_answer,
            analysis=analysis,
            candidates=candidates,
            caller_provider=caller_provider,
            fallback_used=fallback,
            total_latency_ms=(time.time() - t0) * 1000,
        )

    def run_stream(self, question: str, system: Optional[str] = None):
        """
        Fusion パイプラインを段階イベントとしてストリーミングする (Sprint 66A)。

        yield する dict のスキーマ:
          {"stage": "candidates", "data": {...}}  — 候補収集完了
          {"stage": "analysis",   "data": {...}}  — 審査完了
          {"stage": "delta",      "data": {"text": "..."}}  — 最終回答チャンク
          {"stage": "done",       "data": {...}}  — 完了 (FusionResult 全体)
          {"stage": "error",      "data": {"error": "..."}}  — エラー

        最終回答は文字チャンクで delta として流す（既存 streaming.py と整合）。
        """
        t0 = time.time()
        try:
            # 1) 候補収集
            candidates = self._gather_candidates(question, system)
            yield {
                "stage": "candidates",
                "data": {"candidates": [c.to_dict() for c in candidates]},
            }

            # 2) 審査
            analysis = self._judge.analyze(question, candidates, self.config)
            yield {"stage": "analysis", "data": analysis.to_dict()}

            # 3) 合成
            final_answer, caller_provider, fallback = self._synthesize(
                question, candidates, analysis
            )

            # 最終回答を文字チャンクで delta 送出
            for chunk in self._chunk_text(final_answer):
                yield {"stage": "delta", "data": {"text": chunk}}

            result = FusionResult(
                final_answer=final_answer,
                analysis=analysis,
                candidates=candidates,
                caller_provider=caller_provider,
                fallback_used=fallback,
                total_latency_ms=(time.time() - t0) * 1000,
            )
            yield {"stage": "done", "data": result.to_dict()}
        except Exception as e:  # pragma: no cover - 安全網
            yield {"stage": "error", "data": {"error": str(e)}}

    @staticmethod
    def _chunk_text(text: str, size: int = 32):
        """テキストを size 文字ずつのチャンクに分割する"""
        for i in range(0, len(text), size):
            yield text[i:i + size]

    # ---- 内部実装 ----

    def _gather_candidates(
        self, question: str, system: Optional[str]
    ) -> List[CandidateResponse]:
        specs = self.config.candidates or FusionConfig.default().candidates
        results: List[CandidateResponse] = []

        for spec in specs:
            if self._router is not None and self.has_llm:
                results.append(self._call_candidate(spec, question, system))
            else:
                # LLM 不在: プレースホルダ回答（ヒューリスティック審査の対象）
                results.append(CandidateResponse(
                    label=spec.label,
                    text=f"[{spec.label}] {question} への回答（LLM 未接続のスタブ）",
                    provider_used="stub",
                    model="stub",
                ))
        return results

    def _call_candidate(
        self, spec: CandidateSpec, question: str, system: Optional[str]
    ) -> CandidateResponse:
        from open_mythos.skills.llm_providers import LLMRequest, ProviderType
        req = LLMRequest(
            prompt=question,
            system=system,
            max_tokens=spec.max_tokens,
            temperature=spec.temperature,
        )
        preferred = None
        if spec.preferred_provider:
            try:
                preferred = ProviderType(spec.preferred_provider)
            except ValueError:
                preferred = None
        try:
            t0 = time.time()
            resp = self._router.complete(req, preferred=preferred)
            return CandidateResponse(
                label=spec.label,
                text=resp.text,
                provider_used=resp.provider_used,
                model=resp.model,
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return CandidateResponse(
                label=spec.label,
                text="",
                provider_used="error",
                error=str(e),
            )

    def _synthesize(
        self,
        question: str,
        candidates: List[CandidateResponse],
        analysis: FusionAnalysis,
    ) -> tuple:
        """最終回答を合成する。戻り値: (final_answer, caller_provider, fallback_used)"""
        if self._router is not None and self.has_llm:
            try:
                from open_mythos.skills.llm_providers import LLMRequest, ProviderType
                req = LLMRequest(
                    prompt=_build_caller_prompt(question, candidates, analysis),
                    system=_CALLER_SYSTEM,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.caller_temperature,
                )
                preferred = None
                if self.config.caller_provider:
                    try:
                        preferred = ProviderType(self.config.caller_provider)
                    except ValueError:
                        preferred = None
                resp = self._router.complete(req, preferred=preferred)
                if resp.text:
                    return resp.text, resp.provider_used, False
            except Exception:
                if not self.config.fallback_on_error:
                    raise

        # フォールバック: 最良候補をそのまま採用
        best = self._best_candidate(candidates, analysis)
        return (best.text if best else ""), "fallback:best-candidate", True

    @staticmethod
    def _best_candidate(
        candidates: List[CandidateResponse], analysis: FusionAnalysis
    ) -> Optional[CandidateResponse]:
        best_label = analysis.best_label()
        if best_label:
            for c in candidates:
                if c.label == best_label and c.success:
                    return c
        # ランキングで見つからなければ成功した最初の候補
        for c in candidates:
            if c.success:
                return c
        return candidates[0] if candidates else None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ファクトリ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FusionEngineFactory:
    """FusionEngine を構築するファクトリ"""

    @classmethod
    def from_env(
        cls,
        config: Optional[FusionConfig] = None,
        llm: Any = None,
    ) -> FusionEngine:
        """環境変数から API キーを読み込んで構築する"""
        from open_mythos.skills.llm_providers import MultiProviderRouter
        router = MultiProviderRouter.from_env(llm=llm)
        return FusionEngine(config=config, router=router)

    @classmethod
    def from_mock(
        cls,
        responses: List[str],
        config: Optional[FusionConfig] = None,
    ) -> FusionEngine:
        """
        テスト用モックで構築する。

        responses: router.complete が順に返すテキスト列。
        典型的には [候補1, 候補2, 候補3, 審査JSON, 最終回答] の順。
        """
        router = _MockFusionRouter(responses)
        return FusionEngine(config=config, router=router)

    @classmethod
    def rule_based(cls, config: Optional[FusionConfig] = None) -> FusionEngine:
        """LLM なし（ヒューリスティック専用）で構築する"""
        return FusionEngine(config=config, router=None)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# テスト用モックルーター
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _MockFusionRouter:
    """テスト用ルーター。responses を順に返す。"""

    def __init__(self, responses: List[str]) -> None:
        self._responses = responses
        self._idx = 0

    def available_providers(self) -> List[str]:
        return ["mock"]

    def complete(self, req: Any, preferred: Any = None) -> Any:
        from open_mythos.skills.llm_providers import LLMResponse
        text = self._responses[self._idx % len(self._responses)] if self._responses else ""
        self._idx += 1
        return LLMResponse(text=text, provider_used="mock", model="mock-model")
