"""
Sprint 59 — 自律脆弱性スキャン

anthropics/defending-code-reference-harness (external/defending-code-harness/) の
4ステージパイプラインを Python-native に適用したセキュリティスキャンフレームワーク。

ソース: https://github.com/anthropics/defending-code-reference-harness
ライセンス: Apache-2.0 (Anthropic PBC)

ハーネスとの対応:
  TargetConfig   → ScanTarget        スキャン対象コードベース設定
  CrashArtifact  → VulnFinding       脆弱性 1 件 (PoC の代わりに証跡コード)
  GraderVerdict  → VerifyVerdict     5 基準スコアリング (passed / score / criteria)
  PatchVerdict   → PatchCandidate    T0/T1/T2 検証ラダー (T0:構文 / T1:脆弱消滅 / T2:テスト)
  ReportVerdict  → ScanReport        悪用可能性分析 (section_scores / rubric / severity)
  RunResult      → ScanSession       1 回のスキャン全体 (findings + patches + status)

設計方針:
  - harness の find→grade→judge→report→patch 5 ステージを Python 静的解析に移植
  - Docker / ASAN 不要: AST + regex ベースのパターンマッチで Python コードを対象
  - PatchVerdict の T0/T1/T2 ラダーを PatchCandidate に継承
  - GraderVerdict の 5 基準を VerifyVerdict.make() で生成
  - 全データはメモリ内保持（永続化は外部 DB に委ねる設計）
  - known_findings で既知脆弱性を除外 (harness の known_bugs dedup に対応)
"""
from __future__ import annotations

import ast
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enums
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VulnSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"
    INFO     = "info"

    @property
    def score(self) -> int:
        """ソート・比較用スコア (高いほど深刻)。harness の severity_rating に対応。"""
        return {"critical": 5, "high": 4, "medium": 3, "low": 2, "info": 1}[self.value]


class VulnCategory(str, Enum):
    INJECTION = "injection"   # SQL / コマンド / コードインジェクション
    AUTH      = "auth"        # 認証 / 認可の問題
    CRYPTO    = "crypto"      # 弱い暗号化
    MEMORY    = "memory"      # メモリ安全性 (pickle / marshal)
    LOGIC     = "logic"       # ロジック上の欠陥
    DOS       = "dos"         # サービス拒否
    OTHER     = "other"


class PatchStatus(str, Enum):
    PENDING  = "pending"
    APPLIED  = "applied"
    VERIFIED = "verified"
    REJECTED = "rejected"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ScanTarget  (harness: TargetConfig)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ScanTarget:
    """スキャン対象コードベースの設定。harness の TargetConfig に対応。

    harness の Dockerfile / image_tag / binary_path / source_root は
    Python スキャン向けに path / language / focus_areas / test_command へ置換。
    """
    name:           str
    path:           str                        # コードベースルートパス
    language:       str            = "python"
    version:        Optional[str]  = None
    focus_areas:    List[str]      = field(default_factory=list)   # harness: focus_areas
    known_findings: List[str]      = field(default_factory=list)   # harness: known_bugs
    attack_surface: Optional[str]  = None
    test_command:   Optional[str]  = None      # T2 回帰テスト; None → T2 skip
    notes:          str            = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name":           self.name,
            "path":           self.path,
            "language":       self.language,
            "version":        self.version,
            "focus_areas":    self.focus_areas,
            "known_findings": self.known_findings,
            "attack_surface": self.attack_surface,
            "test_command":   self.test_command,
            "notes":          self.notes,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VulnFinding  (harness: CrashArtifact)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class VulnFinding:
    """脆弱性 1 件。harness の CrashArtifact に対応。

    poc_bytes / exit_code は静的解析では不要なため、
    evidence (証跡コード) と reproduction_steps (再現手順) に置換。
    """
    id:                  str
    title:               str
    category:            VulnCategory   = VulnCategory.OTHER
    severity:            VulnSeverity   = VulnSeverity.MEDIUM
    description:         str            = ""
    file_path:           str            = ""
    line_start:          int            = 0
    line_end:            int            = 0
    evidence:            str            = ""    # harness: crash_output
    reproduction_steps:  str            = ""    # harness: reproduction_command
    cwe_id:              Optional[str]  = None  # CWE-89 など
    dup_check:           Optional[str]  = None  # harness: dup_check
    created_at:          int            = field(default_factory=lambda: int(time.time()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":                 self.id,
            "title":              self.title,
            "category":           self.category.value,
            "severity":           self.severity.value,
            "description":        self.description,
            "file_path":          self.file_path,
            "line_start":         self.line_start,
            "line_end":           self.line_end,
            "evidence":           self.evidence,
            "reproduction_steps": self.reproduction_steps,
            "cwe_id":             self.cwe_id,
            "dup_check":          self.dup_check,
            "created_at":         self.created_at,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VerifyVerdict  (harness: GraderVerdict)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class VerifyVerdict:
    """5 基準スコアリング。harness の GraderVerdict に対応。

    5 基準: reproducible / has_evidence / severity_stated / not_duplicate / exploitable
    score = passed_count / 5, passed = score >= 0.6 (3/5 以上)
    """
    passed:   bool
    score:    float                  # 0.0–1.0
    criteria: Dict[str, bool]        # 5 criteria
    evidence: str                    # grader summary

    @classmethod
    def make(
        cls,
        reproducible:    bool,
        has_evidence:    bool,
        severity_stated: bool,
        not_duplicate:   bool,
        exploitable:     bool,
        evidence:        str = "",
    ) -> "VerifyVerdict":
        criteria = {
            "reproducible":    reproducible,
            "has_evidence":    has_evidence,
            "severity_stated": severity_stated,
            "not_duplicate":   not_duplicate,
            "exploitable":     exploitable,
        }
        passed_count = sum(criteria.values())
        score = passed_count / len(criteria)
        return cls(
            passed=score >= 0.6,
            score=score,
            criteria=criteria,
            evidence=evidence,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed":   self.passed,
            "score":    self.score,
            "criteria": self.criteria,
            "evidence": self.evidence,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PatchCandidate  (harness: PatchVerdict T0/T1/T2/T3 ラダー)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class PatchCandidate:
    """パッチ候補と T0/T1/T2 検証ラダー。harness の PatchVerdict に対応。

    T0: パッチ構文が有効か      (harness: t0_builds = コンパイル成功)
    T1: 脆弱パターンが消滅したか (harness: t1_poc_stops = クラッシュ消滅)
    T2: 既存テストが通るか       (harness: t2_tests_pass)
    T3: スタイルスコア           (harness: t3_style_score, advisory only)
    """
    finding_id:     str
    original_code:  str
    patched_code:   str
    explanation:    str
    confidence:     float           = 0.5
    status:         PatchStatus     = PatchStatus.PENDING

    t0_syntax_valid:  Optional[bool]  = None   # harness: t0_builds
    t1_vuln_gone:     Optional[bool]  = None   # harness: t1_poc_stops
    t2_tests_pass:    Optional[bool]  = None   # harness: t2_tests_pass
    t3_style_score:   Optional[float] = None   # harness: t3_style_score (advisory)

    evidence:    Dict[str, str] = field(default_factory=dict)
    created_at:  int            = field(default_factory=lambda: int(time.time()))

    @property
    def passed(self) -> bool:
        """harness の PatchVerdict.passed と同じロジック。"""
        return (
            self.t0_syntax_valid is True
            and self.t1_vuln_gone is True
            and self.t2_tests_pass is not False
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id":      self.finding_id,
            "original_code":   self.original_code,
            "patched_code":    self.patched_code,
            "explanation":     self.explanation,
            "confidence":      self.confidence,
            "status":          self.status.value,
            "t0_syntax_valid": self.t0_syntax_valid,
            "t1_vuln_gone":    self.t1_vuln_gone,
            "t2_tests_pass":   self.t2_tests_pass,
            "t3_style_score":  self.t3_style_score,
            "passed":          self.passed,
            "evidence":        self.evidence,
            "created_at":      self.created_at,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ScanReport  (harness: ReportVerdict)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ScanReport:
    """悪用可能性分析レポート。harness の ReportVerdict に対応。

    harness の section_scores (primitive / reachability / heap_layout /
    escalation_path / constraints) を Python カテゴリ別スコアに置換。
    """
    section_scores:  Dict[str, int]  # injection/auth/crypto/memory/logic → 0/1/2
    rubric_score:    int             # sum, 0..10
    total_score:     float           # rubric / 10
    severity_rating: str            # CRITICAL/HIGH/MEDIUM/LOW/NOT-A-BUG
    reachability:    str            # REACHABLE/LIMITED/UNCLEAR
    novelty_status:  str            # NEW/KNOWN/FIXED/UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_scores":  self.section_scores,
            "rubric_score":    self.rubric_score,
            "total_score":     self.total_score,
            "severity_rating": self.severity_rating,
            "reachability":    self.reachability,
            "novelty_status":  self.novelty_status,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ScanSession  (harness: RunResult)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ScanSession:
    """1 回のスキャンセッション全体。harness の RunResult に対応。

    status: "running" / "completed" / "error"
    harness の find_transcript / grade_transcript はメモリ圧縮のため省略。
    """
    id:           str
    target:       ScanTarget
    status:       str                          = "running"
    findings:     List[VulnFinding]            = field(default_factory=list)
    patches:      List[PatchCandidate]         = field(default_factory=list)
    started_at:   int                          = field(default_factory=lambda: int(time.time()))
    completed_at: Optional[int]                = None
    error:        Optional[str]                = None

    @property
    def finding_count(self) -> int:
        return len(self.findings)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == VulnSeverity.CRITICAL)

    @property
    def elapsed_s(self) -> Optional[float]:
        if self.completed_at:
            return float(self.completed_at - self.started_at)
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":             self.id,
            "target":         self.target.to_dict(),
            "status":         self.status,
            "finding_count":  self.finding_count,
            "critical_count": self.critical_count,
            "findings":       [f.to_dict() for f in self.findings],
            "patches":        [p.to_dict() for p in self.patches],
            "started_at":     self.started_at,
            "completed_at":   self.completed_at,
            "elapsed_s":      self.elapsed_s,
            "error":          self.error,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VulnStore
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VulnStore:
    """VulnFinding / ScanSession の CRUD + フィルタ。"""

    def __init__(self) -> None:
        self._findings: Dict[str, VulnFinding] = {}
        self._sessions: Dict[str, ScanSession] = {}

    # ── Finding CRUD ────────────────────────────────────────────

    def add_finding(self, finding: VulnFinding) -> VulnFinding:
        self._findings[finding.id] = finding
        return finding

    def get_finding(self, finding_id: str) -> Optional[VulnFinding]:
        return self._findings.get(finding_id)

    def list_findings(self) -> List[VulnFinding]:
        return list(self._findings.values())

    def by_severity(self, severity: VulnSeverity) -> List[VulnFinding]:
        return [f for f in self._findings.values() if f.severity == severity]

    def by_category(self, category: VulnCategory) -> List[VulnFinding]:
        return [f for f in self._findings.values() if f.category == category]

    def by_file(self, file_path: str) -> List[VulnFinding]:
        return [f for f in self._findings.values() if f.file_path == file_path]

    def delete_finding(self, finding_id: str) -> bool:
        if finding_id in self._findings:
            del self._findings[finding_id]
            return True
        return False

    # ── Session CRUD ─────────────────────────────────────────────

    def add_session(self, session: ScanSession) -> ScanSession:
        self._sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ScanSession]:
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[ScanSession]:
        return list(self._sessions.values())

    def __len__(self) -> int:
        return len(self._findings)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 検出パターン  (harness: ASAN + find-agent が動的検出するものを静的に代替)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_INJECTION_PATTERNS: List[Tuple[str, str, str]] = [
    (r"os\.system\s*\(",               "os.system() — command injection risk",       "CWE-78"),
    (r"subprocess\.[A-Za-z]+\s*\(.*shell\s*=\s*True", "subprocess shell=True",       "CWE-78"),
    (r"\beval\s*\(",                   "eval() — code injection risk",               "CWE-95"),
    (r"\bexec\s*\(",                   "exec() — code injection risk",               "CWE-95"),
    (r"pickle\.loads?\s*\(",           "pickle.load — deserialization risk",         "CWE-502"),
    (r"marshal\.loads?\s*\(",          "marshal.load — deserialization risk",        "CWE-502"),
    (r"yaml\.load\s*\([^,)]*\)",       "yaml.load without safe Loader",              "CWE-20"),
]

_AUTH_PATTERNS: List[Tuple[str, str, str]] = [
    (r"password\s*==\s*['\"]",         "hardcoded password comparison",              "CWE-259"),
    (r"\bsecret\s*=\s*['\"][^'\"]{1,}['\"]",  "hardcoded secret",                   "CWE-798"),
    (r"\bapi_key\s*=\s*['\"][^'\"]+['\"]",    "hardcoded API key",                   "CWE-798"),
    (r"\btoken\s*=\s*['\"][^'\"]{8,}['\"]",   "hardcoded token",                     "CWE-798"),
]

_CRYPTO_PATTERNS: List[Tuple[str, str, str]] = [
    (r"\bmd5\s*\(",                    "MD5 — weak hash",                            "CWE-327"),
    (r"\bsha1\s*\(",                   "SHA-1 — weak hash",                          "CWE-327"),
    (r"\bDES\b",                       "DES cipher — weak encryption",               "CWE-327"),
    (r"\brandom\.random\s*\(",         "random.random() for security purpose",       "CWE-338"),
    (r"\brandom\.randint\s*\(",        "random.randint() for security purpose",      "CWE-338"),
]

_DOS_PATTERNS: List[Tuple[str, str, str]] = [
    (r"while\s+True\s*:",              "unbounded while loop — DoS risk",            "CWE-835"),
    (r"re\.compile\s*\(",              "regex compile — potential ReDoS",            "CWE-400"),
]

# カテゴリ → (patterns, severity) マッピング
_CATEGORY_MAP: List[Tuple[VulnCategory, List[Tuple[str, str, str]], VulnSeverity]] = [
    (VulnCategory.INJECTION, _INJECTION_PATTERNS, VulnSeverity.HIGH),
    (VulnCategory.AUTH,      _AUTH_PATTERNS,      VulnSeverity.CRITICAL),
    (VulnCategory.CRYPTO,    _CRYPTO_PATTERNS,    VulnSeverity.MEDIUM),
    (VulnCategory.DOS,       _DOS_PATTERNS,       VulnSeverity.LOW),
]


def _scan_source(
    source:    str,
    file_path: str,
    patterns:  List[Tuple[str, str, str]],
    category:  VulnCategory,
    severity:  VulnSeverity,
) -> List[VulnFinding]:
    """ソーステキストを行ごとに正規表現でスキャンして VulnFinding を返す。"""
    findings: List[VulnFinding] = []
    lines = source.splitlines()
    for pattern, title, cwe_id in patterns:
        for i, line in enumerate(lines, start=1):
            if re.search(pattern, line):
                findings.append(VulnFinding(
                    id=str(uuid.uuid4()),
                    title=title,
                    category=category,
                    severity=severity,
                    description=f"Pattern `{pattern}` matched at line {i}",
                    file_path=file_path,
                    line_start=i,
                    line_end=i,
                    evidence=line.strip(),
                    reproduction_steps=f"Review line {i} of {file_path}",
                    cwe_id=cwe_id,
                ))
    return findings


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VulnScanner  (harness: find + recon エージェント)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VulnScanner:
    """静的解析スキャナー。harness の find + recon エージェントに対応。

    harness の run_recon() は focus_areas を自動生成する。
    VulnScanner では ScanTarget.focus_areas で明示指定。
    """

    def __init__(self, store: Optional[VulnStore] = None) -> None:
        self._store = store if store is not None else VulnStore()

    @property
    def store(self) -> VulnStore:
        return self._store

    def scan_source(
        self,
        source:      str,
        file_path:   str           = "<string>",
        focus_areas: Optional[List[str]] = None,
    ) -> List[VulnFinding]:
        """ソースコード文字列をスキャンして VulnFinding リストを返す。"""
        findings: List[VulnFinding] = []
        for category, patterns, severity in _CATEGORY_MAP:
            findings += _scan_source(source, file_path, patterns, category, severity)

        # focus_areas フィルタ (harness: focus_area partition に対応)
        if focus_areas:
            filtered = [
                f for f in findings
                if any(a.lower() in f.title.lower() for a in focus_areas)
            ]
            return filtered
        return findings

    def scan(
        self,
        target: ScanTarget,
        source: Optional[str] = None,
    ) -> ScanSession:
        """ScanTarget に対してスキャンを実行し ScanSession を返す。

        harness の run_find() に相当。source が None の場合は空のソースとして扱う。
        """
        session = ScanSession(
            id=str(uuid.uuid4()),
            target=target,
            status="running",
        )
        try:
            src = source or ""
            findings = self.scan_source(
                source=src,
                file_path=target.path,
                focus_areas=target.focus_areas or None,
            )
            # known_findings 除外 (harness: known_bugs dedup に対応)
            if target.known_findings:
                findings = [
                    f for f in findings
                    if not any(kf.lower() in f.title.lower() for kf in target.known_findings)
                ]
            session.findings = findings
            session.status = "completed"
            session.completed_at = int(time.time())
        except Exception as exc:  # noqa: BLE001
            session.status = "error"
            session.error = str(exc)
            session.completed_at = int(time.time())

        self._store.add_session(session)
        for finding in session.findings:
            self._store.add_finding(finding)
        return session


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 安全な置換候補  (harness: patch エージェントが生成するパッチに対応)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SAFE_REPLACEMENTS: List[Tuple[str, str, str]] = [
    (r"os\.system\s*\(",               "subprocess.run([...], check=True)  # use list, not shell string",
                                       "os.system() replaced by subprocess.run() list form"),
    (r"\beval\s*\(",                   "ast.literal_eval(",
                                       "eval() replaced by ast.literal_eval()"),
    (r"\bexec\s*\(",                   "# exec() removed — use explicit function call",
                                       "exec() removed"),
    (r"pickle\.loads?\s*\(",           "json.loads(",
                                       "pickle replaced by json"),
    (r"yaml\.load\s*\([^,)]*\)",       "yaml.safe_load(",
                                       "yaml.load replaced by yaml.safe_load()"),
    (r"\bmd5\s*\(",                    "hashlib.sha256(",
                                       "MD5 replaced by SHA-256"),
    (r"\bsha1\s*\(",                   "hashlib.sha256(",
                                       "SHA-1 replaced by SHA-256"),
    (r"\brandom\.random\s*\(",         "secrets.token_hex(16)",
                                       "random.random() replaced by secrets.token_hex()"),
    (r"\brandom\.randint\s*\(",        "secrets.randbelow(",
                                       "random.randint() replaced by secrets.randbelow()"),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VulnPatcher  (harness: patch + patch_grade)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VulnPatcher:
    """パッチ候補生成・T0/T1/T2 検証ラダー。harness の patch + patch_grade に対応。"""

    def suggest_patch(self, finding: VulnFinding) -> Optional[PatchCandidate]:
        """VulnFinding に対してパッチ候補を生成する。harness の patch エージェントに対応。"""
        if not finding.evidence:
            return None

        original = finding.evidence
        patched = original
        explanation = ""

        for pattern, replacement, expl in _SAFE_REPLACEMENTS:
            if re.search(pattern, original):
                patched = re.sub(pattern, replacement, original)
                explanation = expl
                break

        if patched == original:
            return None  # 自動パッチ不可

        return PatchCandidate(
            finding_id=finding.id,
            original_code=original,
            patched_code=patched,
            explanation=explanation,
            confidence=0.7,
        )

    def validate_patch(self, candidate: PatchCandidate) -> PatchCandidate:
        """T0/T1/T2 ラダー検証。harness の patch_grade (T0-T2) に対応。

        T0: ast.parse() で構文チェック
        T1: 脆弱パターンが patched_code に残っていないか確認
        T2: test_command なし → None (skip)
        """
        evidence: Dict[str, str] = {}

        # T0: 構文チェック (harness: t0_builds = コンパイル成功)
        try:
            ast.parse(candidate.patched_code)
            candidate.t0_syntax_valid = True
            evidence["t0"] = "ast.parse() passed"
        except SyntaxError as e:
            candidate.t0_syntax_valid = False
            evidence["t0"] = f"SyntaxError: {e}"

        # T1: 脆弱パターン消滅チェック (harness: t1_poc_stops)
        still_vulnerable = False
        for category, patterns, _ in _CATEGORY_MAP:
            for pattern, _, _ in patterns:
                if re.search(pattern, candidate.patched_code):
                    still_vulnerable = True
                    evidence["t1"] = f"Vulnerable pattern still present: {pattern}"
                    break
            if still_vulnerable:
                break

        candidate.t1_vuln_gone = not still_vulnerable
        if candidate.t1_vuln_gone:
            evidence["t1"] = "No vulnerable patterns detected in patched code"

        # T2: テスト (harness: t2_tests_pass) → 外部テストなので None
        candidate.t2_tests_pass = None
        evidence["t2"] = "T2 skipped (test_command not configured)"

        candidate.evidence = evidence

        # ステータス更新
        if candidate.passed:
            candidate.status = PatchStatus.VERIFIED
        elif candidate.t0_syntax_valid is False:
            candidate.status = PatchStatus.REJECTED
        else:
            candidate.status = PatchStatus.APPLIED

        return candidate


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ScanReportEngine  (harness: report + grade エージェント)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ScanReportEngine:
    """Markdown / JSON レポート生成。harness の run_report + grader に対応。"""

    def build_report(self, session: ScanSession) -> ScanReport:
        """ScanSession から ScanReport を生成する。harness の run_report に対応。"""
        findings = session.findings
        if not findings:
            return ScanReport(
                section_scores={
                    "injection": 0, "auth": 0,
                    "crypto": 0, "memory": 0, "logic": 0,
                },
                rubric_score=0,
                total_score=0.0,
                severity_rating="NOT-A-BUG",
                reachability="UNCLEAR",
                novelty_status="UNKNOWN",
            )

        # セクションスコア (harness: section_scores に対応)
        section_map: Dict[str, VulnCategory] = {
            "injection": VulnCategory.INJECTION,
            "auth":      VulnCategory.AUTH,
            "crypto":    VulnCategory.CRYPTO,
            "memory":    VulnCategory.MEMORY,
            "logic":     VulnCategory.LOGIC,
        }
        section_scores: Dict[str, int] = {}
        for section_name, category in section_map.items():
            count = sum(1 for f in findings if f.category == category)
            section_scores[section_name] = min(count * 2, 2)  # 0/1/2 per section

        rubric_score = sum(section_scores.values())
        max_possible = len(section_scores) * 2
        total_score = rubric_score / max_possible if max_possible > 0 else 0.0

        # 最高重要度を severity_rating に (harness: severity_rating)
        max_finding = max(findings, key=lambda f: f.severity.score)
        severity_rating = max_finding.severity.value.upper()

        # 到達可能性 (harness: reachability_verdict)
        has_injection = any(f.category == VulnCategory.INJECTION for f in findings)
        reachability = "REACHABLE" if has_injection else "LIMITED"

        return ScanReport(
            section_scores=section_scores,
            rubric_score=rubric_score,
            total_score=total_score,
            severity_rating=severity_rating,
            reachability=reachability,
            novelty_status="NEW",
        )

    def to_markdown(self, session: ScanSession) -> str:
        """Markdown レポート生成。harness の report テキストに対応。"""
        report = self.build_report(session)
        lines = [
            f"# Vulnerability Scan Report — {session.target.name}",
            "",
            f"**Session ID**: `{session.id}`  ",
            f"**Status**: {session.status}  ",
            f"**Findings**: {session.finding_count} (Critical: {session.critical_count})  ",
            f"**Severity**: {report.severity_rating}  ",
            f"**Reachability**: {report.reachability}  ",
            f"**Score**: {report.rubric_score} / {len(report.section_scores) * 2}  ",
            "",
            "## Section Scores",
            "",
            "| Section | Score |",
            "|---------|-------|",
        ]
        for section, score in report.section_scores.items():
            lines.append(f"| {section} | {score}/2 |")

        lines += ["", "## Findings", ""]
        if not session.findings:
            lines.append("_No findings._")
        else:
            for finding in session.findings:
                lines += [
                    f"### [{finding.severity.value.upper()}] {finding.title}",
                    f"- **Category**: {finding.category.value}",
                    f"- **File**: `{finding.file_path}` line {finding.line_start}",
                    f"- **Evidence**: `{finding.evidence}`",
                    f"- **CWE**: {finding.cwe_id or 'N/A'}",
                    "",
                ]

        if session.patches:
            lines += ["## Patches", ""]
            for patch in session.patches:
                status_str = "PASS" if patch.passed else "FAIL"
                lines += [
                    f"### [{status_str}] Finding `{patch.finding_id}`",
                    f"- **Status**: {patch.status.value}",
                    f"- **T0 (syntax)**: {patch.t0_syntax_valid}",
                    f"- **T1 (vuln gone)**: {patch.t1_vuln_gone}",
                    f"- **T2 (tests)**: {patch.t2_tests_pass}",
                    "",
                ]

        return "\n".join(lines)

    def to_json(self, session: ScanSession) -> Dict[str, Any]:
        """JSON レポート生成。harness の RunResult.to_dict() + ReportVerdict に対応。"""
        report = self.build_report(session)
        return {
            "session": session.to_dict(),
            "report":  report.to_dict(),
        }
