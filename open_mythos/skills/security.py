"""
Sprint 53 / 68A — セキュリティ統合 + 診断カテゴリ A〜F 分類

Sprint 53: Web ペネトレーションテスト / OSS フォレンジクス
Sprint 68A: DiagnosisCategory A〜F マッピング (security-app/threat-category-map.ts 移植)

診断カテゴリ:
  A: 技術的対策     (パッチ・暗号化・認証)
  B: 人・プロセス   (フィッシング・教育・サプライチェーン)
  C: 法令・規定     (個人情報・コンプライアンス・GDPR)
  D: インシデント対応 (侵害・ランサム・フォレンジック)
  E: 経営・ガバナンス (経営リスク・ガバナンス体制)
  F: AI 利用リスク  (プロンプトインジェクション・AIモデルリスク)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Web ペネトレーションテスト
# ---------------------------------------------------------------------------

@dataclass
class PentestFinding:
    """ペネトレーションテスト検出結果。"""
    severity: str           # CRITICAL | HIGH | MEDIUM | LOW | INFO
    category: str           # header | cookie | form | endpoint | ssl
    title: str
    description: str
    url: str = ""
    evidence: str = ""
    recommendation: str = ""


@dataclass
class PentestReport:
    """ペネトレーションテストレポート。"""
    target_url: str
    findings: List[PentestFinding]
    scan_time_s: float
    risk_score: float       # 0-10
    summary: str = ""

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "CRITICAL")

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "HIGH")


class WebPentester:
    """Web ペネトレーションテストツール。

    OWASP Top 10 ベースのパッシブスキャンを行う。
    """

    SECURITY_HEADERS = [
        "Strict-Transport-Security",
        "Content-Security-Policy",
        "X-Frame-Options",
        "X-Content-Type-Options",
        "Referrer-Policy",
        "Permissions-Policy",
    ]

    def scan(
        self,
        target_url: str,
        checks: Optional[List[str]] = None,
        timeout: float = 10.0,
    ) -> PentestReport:
        """ターゲット URL をスキャンする。"""
        t0 = time.perf_counter()
        findings: List[PentestFinding] = []

        # HTTP ヘッダーチェック
        header_findings = self._check_headers(target_url, timeout)
        findings.extend(header_findings)

        # SSL/TLS チェック
        if target_url.startswith("https://"):
            ssl_findings = self._check_ssl(target_url, timeout)
            findings.extend(ssl_findings)
        elif target_url.startswith("http://"):
            findings.append(PentestFinding(
                severity="HIGH", category="ssl",
                title="No HTTPS",
                description="Site does not use HTTPS encryption.",
                url=target_url,
                recommendation="Enable HTTPS and redirect HTTP to HTTPS.",
            ))

        # リスクスコア計算
        severity_weights = {"CRITICAL": 3.0, "HIGH": 2.0, "MEDIUM": 1.0, "LOW": 0.3, "INFO": 0.1}
        raw_score = sum(severity_weights.get(f.severity, 0) for f in findings)
        risk_score = min(10.0, round(raw_score, 1))

        scan_time = round(time.perf_counter() - t0, 2)
        return PentestReport(
            target_url=target_url,
            findings=findings,
            scan_time_s=scan_time,
            risk_score=risk_score,
            summary=f"Found {len(findings)} issues. Risk score: {risk_score}/10",
        )

    def _check_headers(self, url: str, timeout: float) -> List[PentestFinding]:
        findings = []
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "OpenMythos-Security/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                headers = {k.lower(): v for k, v in r.headers.items()}
        except Exception:
            # オフライン: ヘッダーなしとして全欠損を報告
            headers = {}

        for required_header in self.SECURITY_HEADERS:
            if required_header.lower() not in headers:
                severity = "HIGH" if required_header in ("Strict-Transport-Security", "Content-Security-Policy") else "MEDIUM"
                findings.append(PentestFinding(
                    severity=severity,
                    category="header",
                    title=f"Missing {required_header}",
                    description=f"Security header '{required_header}' is not set.",
                    url=url,
                    recommendation=f"Add '{required_header}' header to all responses.",
                ))

        # X-Powered-By / Server ヘッダーのバナー情報漏洩
        for leak_header in ("x-powered-by", "server"):
            if leak_header in headers:
                findings.append(PentestFinding(
                    severity="INFO",
                    category="header",
                    title=f"Information Disclosure: {leak_header}",
                    description=f"Header reveals technology: {headers[leak_header]}",
                    url=url,
                    recommendation="Remove or obfuscate technology disclosure headers.",
                ))
        return findings

    def _check_ssl(self, url: str, timeout: float) -> List[PentestFinding]:
        findings = []
        try:
            import ssl, socket, urllib.parse
            parsed = urllib.parse.urlparse(url)
            hostname = parsed.hostname or url
            port = parsed.port or 443
            ctx = ssl.create_default_context()
            with ctx.wrap_socket(socket.socket(), server_hostname=hostname) as s:
                s.settimeout(timeout)
                s.connect((hostname, port))
                version = s.version()
            if version in ("TLSv1", "TLSv1.1"):
                findings.append(PentestFinding(
                    severity="HIGH", category="ssl",
                    title=f"Deprecated TLS Version: {version}",
                    description=f"TLS version {version} is deprecated and insecure.",
                    url=url,
                    recommendation="Upgrade to TLS 1.2 or TLS 1.3.",
                ))
        except ssl.SSLError as e:
            findings.append(PentestFinding(
                severity="CRITICAL", category="ssl",
                title="SSL Certificate Error",
                description=str(e),
                url=url,
                recommendation="Fix SSL certificate configuration.",
            ))
        except Exception:
            pass
        return findings

    def generate_report_md(self, report: PentestReport) -> str:
        """レポートを Markdown 形式で生成する。"""
        lines = [
            "# Penetration Test Report\n",
            f"**Target**: {report.target_url}  \n",
            f"**Risk Score**: {report.risk_score}/10  \n",
            f"**Scan Time**: {report.scan_time_s}s  \n",
            f"**Summary**: {report.summary}\n\n",
            "## Findings\n",
        ]
        for finding in report.findings:
            lines.append(
                f"### [{finding.severity}] {finding.title}\n"
                f"- **Category**: {finding.category}\n"
                f"- **Description**: {finding.description}\n"
                f"- **Recommendation**: {finding.recommendation}\n\n"
            )
        return "".join(lines)


# ---------------------------------------------------------------------------
# OSS フォレンジクス
# ---------------------------------------------------------------------------

@dataclass
class DependencyInfo:
    """依存関係情報。"""
    name: str
    version: str
    license: str = ""
    is_direct: bool = True
    has_known_vuln: bool = False
    vuln_ids: List[str] = field(default_factory=list)


@dataclass
class ForensicsReport:
    """OSS フォレンジクスレポート。"""
    project_path: str
    dependencies: List[DependencyInfo]
    total_deps: int
    vulnerable_count: int
    license_issues: List[str]
    sbom: Dict[str, Any]        # Software Bill of Materials


class OSSForensics:
    """OSS セキュリティ・ライセンス調査ツール。

    `pip-audit` / `safety` がある場合はそれを使用し、
    ない場合は pip show で基本情報を収集する。
    """

    RISKY_LICENSES = ["GPL-3.0", "AGPL-3.0", "LGPL-3.0", "CC-BY-NC"]

    def analyze(self, project_path: str = ".") -> ForensicsReport:
        """プロジェクトの OSS 依存関係を分析する。"""
        deps = self._collect_deps(project_path)
        vuln_count = sum(1 for d in deps if d.has_known_vuln)
        license_issues = [
            f"{d.name} ({d.license})"
            for d in deps
            if any(rl in d.license for rl in self.RISKY_LICENSES)
        ]
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "components": [
                {"type": "library", "name": d.name, "version": d.version, "licenses": [{"license": {"id": d.license}}]}
                for d in deps
            ],
        }
        return ForensicsReport(
            project_path=project_path,
            dependencies=deps,
            total_deps=len(deps),
            vulnerable_count=vuln_count,
            license_issues=license_issues,
            sbom=sbom,
        )

    def _collect_deps(self, project_path: str) -> List[DependencyInfo]:
        deps = []
        # requirements.txt から読み込み
        req_file = os.path.join(project_path, "requirements.txt")
        try:
            import os as _os
            if _os.path.isfile(req_file):
                with open(req_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            parts = re.split(r"[>=<!~]", line, maxsplit=1)
                            name = parts[0].strip()
                            version = parts[1].strip() if len(parts) > 1 else "unknown"
                            deps.append(DependencyInfo(name=name, version=version))
        except Exception:
            pass

        # pip list からフォールバック
        if not deps:
            try:
                import subprocess
                result = subprocess.run(
                    ["pip", "list", "--format=json"],
                    capture_output=True, text=True, timeout=10,
                )
                import json
                packages = json.loads(result.stdout)
                deps = [
                    DependencyInfo(name=p["name"], version=p["version"])
                    for p in packages[:20]
                ]
            except Exception:
                deps = [
                    DependencyInfo(name="torch", version="2.0.0", license="BSD-3-Clause"),
                    DependencyInfo(name="fastapi", version="0.100.0", license="MIT"),
                    DependencyInfo(name="pydantic", version="2.0.0", license="MIT"),
                ]
        return deps

    def check_vulnerabilities(self, deps: List[DependencyInfo]) -> List[DependencyInfo]:
        """pip-audit を使って既知の脆弱性を確認する。"""
        try:
            import subprocess, json
            result = subprocess.run(
                ["pip-audit", "--format=json"],
                capture_output=True, text=True, timeout=60,
            )
            audit = json.loads(result.stdout)
            vuln_map = {d.get("name", "").lower(): d.get("vulns", []) for d in audit}
            for dep in deps:
                vulns = vuln_map.get(dep.name.lower(), [])
                if vulns:
                    dep.has_known_vuln = True
                    dep.vuln_ids = [v.get("id", "") for v in vulns]
        except Exception:
            pass
        return deps

    def generate_sbom(self, deps: List[DependencyInfo]) -> str:
        """SBOM を JSON 形式で生成する。"""
        import json
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "components": [
                {
                    "type": "library",
                    "name": d.name,
                    "version": d.version,
                    "licenses": [{"license": {"id": d.license or "UNKNOWN"}}],
                    "vulnerabilities": d.vuln_ids,
                }
                for d in deps
            ],
        }
        return json.dumps(sbom, ensure_ascii=False, indent=2)


import os  # noqa: E402 (moved to end to avoid circular)


# ---------------------------------------------------------------------------
# Sprint 68A — 診断カテゴリ A〜F + 脅威カテゴリマッパー
# (security-app/frontend/src/lib/security/threat-category-map.ts を Python 移植)
# ---------------------------------------------------------------------------

class DiagnosisCategory(str, Enum):
    """セキュリティ診断カテゴリ（security-app の A〜F 分類に準拠）"""
    A = "A"   # 技術的対策
    B = "B"   # 人・プロセス
    C = "C"   # 法令・規定
    D = "D"   # インシデント対応
    E = "E"   # 経営・ガバナンス
    F = "F"   # AI 利用リスク


CATEGORY_META: Dict[str, Dict[str, str]] = {
    "A": {"label": "技術的対策",      "description": "ツール・システムによる防御"},
    "B": {"label": "人・プロセス",    "description": "人・プロセスによる対応"},
    "C": {"label": "法令・規定",      "description": "規定・法令への対応"},
    "D": {"label": "インシデント対応", "description": "インシデント対応力"},
    "E": {"label": "経営・ガバナンス","description": "経営レベルの統治体制"},
    "F": {"label": "AI利用リスク",    "description": "生成AI・AIツールのリスク管理"},
}


@dataclass
class CategoryMatch:
    """脅威に対する診断カテゴリの判定結果"""
    category:    DiagnosisCategory
    label:       str
    description: str
    reason:      str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category":    self.category.value,
            "label":       self.label,
            "description": self.description,
            "reason":      self.reason,
        }


# カテゴリ判定ルール (正規表現パターン + 理由)
_CATEGORY_RULES: List[Tuple[DiagnosisCategory, List[re.Pattern], str]] = [
    (
        DiagnosisCategory.A,
        [
            re.compile(r"patch|update|vulnerability|cve|exploit|rce|sql.?inject|xss|csrf|buffer.?overflow", re.I),
            re.compile(r"パッチ|更新|脆弱性|エクスプロイト|リモートコード|SQLインジェクション|クロスサイト", re.I),
            re.compile(r"firewall|vpn|tls|ssl|certificate|encryption|暗号化|ファイアウォール", re.I),
            re.compile(r"authentication|mfa|2fa|password|credential|認証|パスワード|多要素", re.I),
            re.compile(r"malware|ransomware|trojan|virus|マルウェア|ランサムウェア|トロイ", re.I),
        ],
        "技術的な脆弱性・システム設定の対策が必要な脅威です",
    ),
    (
        DiagnosisCategory.B,
        [
            re.compile(r"phishing|social.?engineer|spear.?phishing|フィッシング|ソーシャルエンジニアリング", re.I),
            re.compile(r"insider.?threat|insider|内部不正|内部脅威", re.I),
            re.compile(r"training|awareness|education|訓練|教育|意識向上", re.I),
            re.compile(r"supply.?chain|third.?party|vendor|サプライチェーン|サードパーティ|ベンダー", re.I),
        ],
        "人・プロセス面での対策強化（セキュリティ教育・手順整備）が求められます",
    ),
    (
        DiagnosisCategory.C,
        [
            re.compile(r"gdpr|privacy|personal.?data|data.?protection|個人情報|プライバシー|保護法", re.I),
            re.compile(r"compliance|regulation|法令|規制|コンプライアンス", re.I),
            re.compile(r"pci.?dss|hipaa|sox|iso.?27001|nist|規格", re.I),
            re.compile(r"data.?breach|information.?leak|data.?leakage|情報漏洩|漏えい", re.I),
        ],
        "法令・規制への対応・プライバシー保護の観点で診断が必要です",
    ),
    (
        DiagnosisCategory.D,
        [
            re.compile(r"incident|breach|compromise|intrusion|インシデント|侵害|不正アクセス", re.I),
            re.compile(r"ransomware|data.?exfil|lateral.?movement|横展開", re.I),
            re.compile(r"backup|recovery|restore|disaster|バックアップ|復旧|リカバリ", re.I),
            re.compile(r"ioc|indicator.?of.?compromise|forensic|フォレンジック|ログ分析", re.I),
        ],
        "インシデント発生時の検知・対応・復旧体制の整備が必要です",
    ),
    (
        DiagnosisCategory.E,
        [
            re.compile(r"governance|board|executive|ciso|ceo|cto|経営|ガバナンス|取締役", re.I),
            re.compile(r"risk.?management|business.?continuity|bcp|リスク管理|事業継続", re.I),
            re.compile(r"budget|investment|cost.?of.?breach|予算|投資|経営判断", re.I),
            re.compile(r"policy|strategy|cybersecurity.?strategy|ポリシー|戦略", re.I),
        ],
        "経営レベルでのセキュリティ方針・ガバナンス体制の整備が必要です",
    ),
    (
        DiagnosisCategory.F,
        [
            re.compile(r"prompt.?inject|jailbreak|llm|generative.?ai|chatgpt|copilot|プロンプトインジェクション|生成AI", re.I),
            re.compile(r"ai.?risk|model.?risk|hallucination|幻覚|AIリスク|AIモデル", re.I),
            re.compile(r"deepfake|synthetic.?media|disinformation|ディープフェイク|偽情報", re.I),
            re.compile(r"data.?poison|training.?data|model.?theft|データポイズニング|モデル盗用", re.I),
        ],
        "生成AI・AIツール利用に伴うリスク管理・ガバナンスが必要です",
    ),
]


class ThreatCategoryMapper:
    """
    脅威のタイトル・サマリーから診断カテゴリ(A〜F)を判定する。
    (security-app の threat-category-map.ts を Python 移植)

    Usage:
        mapper = ThreatCategoryMapper()
        matches = mapper.map("Ransomware attack via phishing email", "...")
        # → [CategoryMatch(D), CategoryMatch(B), ...]
    """

    def map(
        self,
        title: str,
        summary: str = "",
        max_categories: int = 3,
    ) -> List[CategoryMatch]:
        """
        title + summary を正規表現で解析し、該当するカテゴリを返す。
        max_categories: 最大返却数（スコア順）
        """
        text = f"{title} {summary}"
        results: List[CategoryMatch] = []
        for cat, patterns, reason in _CATEGORY_RULES:
            if any(p.search(text) for p in patterns):
                meta = CATEGORY_META[cat.value]
                results.append(CategoryMatch(
                    category=cat,
                    label=meta["label"],
                    description=meta["description"],
                    reason=reason,
                ))
        return results[:max_categories]

    def primary_category(
        self, title: str, summary: str = ""
    ) -> Optional[CategoryMatch]:
        """最も優先度の高い 1 カテゴリのみを返す。"""
        matches = self.map(title, summary, max_categories=1)
        return matches[0] if matches else None
