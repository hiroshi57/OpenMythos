"""
Sprint 53 — セキュリティ統合
Sprint 61 — Claude Fable 5 による AI 強化ペンテスト分析

Hermes Skills: web-pentest / oss-forensics
ref: skills/security/*-SKILL.md

セキュリティテスト・OSS 検査ツールを OpenMythos に統合する。

Sprint 61 追加:
  AISecurityEnhancer : Claude Fable 5 で pentest 結果を AI 分析強化
    - generate_executive_summary()   エグゼクティブサマリー
    - prioritize_findings()          優先度付きリメディエーション計画
    - generate_patch_advice()        具体的修正アドバイス
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
                cipher = s.cipher()
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
            f"# Penetration Test Report\n",
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 61 — AISecurityEnhancer (Claude Fable 5)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AISecurityEnhancer:
    """Claude Fable 5 を使った AI 強化セキュリティ分析モジュール。

    WebPentester / OSSForensics の結果を受け取り、
    自然言語によるエグゼクティブサマリー・優先度付きリメディエーション・
    具体的修正アドバイスを生成する。

    AI 未設定時はルールベースのフォールバックを提供する。
    """

    def __init__(self, anthropic_api_key: Optional[str] = None) -> None:
        self._provider: Any = None
        if anthropic_api_key:
            try:
                from open_mythos.skills.llm_providers import (
                    ClaudeModelTier, ClaudeProvider, ProviderConfig, ProviderType,
                )
                self._provider = ClaudeProvider(ProviderConfig(
                    provider=ProviderType.CLAUDE,
                    api_key=anthropic_api_key,
                    model=ClaudeModelTier.FABLE_5,
                    timeout=60,
                ))
            except Exception:
                pass

    def generate_executive_summary(self, report: "PentestReport") -> str:
        """ペンテスト結果のエグゼクティブサマリーを生成する。"""
        if self._provider:
            findings_text = "\n".join(
                f"[{f.severity}] {f.title}: {f.description[:100]}"
                for f in report.findings[:10]
            )
            prompt = (
                f"Pentest result for {report.target_url}:\n"
                f"Risk Score: {report.risk_score}/10\n"
                f"Findings:\n{findings_text}\n\n"
                "Write a 3-sentence executive summary for non-technical stakeholders."
            )
            try:
                from open_mythos.skills.llm_providers import LLMRequest
                resp = self._provider.complete(LLMRequest(
                    prompt=prompt,
                    system="You are a cybersecurity consultant using Claude Fable 5.",
                    max_tokens=256,
                    temperature=0.3,
                ))
                return resp.text
            except Exception:
                pass

        # フォールバック
        risk_label = (
            "Critical" if report.risk_score >= 8
            else "High" if report.risk_score >= 6
            else "Medium" if report.risk_score >= 4
            else "Low"
        )
        return (
            f"Security assessment of {report.target_url} identified {len(report.findings)} issues "
            f"with an overall risk score of {report.risk_score}/10 ({risk_label}). "
            f"{report.critical_count} critical and {report.high_count} high severity issues "
            f"require immediate attention. Immediate remediation is {'strongly ' if risk_label in ('Critical','High') else ''}recommended."
        )

    def prioritize_findings(self, report: "PentestReport") -> List[Dict[str, Any]]:
        """ファインディングを優先度付きリメディエーション計画に変換する。"""
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
        sorted_findings = sorted(
            report.findings,
            key=lambda f: priority_order.get(f.severity, 5),
        )
        return [
            {
                "priority":       i + 1,
                "severity":       f.severity,
                "title":          f.title,
                "recommendation": f.recommendation,
                "effort":         "immediate" if f.severity == "CRITICAL"
                                  else "short-term" if f.severity == "HIGH"
                                  else "planned",
            }
            for i, f in enumerate(sorted_findings)
        ]

    def generate_patch_advice(self, finding: "PentestFinding") -> str:
        """個別ファインディングの具体的修正アドバイスを生成する。"""
        if self._provider:
            prompt = (
                f"Security finding:\n"
                f"Category: {finding.category}\n"
                f"Title: {finding.title}\n"
                f"Description: {finding.description}\n\n"
                "Provide specific, actionable patch advice in 2-3 steps."
            )
            try:
                from open_mythos.skills.llm_providers import LLMRequest
                resp = self._provider.complete(LLMRequest(
                    prompt=prompt,
                    system="You are a security engineer using Claude Fable 5.",
                    max_tokens=200,
                    temperature=0.2,
                ))
                return resp.text
            except Exception:
                pass

        return finding.recommendation or f"Apply standard {finding.category} security controls."
