"""
Sprint 79 — 経営チーム エージェント基盤 (keiei-os)

OpenClaw × Claude Code で実現する AI 経営 OS の Python-native 実装。
9 役割の Management Agent を統合し、経営判断・財務・マーケ・オペレーション・
技術・秘書・ナレッジ・プロダクト・クライアント管理を一元化する。

参考:
  - https://github.com/hiroshi57/di-kb (DI社ナレッジベース構造)
  - Qiita「AIエージェント9体で経営OSを作った話」
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


# ─── Enums ────────────────────────────────────────────────────────


class AgentRole(str, Enum):
    CEO       = "ceo"        # 経営判断・優先順位付け
    CFO       = "cfo"        # 財務分析・仕訳
    CMO       = "cmo"        # マーケティング戦略
    COO       = "coo"        # オペレーション管理
    CTO       = "cto"        # 技術判断・アーキテクチャ
    SECRETARY = "secretary"  # inbox/today/archive管理
    KNOWLEDGE = "knowledge"  # 教訓・パターン・インシデント記録
    PRODUCTS  = "products"   # プロダクト管理
    CLIENTS   = "clients"    # クライアント管理


class TaskPriority(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    PENDING     = "pending"
    IN_PROGRESS = "in_progress"
    DONE        = "done"
    BLOCKED     = "blocked"


class DecisionType(str, Enum):
    APPROVE  = "approve"
    REJECT   = "reject"
    DEFER    = "defer"
    ESCALATE = "escalate"
    DELEGATE = "delegate"


# ─── Data Classes ─────────────────────────────────────────────────


@dataclass
class AgentTask:
    """エージェントが担当するタスク。"""
    id: str
    title: str
    description: str
    owner_role: AgentRole
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    due_date: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "owner_role": self.owner_role.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "due_date": self.due_date,
            "tags": self.tags,
        }


@dataclass
class AgentDecision:
    """エージェントが下した意思決定。"""
    agent_role: AgentRole
    topic: str
    decision_type: DecisionType
    rationale: str
    action_items: List[str] = field(default_factory=list)
    confidence: float = 0.8  # 0.0 – 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "agent_role": self.agent_role.value,
            "topic": self.topic,
            "decision_type": self.decision_type.value,
            "rationale": self.rationale,
            "action_items": self.action_items,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentReport:
    """エージェントの定期報告。"""
    agent_role: AgentRole
    period: str
    highlights: List[str]
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, str] = field(default_factory=dict)
    next_actions: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"## {self.agent_role.value.upper()} レポート ({self.period})",
            "",
            "### ハイライト",
        ]
        for h in self.highlights:
            lines.append(f"- {h}")
        if self.issues:
            lines.append("\n### 課題")
            for i in self.issues:
                lines.append(f"- {i}")
        if self.metrics:
            lines.append("\n### KPI")
            for k, v in self.metrics.items():
                lines.append(f"- **{k}**: {v}")
        if self.next_actions:
            lines.append("\n### 次のアクション")
            for a in self.next_actions:
                lines.append(f"- {a}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "agent_role": self.agent_role.value,
            "period": self.period,
            "highlights": self.highlights,
            "issues": self.issues,
            "metrics": self.metrics,
            "next_actions": self.next_actions,
        }


@dataclass
class MeetingRecord:
    """経営会議の議事録。"""
    id: str
    title: str
    participants: List[AgentRole]
    agenda: List[str]
    decisions: List[AgentDecision] = field(default_factory=list)
    minutes: str = ""
    held_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "participants": [r.value for r in self.participants],
            "agenda": self.agenda,
            "decisions": [d.to_dict() for d in self.decisions],
            "minutes": self.minutes,
            "held_at": self.held_at,
        }


# ─── Agent Base Class ──────────────────────────────────────────────


class KeieiAgent:
    """経営チーム エージェントの基底クラス。"""

    def __init__(self, role: AgentRole, name: str, expertise: List[str]):
        self.role = role
        self.name = name
        self.expertise = expertise
        self._tasks: List[AgentTask] = []
        self._decisions: List[AgentDecision] = []

    # ── コア: 思考 ────────────────────────────────────────────────

    def think(self, topic: str, context: str = "") -> AgentDecision:
        """専門知識に基づいて意思決定を行う。"""
        relevant = any(kw.lower() in topic.lower() for kw in self.expertise)
        if relevant:
            dt = DecisionType.APPROVE
            rationale = f"{self.name}の専門領域として対応可能: {topic}"
            confidence = 0.85
        else:
            dt = DecisionType.DELEGATE
            rationale = f"{self.name}の専門外のため委譲が必要: {topic}"
            confidence = 0.5
        decision = AgentDecision(
            agent_role=self.role,
            topic=topic,
            decision_type=dt,
            rationale=rationale,
            action_items=[f"[{self.role.value}] {topic} の調査・対応"],
            confidence=confidence,
        )
        self._decisions.append(decision)
        return decision

    # ── タスク管理 ────────────────────────────────────────────────

    def assign_task(self, task: AgentTask) -> None:
        task.owner_role = self.role
        self._tasks.append(task)

    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[AgentTask]:
        if status is not None:
            return [t for t in self._tasks if t.status == status]
        return list(self._tasks)

    def complete_task(self, task_id: str) -> bool:
        for t in self._tasks:
            if t.id == task_id:
                t.status = TaskStatus.DONE
                return True
        return False

    # ── 報告 ──────────────────────────────────────────────────────

    def report(self, period: str = "今週") -> AgentReport:
        done = sum(1 for t in self._tasks if t.status == TaskStatus.DONE)
        pending = sum(1 for t in self._tasks if t.status == TaskStatus.PENDING)
        blocked = sum(1 for t in self._tasks if t.status == TaskStatus.BLOCKED)
        return AgentReport(
            agent_role=self.role,
            period=period,
            highlights=[f"タスク管理: 合計 {len(self._tasks)} 件 (完了: {done})"],
            issues=(
                [f"未対応: {pending} 件", f"ブロック: {blocked} 件"]
                if (pending or blocked) else []
            ),
            metrics={"完了率": f"{done}/{len(self._tasks)}" if self._tasks else "N/A"},
            next_actions=[t.title for t in self._tasks if t.status == TaskStatus.PENDING][:3],
        )

    def to_dict(self) -> dict:
        return {
            "role": self.role.value,
            "name": self.name,
            "expertise": self.expertise,
            "task_count": len(self._tasks),
            "decision_count": len(self._decisions),
        }


# ─── Specific Agents ──────────────────────────────────────────────


class CeoAgent(KeieiAgent):
    """CEO — 経営判断・優先順位付け"""

    def __init__(self):
        super().__init__(
            role=AgentRole.CEO,
            name="CEO",
            expertise=["経営", "戦略", "優先順位", "ビジョン", "投資", "M&A",
                       "資金", "組織", "意思決定", "目標", "OKR", "KGI"],
        )

    def prioritize(self, items: List[str]) -> List[str]:
        """経営優先順位をリスク・緊急度で並べ替える。"""
        _urgent = {"緊急", "critical", "リスク", "損失", "障害", "コンプライアンス"}
        return sorted(
            items,
            key=lambda x: 0 if any(w in x.lower() for w in _urgent) else 1,
        )

    def set_vision(self, vision: str) -> AgentDecision:
        decision = AgentDecision(
            agent_role=self.role,
            topic=f"ビジョン設定: {vision}",
            decision_type=DecisionType.APPROVE,
            rationale="経営ビジョンを確定し全エージェントに通達",
            action_items=["全チームへのビジョン共有", "四半期 OKR への落とし込み"],
            confidence=0.9,
        )
        self._decisions.append(decision)
        return decision


class CfoAgent(KeieiAgent):
    """CFO — 財務分析・仕訳"""

    def __init__(self):
        super().__init__(
            role=AgentRole.CFO,
            name="CFO",
            expertise=["財務", "仕訳", "予算", "キャッシュフロー", "会計",
                       "決算", "コスト", "ROI", "収益", "損益", "資金繰り"],
        )

    def analyze_budget(self, budget: float, spent: float) -> dict:
        """予算執行状況を分析する。"""
        remaining = budget - spent
        utilization = spent / budget * 100 if budget > 0 else 0.0
        return {
            "budget": budget,
            "spent": spent,
            "remaining": remaining,
            "utilization_pct": round(utilization, 1),
            "status": "over" if spent > budget else "ok",
        }

    def forecast(self, current: float, growth_rate: float, periods: int = 4) -> List[float]:
        """複利成長率で将来値を予測する。"""
        result: List[float] = []
        v = current
        for _ in range(periods):
            v = v * (1 + growth_rate)
            result.append(round(v, 2))
        return result

    def roi(self, gain: float, cost: float) -> float:
        """ROI を計算する (0 除算は 0 を返す)。"""
        if cost == 0:
            return 0.0
        return round((gain - cost) / cost * 100, 2)


class CmoAgent(KeieiAgent):
    """CMO — マーケティング戦略"""

    def __init__(self):
        super().__init__(
            role=AgentRole.CMO,
            name="CMO",
            expertise=["マーケティング", "SEO", "広告", "ブランド", "コンテンツ",
                       "LLM", "LLMO", "SNS", "キャンペーン", "CVR", "CTR", "CPA",
                       "ROAS", "リード", "認知", "施策"],
        )

    def plan_campaign(self, name: str, channel: str, budget: float) -> dict:
        """広告キャンペーン計画を生成する。"""
        return {
            "campaign": name,
            "channel": channel,
            "budget": budget,
            "kpis": ["CTR", "CVR", "CPA", "ROAS"],
            "phases": ["企画", "制作", "配信", "分析", "改善"],
        }

    def evaluate_channel(self, channel: str, spend: float,
                          conversions: int) -> dict:
        """チャネルの費用対効果を評価する。"""
        cpa = spend / conversions if conversions > 0 else None
        return {
            "channel": channel,
            "spend": spend,
            "conversions": conversions,
            "cpa": round(cpa, 2) if cpa is not None else None,
            "grade": "A" if (cpa and cpa < 5000) else "B" if (cpa and cpa < 15000) else "C",
        }


class CooAgent(KeieiAgent):
    """COO — オペレーション管理"""

    def __init__(self):
        super().__init__(
            role=AgentRole.COO,
            name="COO",
            expertise=["オペレーション", "プロセス", "品質", "納品", "効率化",
                       "チーム", "スケジュール", "稼働", "リソース", "SLA"],
        )

    def check_delivery(self, tasks: List[AgentTask]) -> dict:
        """タスク群の進捗・ブロッカーを確認する。"""
        done = [t for t in tasks if t.status == TaskStatus.DONE]
        blocked = [t for t in tasks if t.status == TaskStatus.BLOCKED]
        in_progress = [t for t in tasks if t.status == TaskStatus.IN_PROGRESS]
        return {
            "total": len(tasks),
            "done": len(done),
            "in_progress": len(in_progress),
            "blocked": len(blocked),
            "on_track": len(blocked) == 0,
            "completion_rate": round(len(done) / len(tasks) * 100, 1) if tasks else 0.0,
        }

    def escalate_blocked(self, tasks: List[AgentTask]) -> List[AgentTask]:
        """ブロックされたタスクをエスカレーションする。"""
        blocked = [t for t in tasks if t.status == TaskStatus.BLOCKED]
        for t in blocked:
            t.priority = TaskPriority.CRITICAL
        return blocked


class CtoAgent(KeieiAgent):
    """CTO — 技術判断・アーキテクチャ"""

    def __init__(self):
        super().__init__(
            role=AgentRole.CTO,
            name="CTO",
            expertise=["技術", "アーキテクチャ", "AI", "API", "インフラ",
                       "セキュリティ", "パフォーマンス", "Python", "LLM",
                       "データベース", "クラウド", "CI/CD", "テスト"],
        )

    def tech_review(self, proposal: str) -> AgentDecision:
        """技術提案をレビューする。"""
        decision = AgentDecision(
            agent_role=self.role,
            topic=f"技術レビュー: {proposal}",
            decision_type=DecisionType.APPROVE,
            rationale="技術的実現可能性を確認。セキュリティ・スケーラビリティ要件を満たす",
            action_items=["PoC 実装", "セキュリティ監査", "パフォーマンス計測"],
            confidence=0.85,
        )
        self._decisions.append(decision)
        return decision

    def estimate_effort(self, complexity: str) -> dict:
        """タスク規模の工数見積もり（T-shirt sizing）。"""
        sizing = {
            "xs": {"days": 1,  "risk": "low"},
            "s":  {"days": 3,  "risk": "low"},
            "m":  {"days": 5,  "risk": "medium"},
            "l":  {"days": 10, "risk": "medium"},
            "xl": {"days": 20, "risk": "high"},
        }
        key = complexity.lower()
        return sizing.get(key, {"days": 5, "risk": "medium"})


class SecretaryAgent(KeieiAgent):
    """Secretary — inbox/today/archive管理"""

    def __init__(self):
        super().__init__(
            role=AgentRole.SECRETARY,
            name="Secretary",
            expertise=["スケジュール", "inbox", "today", "archive",
                       "会議", "議事録", "タスク管理", "調整", "連絡"],
        )
        self._inbox: List[dict] = []
        self._today: List[dict] = []
        self._archive: List[dict] = []

    def add_inbox(self, item: dict) -> None:
        """inbox にアイテムを追加する。"""
        self._inbox.append(item)

    def triage(self) -> dict:
        """inbox をトリアージして today / archive に仕分ける。"""
        urgent = [i for i in self._inbox if i.get("urgent")]
        normal = [i for i in self._inbox if not i.get("urgent")]
        self._today.extend(urgent)
        self._archive.extend(normal)
        moved = len(self._inbox)
        self._inbox.clear()
        return {
            "processed": moved,
            "today": len(self._today),
            "archived": len(self._archive),
        }

    def get_today(self) -> List[dict]:
        return list(self._today)

    def get_archive(self) -> List[dict]:
        return list(self._archive)

    def clear_today(self) -> int:
        n = len(self._today)
        archived = [dict(item, archived_at=datetime.now().isoformat())
                    for item in self._today]
        self._archive.extend(archived)
        self._today.clear()
        return n


class KnowledgeAgent(KeieiAgent):
    """Knowledge — 教訓・パターン・インシデント記録"""

    def __init__(self):
        super().__init__(
            role=AgentRole.KNOWLEDGE,
            name="Knowledge Manager",
            expertise=["教訓", "パターン", "インシデント", "ナレッジ",
                       "ドキュメント", "学習", "ベストプラクティス", "振り返り", "改善"],
        )
        self._lessons: List[dict] = []
        self._incidents: List[dict] = []

    def record_lesson(self, title: str, body: str,
                      tags: Optional[List[str]] = None) -> dict:
        """教訓・ベストプラクティスを記録する。"""
        lesson = {
            "id": f"L{len(self._lessons) + 1:03d}",
            "title": title,
            "body": body,
            "tags": tags or [],
            "recorded_at": datetime.now().isoformat(),
        }
        self._lessons.append(lesson)
        return lesson

    def record_incident(self, title: str, severity: str,
                        resolution: str) -> dict:
        """インシデントと解決策を記録する。"""
        incident = {
            "id": f"I{len(self._incidents) + 1:03d}",
            "title": title,
            "severity": severity,
            "resolution": resolution,
            "recorded_at": datetime.now().isoformat(),
        }
        self._incidents.append(incident)
        return incident

    def search(self, query: str) -> List[dict]:
        """教訓・インシデントを全文検索する。"""
        q = query.lower()
        results = []
        for lesson in self._lessons:
            if q in lesson["title"].lower() or q in lesson["body"].lower():
                results.append({"type": "lesson", **lesson})
        for incident in self._incidents:
            if q in incident["title"].lower() or q in incident["resolution"].lower():
                results.append({"type": "incident", **incident})
        return results

    def list_lessons(self, tag: Optional[str] = None) -> List[dict]:
        if tag:
            return [l for l in self._lessons if tag in l.get("tags", [])]
        return list(self._lessons)

    def list_incidents(self, severity: Optional[str] = None) -> List[dict]:
        if severity:
            return [i for i in self._incidents if i["severity"] == severity]
        return list(self._incidents)


class ProductsAgent(KeieiAgent):
    """Products — プロダクト管理"""

    def __init__(self):
        super().__init__(
            role=AgentRole.PRODUCTS,
            name="Product Manager",
            expertise=["プロダクト", "ロードマップ", "機能", "リリース",
                       "UI", "UX", "フィードバック", "MVP", "Sprint", "PdM"],
        )
        self._products: List[dict] = []
        self._roadmap: List[dict] = []

    def add_product(self, name: str, description: str,
                    status: str = "active") -> dict:
        """プロダクトを登録する。"""
        product = {
            "id": f"P{len(self._products) + 1:03d}",
            "name": name,
            "description": description,
            "status": status,
        }
        self._products.append(product)
        return product

    def add_roadmap_item(self, feature: str, quarter: str,
                         priority: str = "medium") -> dict:
        """ロードマップにフィーチャーを追加する。"""
        item = {
            "id": f"R{len(self._roadmap) + 1:03d}",
            "feature": feature,
            "quarter": quarter,
            "priority": priority,
        }
        self._roadmap.append(item)
        return item

    def get_roadmap(self, quarter: Optional[str] = None) -> List[dict]:
        if quarter:
            return [i for i in self._roadmap if i["quarter"] == quarter]
        return list(self._roadmap)

    def list_products(self, status: Optional[str] = None) -> List[dict]:
        if status:
            return [p for p in self._products if p["status"] == status]
        return list(self._products)


class ClientsAgent(KeieiAgent):
    """Clients — クライアント管理"""

    def __init__(self):
        super().__init__(
            role=AgentRole.CLIENTS,
            name="Client Manager",
            expertise=["クライアント", "営業", "契約", "提案", "商談",
                       "顧客", "KPI", "レポート", "アカウント", "受注"],
        )
        self._clients: List[dict] = []

    def add_client(self, name: str, industry: str,
                   contract_value: float = 0.0) -> dict:
        """クライアントを登録する。"""
        client = {
            "id": f"C{len(self._clients) + 1:03d}",
            "name": name,
            "industry": industry,
            "contract_value": contract_value,
            "status": "active",
        }
        self._clients.append(client)
        return client

    def list_clients(self, industry: Optional[str] = None,
                     status: Optional[str] = None) -> List[dict]:
        result = self._clients
        if industry:
            result = [c for c in result if c["industry"] == industry]
        if status:
            result = [c for c in result if c["status"] == status]
        return list(result)

    def total_contract_value(self) -> float:
        return sum(c["contract_value"] for c in self._clients if c["status"] == "active")

    def top_clients(self, n: int = 5) -> List[dict]:
        """契約金額上位 n 社を返す。"""
        return sorted(
            [c for c in self._clients if c["status"] == "active"],
            key=lambda c: c["contract_value"],
            reverse=True,
        )[:n]


# ─── KeieiStore ───────────────────────────────────────────────────


class KeieiStore:
    """全エージェントの意思決定・会議・タスクを管理するストア。"""

    def __init__(self):
        self._decisions: List[AgentDecision] = []
        self._meetings: List[MeetingRecord] = []
        self._tasks: List[AgentTask] = []

    # ── 意思決定 ──────────────────────────────────────────────────

    def add_decision(self, decision: AgentDecision) -> None:
        self._decisions.append(decision)

    def get_decisions(self, role: Optional[AgentRole] = None) -> List[AgentDecision]:
        if role is not None:
            return [d for d in self._decisions if d.agent_role == role]
        return list(self._decisions)

    def decision_count(self) -> int:
        return len(self._decisions)

    # ── 会議 ──────────────────────────────────────────────────────

    def add_meeting(self, meeting: MeetingRecord) -> None:
        self._meetings.append(meeting)

    def get_meetings(self) -> List[MeetingRecord]:
        return list(self._meetings)

    def get_meeting(self, meeting_id: str) -> Optional[MeetingRecord]:
        for m in self._meetings:
            if m.id == meeting_id:
                return m
        return None

    # ── タスク ────────────────────────────────────────────────────

    def add_task(self, task: AgentTask) -> None:
        self._tasks.append(task)

    def get_tasks(self, role: Optional[AgentRole] = None) -> List[AgentTask]:
        if role is not None:
            return [t for t in self._tasks if t.owner_role == role]
        return list(self._tasks)


# ─── KeieiOrchestrator ────────────────────────────────────────────


class KeieiOrchestrator:
    """9 エージェントの経営チームを指揮するオーケストレーター。"""

    def __init__(self, store: Optional[KeieiStore] = None):
        self.store = store if store is not None else KeieiStore()
        self.agents: Dict[AgentRole, KeieiAgent] = {
            AgentRole.CEO:       CeoAgent(),
            AgentRole.CFO:       CfoAgent(),
            AgentRole.CMO:       CmoAgent(),
            AgentRole.COO:       CooAgent(),
            AgentRole.CTO:       CtoAgent(),
            AgentRole.SECRETARY: SecretaryAgent(),
            AgentRole.KNOWLEDGE: KnowledgeAgent(),
            AgentRole.PRODUCTS:  ProductsAgent(),
            AgentRole.CLIENTS:   ClientsAgent(),
        }

    # ── エージェント取得 ──────────────────────────────────────────

    def get_agent(self, role: AgentRole) -> KeieiAgent:
        return self.agents[role]

    def list_agents(self) -> List[dict]:
        return [agent.to_dict() for agent in self.agents.values()]

    # ── 諮問 ──────────────────────────────────────────────────────

    def consult(self, topic: str, context: str = "") -> List[AgentDecision]:
        """全エージェント（9体）にトピックを諮問する。"""
        decisions = []
        for agent in self.agents.values():
            decision = agent.think(topic, context)
            self.store.add_decision(decision)
            decisions.append(decision)
        return decisions

    def consult_roles(self, topic: str,
                      roles: List[AgentRole]) -> List[AgentDecision]:
        """指定ロールにのみ諮問する。"""
        decisions = []
        for role in roles:
            decision = self.agents[role].think(topic)
            self.store.add_decision(decision)
            decisions.append(decision)
        return decisions

    # ── 最終意思決定 ──────────────────────────────────────────────

    def make_decision(self, topic: str) -> AgentDecision:
        """全エージェントを諮問した後、CEO が最終判断を下す。"""
        consultations = self.consult(topic)
        approvals = sum(1 for d in consultations
                        if d.decision_type == DecisionType.APPROVE)
        total = len(consultations)

        if approvals >= total * 0.6:
            dt = DecisionType.APPROVE
            rationale = (f"諮問 {total} 件中 {approvals} 件が承認。"
                         f"経営判断: 可決 (同意率 {approvals/total:.0%})")
        elif approvals >= total * 0.3:
            dt = DecisionType.DEFER
            rationale = (f"諮問 {total} 件中 {approvals} 件が承認。"
                         f"追加検討が必要 (同意率 {approvals/total:.0%})")
        else:
            dt = DecisionType.REJECT
            rationale = (f"諮問 {total} 件中 {approvals} 件のみ承認。"
                         f"否決 (同意率 {approvals/total:.0%})")

        # CEO が集約・最終決定
        action_items = []
        for d in consultations:
            if d.action_items:
                action_items.append(d.action_items[0])

        final = AgentDecision(
            agent_role=AgentRole.CEO,
            topic=topic,
            decision_type=dt,
            rationale=rationale,
            action_items=action_items,
            confidence=round(approvals / total, 2),
        )
        self.store.add_decision(final)
        return final

    # ── 会議 ──────────────────────────────────────────────────────

    def hold_meeting(
        self,
        title: str,
        agenda: List[str],
        participants: Optional[List[AgentRole]] = None,
    ) -> MeetingRecord:
        """経営会議を開催して議事録を生成する。"""
        if participants is None:
            participants = list(self.agents.keys())

        decisions: List[AgentDecision] = []
        for item in agenda:
            # 各アジェンダ項目に CEO 以外の参加者が意見を述べる
            reviewers = [r for r in participants if r != AgentRole.CEO][:3]
            for role in reviewers:
                d = self.agents[role].think(item)
                self.store.add_decision(d)
                decisions.append(d)

        meeting_id = f"M{len(self.store.get_meetings()) + 1:03d}"
        minutes_lines = [f"# {title}", ""]
        for item in agenda:
            minutes_lines.append(f"## {item}")
            minutes_lines.append("検討済み\n")

        meeting = MeetingRecord(
            id=meeting_id,
            title=title,
            participants=participants,
            agenda=agenda,
            decisions=decisions,
            minutes="\n".join(minutes_lines),
        )
        self.store.add_meeting(meeting)
        return meeting

    # ── 週次報告 ──────────────────────────────────────────────────

    def weekly_report(self, period: str = "今週") -> str:
        """全エージェントのレポートを統合した週次経営報告書を生成する。"""
        lines = [f"# 週次経営報告書 ({period})", ""]
        for agent in self.agents.values():
            rpt = agent.report(period)
            lines.append(rpt.to_markdown())
            lines.append("")
        total_decisions = self.store.decision_count()
        lines.append(f"---\n**意思決定合計**: {total_decisions} 件")
        return "\n".join(lines)
