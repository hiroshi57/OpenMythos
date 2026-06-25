"""
Sprint 79 — 経営チームエージェント基盤 (keiei-os) テストスイート
"""
import pytest

from open_mythos.skills.keiei_os import (
    AgentRole,
    TaskPriority,
    TaskStatus,
    DecisionType,
    AgentTask,
    AgentDecision,
    AgentReport,
    MeetingRecord,
    KeieiAgent,
    CeoAgent,
    CfoAgent,
    CmoAgent,
    CooAgent,
    CtoAgent,
    SecretaryAgent,
    KnowledgeAgent,
    ProductsAgent,
    ClientsAgent,
    KeieiStore,
    KeieiOrchestrator,
)


# ─── Enums ────────────────────────────────────────────────────────

class TestEnums:
    def test_agent_roles_count(self):
        assert len(AgentRole) == 9

    def test_agent_role_values(self):
        roles = {r.value for r in AgentRole}
        assert "ceo" in roles
        assert "cfo" in roles
        assert "cmo" in roles
        assert "coo" in roles
        assert "cto" in roles
        assert "secretary" in roles
        assert "knowledge" in roles
        assert "products" in roles
        assert "clients" in roles

    def test_task_priority_values(self):
        assert TaskPriority.CRITICAL.value == "critical"
        assert TaskPriority.LOW.value == "low"

    def test_task_status_values(self):
        assert TaskStatus.DONE.value == "done"
        assert TaskStatus.BLOCKED.value == "blocked"

    def test_decision_type_values(self):
        assert DecisionType.APPROVE.value == "approve"
        assert DecisionType.REJECT.value == "reject"
        assert DecisionType.DELEGATE.value == "delegate"


# ─── Data Classes ─────────────────────────────────────────────────

class TestAgentTask:
    def test_create(self):
        t = AgentTask(
            id="T001",
            title="予算策定",
            description="Q3 予算の策定",
            owner_role=AgentRole.CFO,
        )
        assert t.status == TaskStatus.PENDING
        assert t.priority == TaskPriority.MEDIUM

    def test_to_dict(self):
        t = AgentTask(id="T001", title="test", description="desc",
                      owner_role=AgentRole.CEO)
        d = t.to_dict()
        assert d["id"] == "T001"
        assert d["owner_role"] == "ceo"
        assert d["status"] == "pending"


class TestAgentDecision:
    def test_create(self):
        d = AgentDecision(
            agent_role=AgentRole.CEO,
            topic="採用強化",
            decision_type=DecisionType.APPROVE,
            rationale="成長フェーズに必要",
        )
        assert 0 < d.confidence <= 1.0
        assert d.action_items == []

    def test_to_dict(self):
        d = AgentDecision(
            agent_role=AgentRole.CTO,
            topic="AI導入",
            decision_type=DecisionType.DEFER,
            rationale="検討中",
            action_items=["PoC"],
        )
        result = d.to_dict()
        assert result["agent_role"] == "cto"
        assert result["decision_type"] == "defer"
        assert "PoC" in result["action_items"]


class TestAgentReport:
    def test_to_markdown(self):
        r = AgentReport(
            agent_role=AgentRole.CMO,
            period="2026-Q2",
            highlights=["新規リード 100 件"],
            issues=["予算超過"],
            metrics={"CTR": "3.2%"},
            next_actions=["SNS 強化"],
        )
        md = r.to_markdown()
        assert "CMO" in md
        assert "2026-Q2" in md
        assert "新規リード" in md
        assert "予算超過" in md
        assert "CTR" in md

    def test_to_dict(self):
        r = AgentReport(
            agent_role=AgentRole.CFO,
            period="今週",
            highlights=["決算完了"],
        )
        d = r.to_dict()
        assert d["agent_role"] == "cfo"
        assert d["highlights"] == ["決算完了"]


class TestMeetingRecord:
    def test_create(self):
        m = MeetingRecord(
            id="M001",
            title="週次経営会議",
            participants=[AgentRole.CEO, AgentRole.CFO],
            agenda=["Q3 計画", "採用"],
        )
        assert m.decisions == []
        assert m.minutes == ""

    def test_to_dict(self):
        m = MeetingRecord(
            id="M001",
            title="MTG",
            participants=[AgentRole.CEO],
            agenda=["課題検討"],
        )
        d = m.to_dict()
        assert d["id"] == "M001"
        assert "ceo" in d["participants"]


# ─── KeieiAgent Base ──────────────────────────────────────────────

class TestKeieiAgent:
    def setup_method(self):
        self.agent = KeieiAgent(
            role=AgentRole.CEO,
            name="Test CEO",
            expertise=["経営", "戦略"],
        )

    def test_think_relevant(self):
        d = self.agent.think("経営判断が必要")
        assert d.decision_type == DecisionType.APPROVE

    def test_think_irrelevant(self):
        d = self.agent.think("JavaScriptのバグ修正")
        assert d.decision_type == DecisionType.DELEGATE

    def test_assign_task(self):
        t = AgentTask(id="T001", title="戦略策定", description="",
                      owner_role=AgentRole.CEO)
        self.agent.assign_task(t)
        assert len(self.agent.list_tasks()) == 1

    def test_complete_task(self):
        t = AgentTask(id="T001", title="戦略策定", description="",
                      owner_role=AgentRole.CEO)
        self.agent.assign_task(t)
        result = self.agent.complete_task("T001")
        assert result is True
        assert self.agent.list_tasks(TaskStatus.DONE)[0].id == "T001"

    def test_complete_nonexistent_task(self):
        assert self.agent.complete_task("NONE") is False

    def test_list_tasks_by_status(self):
        t1 = AgentTask(id="T1", title="A", description="", owner_role=AgentRole.CEO)
        t2 = AgentTask(id="T2", title="B", description="", owner_role=AgentRole.CEO,
                       status=TaskStatus.DONE)
        self.agent.assign_task(t1)
        self.agent.assign_task(t2)
        pending = self.agent.list_tasks(TaskStatus.PENDING)
        assert len(pending) == 1

    def test_report(self):
        r = self.agent.report("今週")
        assert r.agent_role == AgentRole.CEO
        assert "今週" in r.period

    def test_to_dict(self):
        d = self.agent.to_dict()
        assert d["role"] == "ceo"
        assert d["name"] == "Test CEO"


# ─── Specific Agents ──────────────────────────────────────────────

class TestCeoAgent:
    def setup_method(self):
        self.agent = CeoAgent()

    def test_role(self):
        assert self.agent.role == AgentRole.CEO

    def test_prioritize(self):
        items = ["採用計画", "緊急: サーバー障害", "予算策定"]
        ordered = self.agent.prioritize(items)
        assert "緊急" in ordered[0]

    def test_set_vision(self):
        d = self.agent.set_vision("2030年に業界No.1")
        assert d.decision_type == DecisionType.APPROVE
        assert "ビジョン設定" in d.topic


class TestCfoAgent:
    def setup_method(self):
        self.agent = CfoAgent()

    def test_analyze_budget_ok(self):
        r = self.agent.analyze_budget(1_000_000, 800_000)
        assert r["status"] == "ok"
        assert r["utilization_pct"] == 80.0
        assert r["remaining"] == 200_000

    def test_analyze_budget_over(self):
        r = self.agent.analyze_budget(500_000, 600_000)
        assert r["status"] == "over"

    def test_forecast(self):
        result = self.agent.forecast(100, 0.1, 3)
        assert len(result) == 3
        assert result[0] == pytest.approx(110.0)

    def test_roi(self):
        assert self.agent.roi(150_000, 100_000) == pytest.approx(50.0)

    def test_roi_zero_cost(self):
        assert self.agent.roi(100, 0) == 0.0


class TestCmoAgent:
    def setup_method(self):
        self.agent = CmoAgent()

    def test_plan_campaign(self):
        plan = self.agent.plan_campaign("春季キャンペーン", "Google", 500_000)
        assert plan["campaign"] == "春季キャンペーン"
        assert "CTR" in plan["kpis"]
        assert len(plan["phases"]) >= 4

    def test_evaluate_channel_grade_a(self):
        r = self.agent.evaluate_channel("Google", 100_000, 30)
        assert r["grade"] == "A"
        assert r["cpa"] == pytest.approx(100_000 / 30, rel=1e-3)

    def test_evaluate_channel_no_conversions(self):
        r = self.agent.evaluate_channel("TikTok", 50_000, 0)
        assert r["cpa"] is None


class TestCooAgent:
    def setup_method(self):
        self.agent = CooAgent()

    def test_check_delivery(self):
        tasks = [
            AgentTask(id="T1", title="A", description="", owner_role=AgentRole.COO,
                      status=TaskStatus.DONE),
            AgentTask(id="T2", title="B", description="", owner_role=AgentRole.COO,
                      status=TaskStatus.IN_PROGRESS),
            AgentTask(id="T3", title="C", description="", owner_role=AgentRole.COO,
                      status=TaskStatus.BLOCKED),
        ]
        r = self.agent.check_delivery(tasks)
        assert r["total"] == 3
        assert r["done"] == 1
        assert r["blocked"] == 1
        assert r["on_track"] is False

    def test_check_delivery_empty(self):
        r = self.agent.check_delivery([])
        assert r["completion_rate"] == 0.0

    def test_escalate_blocked(self):
        t = AgentTask(id="T1", title="X", description="", owner_role=AgentRole.COO,
                      status=TaskStatus.BLOCKED, priority=TaskPriority.LOW)
        result = self.agent.escalate_blocked([t])
        assert len(result) == 1
        assert result[0].priority == TaskPriority.CRITICAL


class TestCtoAgent:
    def setup_method(self):
        self.agent = CtoAgent()

    def test_tech_review(self):
        d = self.agent.tech_review("LLM API 統合")
        assert d.decision_type == DecisionType.APPROVE
        assert "技術レビュー" in d.topic

    def test_estimate_effort(self):
        r = self.agent.estimate_effort("m")
        assert r["days"] == 5
        assert r["risk"] == "medium"

    def test_estimate_effort_xl(self):
        r = self.agent.estimate_effort("XL")
        assert r["days"] == 20
        assert r["risk"] == "high"

    def test_estimate_effort_unknown(self):
        r = self.agent.estimate_effort("unknown")
        assert "days" in r


class TestSecretaryAgent:
    def setup_method(self):
        self.agent = SecretaryAgent()

    def test_add_inbox(self):
        self.agent.add_inbox({"subject": "承認依頼", "urgent": True})
        self.agent.add_inbox({"subject": "FYI", "urgent": False})
        result = self.agent.triage()
        assert result["processed"] == 2
        assert result["today"] == 1
        assert result["archived"] == 1

    def test_get_today(self):
        self.agent.add_inbox({"subject": "緊急MTG", "urgent": True})
        self.agent.triage()
        today = self.agent.get_today()
        assert len(today) == 1
        assert today[0]["subject"] == "緊急MTG"

    def test_clear_today(self):
        self.agent.add_inbox({"subject": "task", "urgent": True})
        self.agent.triage()          # urgent → today (archive still 0)
        n = self.agent.clear_today() # today → archive (1 item)
        assert n == 1
        assert self.agent.get_today() == []
        assert len(self.agent.get_archive()) == 1  # clear_today 分だけ


class TestKnowledgeAgent:
    def setup_method(self):
        self.agent = KnowledgeAgent()

    def test_record_lesson(self):
        l = self.agent.record_lesson("キャンペーン教訓", "早期計画が重要", ["marketing"])
        assert l["id"] == "L001"
        assert "marketing" in l["tags"]

    def test_record_incident(self):
        i = self.agent.record_incident("API障害", "high", "ロールバックで解決")
        assert i["id"] == "I001"
        assert i["severity"] == "high"

    def test_search_lesson(self):
        self.agent.record_lesson("SEO施策", "コンテンツ品質が重要", ["seo"])
        results = self.agent.search("SEO")
        assert len(results) == 1
        assert results[0]["type"] == "lesson"

    def test_search_incident(self):
        self.agent.record_incident("DB障害", "critical", "フェイルオーバーで復旧")
        results = self.agent.search("フェイルオーバー")
        assert len(results) == 1
        assert results[0]["type"] == "incident"

    def test_search_no_match(self):
        results = self.agent.search("存在しないキーワード")
        assert results == []

    def test_list_lessons_by_tag(self):
        self.agent.record_lesson("L1", "body", ["marketing"])
        self.agent.record_lesson("L2", "body", ["tech"])
        result = self.agent.list_lessons(tag="marketing")
        assert len(result) == 1


class TestProductsAgent:
    def setup_method(self):
        self.agent = ProductsAgent()

    def test_add_product(self):
        p = self.agent.add_product("Forté.AI", "LLMOプラットフォーム")
        assert p["id"] == "P001"
        assert p["status"] == "active"

    def test_add_roadmap_item(self):
        r = self.agent.add_roadmap_item("マルチモーダル対応", "2026-Q3", "high")
        assert r["feature"] == "マルチモーダル対応"
        assert r["quarter"] == "2026-Q3"

    def test_get_roadmap_filtered(self):
        self.agent.add_roadmap_item("F1", "2026-Q3")
        self.agent.add_roadmap_item("F2", "2026-Q4")
        q3 = self.agent.get_roadmap("2026-Q3")
        assert len(q3) == 1

    def test_list_products_by_status(self):
        self.agent.add_product("P1", "desc1", "active")
        self.agent.add_product("P2", "desc2", "deprecated")
        active = self.agent.list_products("active")
        assert len(active) == 1


class TestClientsAgent:
    def setup_method(self):
        self.agent = ClientsAgent()

    def test_add_client(self):
        c = self.agent.add_client("サンプル株式会社", "金融", 1_200_000)
        assert c["id"] == "C001"
        assert c["status"] == "active"

    def test_list_clients_by_industry(self):
        self.agent.add_client("A社", "金融", 500_000)
        self.agent.add_client("B社", "不動産", 300_000)
        finance = self.agent.list_clients(industry="金融")
        assert len(finance) == 1

    def test_total_contract_value(self):
        self.agent.add_client("A社", "金融", 1_000_000)
        self.agent.add_client("B社", "IT", 2_000_000)
        assert self.agent.total_contract_value() == 3_000_000

    def test_top_clients(self):
        for i in range(7):
            self.agent.add_client(f"C{i}", "IT", float(i * 100_000))
        top = self.agent.top_clients(3)
        assert len(top) == 3
        assert top[0]["contract_value"] >= top[1]["contract_value"]


# ─── KeieiStore ───────────────────────────────────────────────────

class TestKeieiStore:
    def setup_method(self):
        self.store = KeieiStore()

    def test_add_and_get_decision(self):
        d = AgentDecision(
            agent_role=AgentRole.CEO,
            topic="採用",
            decision_type=DecisionType.APPROVE,
            rationale="成長に必要",
        )
        self.store.add_decision(d)
        assert self.store.decision_count() == 1

    def test_get_decisions_by_role(self):
        for role in [AgentRole.CEO, AgentRole.CFO, AgentRole.CEO]:
            self.store.add_decision(AgentDecision(
                agent_role=role, topic="test",
                decision_type=DecisionType.APPROVE, rationale="test",
            ))
        ceo_decisions = self.store.get_decisions(AgentRole.CEO)
        assert len(ceo_decisions) == 2

    def test_add_and_get_meeting(self):
        m = MeetingRecord(id="M001", title="経営会議",
                          participants=[AgentRole.CEO], agenda=["予算"])
        self.store.add_meeting(m)
        assert len(self.store.get_meetings()) == 1

    def test_get_meeting_by_id(self):
        m = MeetingRecord(id="M001", title="MTG",
                          participants=[AgentRole.CEO], agenda=[])
        self.store.add_meeting(m)
        found = self.store.get_meeting("M001")
        assert found is not None
        assert found.title == "MTG"

    def test_get_meeting_not_found(self):
        assert self.store.get_meeting("NONE") is None

    def test_add_and_get_task(self):
        t = AgentTask(id="T001", title="A", description="", owner_role=AgentRole.CFO)
        self.store.add_task(t)
        tasks = self.store.get_tasks(AgentRole.CFO)
        assert len(tasks) == 1


# ─── KeieiOrchestrator ────────────────────────────────────────────

class TestKeieiOrchestrator:
    def setup_method(self):
        self.orch = KeieiOrchestrator()

    def test_agents_count(self):
        assert len(self.orch.agents) == 9

    def test_list_agents(self):
        agents = self.orch.list_agents()
        assert len(agents) == 9
        roles = {a["role"] for a in agents}
        assert "ceo" in roles
        assert "clients" in roles

    def test_get_agent(self):
        ceo = self.orch.get_agent(AgentRole.CEO)
        assert isinstance(ceo, CeoAgent)

    def test_consult_returns_9_decisions(self):
        decisions = self.orch.consult("新規事業参入の検討")
        assert len(decisions) == 9

    def test_consult_roles(self):
        decisions = self.orch.consult_roles(
            "財務計画", [AgentRole.CFO, AgentRole.CEO]
        )
        assert len(decisions) == 2

    def test_make_decision_approve(self):
        # 経営・戦略は多くのエージェントの専門領域に関連する
        d = self.orch.make_decision("経営戦略の策定")
        assert d.agent_role == AgentRole.CEO
        assert d.decision_type in {
            DecisionType.APPROVE, DecisionType.DEFER, DecisionType.REJECT
        }
        assert 0 <= d.confidence <= 1.0

    def test_make_decision_stores_in_store(self):
        before = self.orch.store.decision_count()
        self.orch.make_decision("テスト判断")
        # consult (9) + final (1) = 10
        assert self.orch.store.decision_count() == before + 10

    def test_hold_meeting(self):
        meeting = self.orch.hold_meeting(
            "Q3 経営会議",
            ["予算承認", "採用計画"],
        )
        assert meeting.id == "M001"
        assert meeting.title == "Q3 経営会議"
        assert len(meeting.agenda) == 2
        assert len(meeting.participants) == 9
        assert meeting.minutes != ""

    def test_hold_meeting_with_participants(self):
        meeting = self.orch.hold_meeting(
            "緊急CFO会議",
            ["資金調達"],
            participants=[AgentRole.CEO, AgentRole.CFO],
        )
        assert len(meeting.participants) == 2

    def test_weekly_report(self):
        report = self.orch.weekly_report("2026-W26")
        assert "週次経営報告書" in report
        assert "2026-W26" in report
        assert "CEO" in report
        assert "CFO" in report
        assert "意思決定合計" in report

    def test_store_shared_with_agents(self):
        store = KeieiStore()
        orch = KeieiOrchestrator(store=store)
        orch.consult("テスト")
        assert store.decision_count() == 9
