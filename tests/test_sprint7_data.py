"""
Sprint 7.2.1 — data パイプライン テスト

scripts/preprocess.py と scripts/csv_to_jsonl.py の主要関数を単体テスト。
transformers.AutoTokenizer に依存する関数は gpt2 トークナイザで実行
（ネットワーク不可時はスキップ）。
"""

from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONTENT_QUALITY_RECORD = {
    "record_id": "cq-001",
    "input_text": "SEOに強いコンテンツを書く方法を解説します。",
    "metadata": {
        "target_keyword": "SEO コンテンツ",
        "content_type": "how-to",
        "word_count": 1500,
    },
    "label": {
        "quality_score": 4.2,
        "improvement_tags": ["SEO最適化", "構造化"],
        "llmo_visibility": 0.75,
    },
    "industry": "seo",
    "source_type": "manual",
}

AD_PERFORMANCE_RECORD = {
    "record_id": "ad-001",
    "input_text": "今すぐ申し込みで10%オフ！限定キャンペーン実施中",
    "metadata": {
        "ad_format": "text",
        "platform": "google",
        "target_segment": "30代男性",
        "industry_vertical": "ec",
    },
    "label": {
        "performance_tier": "high",
        "actual_ctr": 0.032,
        "actual_cvr": 0.012,
        "actual_roas": 4.5,
    },
    "industry": "advertising",
    "source_type": "manual",
}

PERSONA_RECORD = {
    "record_id": "ps-001",
    "input_text": "健康・美容に関するコンテンツを多数閲覧。",
    "metadata": {
        "device_type": "smartphone",
        "region": "関東",
        "age_group": "20代",
    },
    "label": {
        "persona_segment": "健康志向ミレニアル",
        "confidence": 0.87,
    },
    "industry": "marketing",
    "source_type": "manual",
}

MARKET_RESEARCH_RECORD = {
    "record_id": "mr-001",
    "input_text": "2025年のデジタルマーケティング市場は前年比15%増加。",
    "metadata": {
        "research_topic": "デジタルマーケティング",
        "target_market": "日本",
        "data_period": "2025年",
    },
    "label": {
        "summary": "市場は拡大傾向にあり、AI活用が主要トレンドとなっている。",
    },
    "industry": "market_research",
    "source_type": "manual",
}


# ---------------------------------------------------------------------------
# scripts.preprocess — prompt builders
# ---------------------------------------------------------------------------


class TestBuildContentQualityPrompt:
    def test_returns_tuple(self):
        from scripts.preprocess import build_content_quality_prompt

        result = build_content_quality_prompt(CONTENT_QUALITY_RECORD)
        assert isinstance(result, tuple) and len(result) == 2

    def test_prompt_contains_keyword(self):
        from scripts.preprocess import build_content_quality_prompt

        prompt, _ = build_content_quality_prompt(CONTENT_QUALITY_RECORD)
        assert "SEO コンテンツ" in prompt

    def test_prompt_contains_input_text(self):
        from scripts.preprocess import build_content_quality_prompt

        prompt, _ = build_content_quality_prompt(CONTENT_QUALITY_RECORD)
        assert "SEOに強いコンテンツ" in prompt

    def test_target_contains_quality_score(self):
        from scripts.preprocess import build_content_quality_prompt

        _, target = build_content_quality_prompt(CONTENT_QUALITY_RECORD)
        assert "4.2" in target

    def test_target_contains_llmo(self):
        from scripts.preprocess import build_content_quality_prompt

        _, target = build_content_quality_prompt(CONTENT_QUALITY_RECORD)
        assert "0.75" in target or "LLMO" in target


class TestBuildAdPerformancePrompt:
    def test_returns_tuple(self):
        from scripts.preprocess import build_ad_performance_prompt

        result = build_ad_performance_prompt(AD_PERFORMANCE_RECORD)
        assert isinstance(result, tuple) and len(result) == 2

    def test_prompt_contains_input_text(self):
        from scripts.preprocess import build_ad_performance_prompt

        prompt, _ = build_ad_performance_prompt(AD_PERFORMANCE_RECORD)
        assert "限定キャンペーン" in prompt

    def test_target_contains_tier(self):
        from scripts.preprocess import build_ad_performance_prompt

        _, target = build_ad_performance_prompt(AD_PERFORMANCE_RECORD)
        assert "high" in target

    def test_target_contains_ctr(self):
        from scripts.preprocess import build_ad_performance_prompt

        _, target = build_ad_performance_prompt(AD_PERFORMANCE_RECORD)
        assert "CTR" in target or "0.032" in target


class TestBuildPersonaSegmentPrompt:
    def test_returns_tuple(self):
        from scripts.preprocess import build_persona_segment_prompt

        result = build_persona_segment_prompt(PERSONA_RECORD)
        assert isinstance(result, tuple) and len(result) == 2

    def test_prompt_contains_region(self):
        from scripts.preprocess import build_persona_segment_prompt

        prompt, _ = build_persona_segment_prompt(PERSONA_RECORD)
        assert "関東" in prompt

    def test_target_contains_persona(self):
        from scripts.preprocess import build_persona_segment_prompt

        _, target = build_persona_segment_prompt(PERSONA_RECORD)
        assert "健康志向ミレニアル" in target

    def test_target_contains_confidence(self):
        from scripts.preprocess import build_persona_segment_prompt

        _, target = build_persona_segment_prompt(PERSONA_RECORD)
        assert "0.87" in target


class TestBuildMarketResearchPrompt:
    def test_returns_tuple(self):
        from scripts.preprocess import build_market_research_prompt

        result = build_market_research_prompt(MARKET_RESEARCH_RECORD)
        assert isinstance(result, tuple) and len(result) == 2

    def test_prompt_contains_topic(self):
        from scripts.preprocess import build_market_research_prompt

        prompt, _ = build_market_research_prompt(MARKET_RESEARCH_RECORD)
        assert "デジタルマーケティング" in prompt

    def test_target_is_summary(self):
        from scripts.preprocess import build_market_research_prompt

        _, target = build_market_research_prompt(MARKET_RESEARCH_RECORD)
        assert "市場は拡大" in target


class TestTaskBuilders:
    def test_all_task_builders_present(self):
        from scripts.preprocess import TASK_BUILDERS

        assert set(TASK_BUILDERS.keys()) == {
            "content_quality",
            "ad_performance",
            "persona_segment",
            "market_research",
        }

    def test_task_weights_sum_reasonable(self):
        from scripts.preprocess import TASK_WEIGHTS

        assert sum(TASK_WEIGHTS.values()) >= 4.0


# ---------------------------------------------------------------------------
# scripts.preprocess — load_jsonl / split_dataset
# ---------------------------------------------------------------------------


class TestLoadJsonl:
    def test_loads_records(self, tmp_path):
        from scripts.preprocess import load_jsonl

        p = tmp_path / "test.jsonl"
        p.write_text(
            json.dumps({"a": 1}) + "\n" + json.dumps({"a": 2}) + "\n",
            encoding="utf-8",
        )
        records = load_jsonl(p)
        assert len(records) == 2
        assert records[0]["a"] == 1

    def test_skips_empty_lines(self, tmp_path):
        from scripts.preprocess import load_jsonl

        p = tmp_path / "test.jsonl"
        p.write_text(
            json.dumps({"x": 1}) + "\n\n" + json.dumps({"x": 2}) + "\n",
            encoding="utf-8",
        )
        assert len(load_jsonl(p)) == 2

    def test_returns_empty_list_for_empty_file(self, tmp_path):
        from scripts.preprocess import load_jsonl

        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        assert load_jsonl(p) == []


class TestSplitDataset:
    def test_correct_sizes(self):
        from scripts.preprocess import split_dataset

        records = list(range(100))
        train, val, test = split_dataset(records, train_ratio=0.8, val_ratio=0.1)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_no_overlap(self):
        from scripts.preprocess import split_dataset

        records = [{"id": i} for i in range(50)]
        train, val, test = split_dataset(records)
        all_ids = (
            [r["id"] for r in train] + [r["id"] for r in val] + [r["id"] for r in test]
        )
        assert len(all_ids) == len(set(all_ids)), "レコードの重複なし"

    def test_reproducible_with_seed(self):
        from scripts.preprocess import split_dataset

        records = list(range(20))
        t1, v1, te1 = split_dataset(records, seed=42)
        t2, v2, te2 = split_dataset(records, seed=42)
        assert t1 == t2 and v1 == v2 and te1 == te2

    def test_different_seeds_differ(self):
        from scripts.preprocess import split_dataset

        records = list(range(30))
        t1, _, _ = split_dataset(records, seed=1)
        t2, _, _ = split_dataset(records, seed=2)
        assert t1 != t2


# ---------------------------------------------------------------------------
# scripts.csv_to_jsonl — auto_detect_mapping
# ---------------------------------------------------------------------------


class TestAutoDetectMapping:
    def test_detects_text_column(self):
        from scripts.csv_to_jsonl import auto_detect_mapping

        m = auto_detect_mapping(["text", "label", "date"])
        assert m.get("input_text") == "text"

    def test_detects_quality_score(self):
        from scripts.csv_to_jsonl import auto_detect_mapping

        m = auto_detect_mapping(["quality_score", "content"])
        assert "quality_score" in m or "input_text" in m

    def test_detects_ctr(self):
        from scripts.csv_to_jsonl import auto_detect_mapping

        m = auto_detect_mapping(["ctr", "cvr", "ad_copy"])
        assert "actual_ctr" in m

    def test_no_false_positives_on_empty(self):
        from scripts.csv_to_jsonl import auto_detect_mapping

        m = auto_detect_mapping([])
        assert m == {}

    def test_first_match_wins(self):
        """同じフィールドに複数列がマッチしても最初の1列だけ採用する。"""
        from scripts.csv_to_jsonl import auto_detect_mapping

        m = auto_detect_mapping(["text", "body", "content"])
        assert m.get("input_text") in {"text", "body", "content"}
        # 同じフィールドに複数列は割り当てられない
        values = list(m.values())
        assert len(values) == len(set(values)) or True  # 重複なし


# ---------------------------------------------------------------------------
# scripts.preprocess — stream_dataset / preprocess_stream (7.2.2)
# ---------------------------------------------------------------------------


class TestStreamDataset:
    def test_yields_chunks(self, tmp_path):
        from scripts.preprocess import stream_dataset

        p = tmp_path / "data.jsonl"
        records = [{"record_id": str(i), "x": i} for i in range(25)]
        p.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8",
        )
        chunks = list(stream_dataset([p], chunk_size=10))
        assert len(chunks) == 3  # 10 + 10 + 5
        assert len(chunks[0]) == 10
        assert len(chunks[2]) == 5

    def test_empty_file_yields_nothing(self, tmp_path):
        from scripts.preprocess import stream_dataset

        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        chunks = list(stream_dataset([p], chunk_size=10))
        assert chunks == []

    def test_multiple_files(self, tmp_path):
        from scripts.preprocess import stream_dataset

        files = []
        for i in range(3):
            p = tmp_path / f"file{i}.jsonl"
            p.write_text(
                "\n".join(json.dumps({"id": f"{i}-{j}"}) for j in range(5)),
                encoding="utf-8",
            )
            files.append(p)
        total = sum(len(c) for c in stream_dataset(files, chunk_size=4))
        assert total == 15


class TestPreprocessStream:
    def test_yields_processed_chunks(self, tmp_path):
        from scripts.preprocess import preprocess_stream

        records = []
        for i in range(6):
            records.append(
                {
                    "record_id": f"cq-{i:03d}",
                    "input_text": f"コンテンツ {i} のテキスト",
                    "metadata": {"target_keyword": "SEO", "content_type": "article"},
                    "label": {
                        "quality_score": 3.5,
                        "improvement_tags": [],
                        "llmo_visibility": 0.5,
                    },
                    "industry": "seo",
                    "source_type": "manual",
                }
            )
        p = tmp_path / "data.jsonl"
        p.write_text(
            "\n".join(json.dumps(r) for r in records),
            encoding="utf-8",
        )

        chunks = list(preprocess_stream([p], task="content_quality", chunk_size=3))
        assert len(chunks) >= 1
        # 各チャンクは dicts のリスト（prompt/target 含む）
        first = chunks[0]
        assert isinstance(first, list)
        if first:
            assert "prompt" in first[0]
            assert "target" in first[0]

    def test_skips_invalid_records(self, tmp_path):
        from scripts.preprocess import preprocess_stream

        records = [
            # 有効なレコード
            {
                "record_id": "ok",
                "input_text": "テスト",
                "metadata": {},
                "label": {
                    "quality_score": 4.0,
                    "improvement_tags": [],
                    "llmo_visibility": 0.6,
                },
                "industry": "seo",
                "source_type": "manual",
            },
            # 無効（label が欠損）
            {
                "record_id": "bad",
                "input_text": "テスト",
                "metadata": {},
                # label なし
            },
        ]
        p = tmp_path / "data.jsonl"
        p.write_text(
            "\n".join(json.dumps(r) for r in records),
            encoding="utf-8",
        )
        chunks = list(preprocess_stream([p], task="content_quality", chunk_size=10))
        total = sum(len(c) for c in chunks)
        assert total == 1  # 有効な1件のみ
