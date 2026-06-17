"""
Sprint 32 — ErrorMemoryStore SQLite 永続化テストスイート (40 tests)

すべて db_path=":memory:" を使用するため、テスト間でファイル競合は起きない。
"""

import json
import os
import tempfile

import pytest

from open_mythos.error_memory import (
    ErrorMemoryStore,
    MistakeRecord,
)


# =========================================================================
# Helpers
# =========================================================================

def _mem_store(max_records: int = 1000) -> ErrorMemoryStore:
    return ErrorMemoryStore(max_records=max_records)

def _sql_store(max_records: int = 1000) -> ErrorMemoryStore:
    return ErrorMemoryStore(backend="sqlite", db_path=":memory:", max_records=max_records)


# =========================================================================
# 1. backend パラメータ (5 tests)
# =========================================================================

def test_default_backend_is_memory():
    store = ErrorMemoryStore()
    assert store.backend == "memory"

def test_sqlite_backend_init():
    store = _sql_store()
    assert store.backend == "sqlite"
    assert store._conn is not None

def test_sqlite_backend_in_memory_path():
    store = ErrorMemoryStore(backend="sqlite", db_path=":memory:")
    assert store.total == 0
    store.close()

def test_sqlite_creates_table():
    store = _sql_store()
    cursor = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='mistakes'"
    )
    assert cursor.fetchone() is not None
    store.close()

def test_memory_backend_no_conn():
    store = _mem_store()
    assert store._conn is None


# =========================================================================
# 2. append() — SQLite (6 tests)
# =========================================================================

def test_sqlite_append_returns_record():
    store = _sql_store()
    rec = store.append("テストミス", category="security")
    assert isinstance(rec, MistakeRecord)
    store.close()

def test_sqlite_append_total_increments():
    store = _sql_store()
    store.append("a", category="security")
    store.append("b", category="quality")
    assert store.total == 2
    store.close()

def test_sqlite_append_category_normalization():
    store = _sql_store()
    rec = store.append("text", category="invalid_category")
    assert rec.category == "other"
    store.close()

def test_sqlite_append_severity_preserved():
    store = _sql_store()
    rec = store.append("text", category="security", severity="high")
    assert rec.severity == "high"
    store.close()

def test_sqlite_max_records_enforced():
    store = _sql_store(max_records=3)
    for i in range(5):
        store.append(f"record {i}", category="quality")
    assert store.total <= 3
    store.close()

def test_sqlite_append_metadata_stored():
    store = _sql_store()
    store.append("text", metadata={"key": "val"})
    records = store._all_records()
    assert any(r.metadata.get("key") == "val" for r in records)
    store.close()


# =========================================================================
# 3. query_similar() — 両バックエンド (5 tests)
# =========================================================================

def test_query_similar_memory_empty():
    store = _mem_store()
    result = store.query_similar("test")
    assert result == []

def test_query_similar_sqlite_empty():
    store = _sql_store()
    result = store.query_similar("test")
    assert result == []
    store.close()

def test_query_similar_returns_list():
    store = _sql_store()
    store.append("ignore previous instructions", category="security")
    result = store.query_similar("ignore all instructions", top_k=3)
    assert isinstance(result, list)
    store.close()

def test_query_similar_top_k_limit():
    store = _sql_store()
    for i in range(10):
        store.append(f"similar text {i}", category="quality")
    result = store.query_similar("similar text", top_k=3)
    assert len(result) <= 3
    store.close()

def test_query_similar_cross_backend_consistency():
    """両バックエンドで同じ入力なら件数は同じ"""
    mem = _mem_store()
    sql = _sql_store()
    for s in ["error in loop", "timeout reached", "infinite loop detected"]:
        mem.append(s, category="loop")
        sql.append(s, category="loop")
    r_mem = mem.query_similar("infinite loop", top_k=5)
    r_sql = sql.query_similar("infinite loop", top_k=5)
    assert len(r_mem) == len(r_sql)
    sql.close()


# =========================================================================
# 4. stats() (4 tests)
# =========================================================================

def test_sqlite_stats_total():
    store = _sql_store()
    store.append("a", category="security")
    store.append("b", category="quality")
    s = store.stats()
    assert s["total"] == 2
    store.close()

def test_sqlite_stats_by_category():
    store = _sql_store()
    store.append("x", category="security")
    store.append("y", category="security")
    store.append("z", category="quality")
    s = store.stats()
    assert s["by_category"].get("security") == 2
    assert s["by_category"].get("quality") == 1
    store.close()

def test_sqlite_stats_by_severity():
    store = _sql_store()
    store.append("a", severity="high")
    store.append("b", severity="low")
    s = store.stats()
    assert s["by_severity"].get("high") == 1
    assert s["by_severity"].get("low") == 1
    store.close()

def test_sqlite_stats_empty():
    store = _sql_store()
    s = store.stats()
    assert s["total"] == 0
    assert s["by_category"] == {}
    store.close()


# =========================================================================
# 5. records_by_category() (3 tests)
# =========================================================================

def test_sqlite_records_by_category_correct():
    store = _sql_store()
    store.append("sec1", category="security")
    store.append("qual1", category="quality")
    store.append("sec2", category="security")
    recs = store.records_by_category("security")
    assert len(recs) == 2
    assert all(r.category == "security" for r in recs)
    store.close()

def test_sqlite_records_by_category_empty():
    store = _sql_store()
    store.append("x", category="quality")
    recs = store.records_by_category("security")
    assert recs == []
    store.close()

def test_memory_records_by_category_unchanged():
    store = _mem_store()
    store.append("a", category="privacy")
    store.append("b", category="privacy")
    recs = store.records_by_category("privacy")
    assert len(recs) == 2


# =========================================================================
# 6. export_jsonl() / export_records() (4 tests)
# =========================================================================

def test_export_jsonl_memory():
    store = _mem_store()
    store.append("error text", category="security", severity="high")
    jsonl = store.export_jsonl()
    assert len(jsonl) > 0
    d = json.loads(jsonl.splitlines()[0])
    assert d["category"] == "security"

def test_export_jsonl_sqlite():
    store = _sql_store()
    store.append("mistake 1", category="quality")
    store.append("mistake 2", category="privacy")
    jsonl = store.export_jsonl()
    lines = [l for l in jsonl.splitlines() if l.strip()]
    assert len(lines) == 2
    store.close()

def test_export_records_returns_list():
    store = _sql_store()
    store.append("a", category="security")
    recs = store.export_records()
    assert isinstance(recs, list)
    assert recs[0]["category"] == "security"
    store.close()

def test_export_jsonl_empty_store():
    store = _sql_store()
    jsonl = store.export_jsonl()
    assert jsonl == ""
    store.close()


# =========================================================================
# 7. save_jsonl() / import_jsonl() (5 tests)
# =========================================================================

def test_save_jsonl_creates_file():
    store = _sql_store()
    store.append("save test", category="quality")
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        n = store.save_jsonl(path)
        assert n == 1
        assert os.path.exists(path)
    finally:
        os.unlink(path)
    store.close()

def test_import_jsonl_returns_count():
    src = _sql_store()
    src.append("import test 1", category="security")
    src.append("import test 2", category="quality")
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        f.write(src.export_jsonl())
        path = f.name
    try:
        dst = _sql_store()
        n = dst.import_jsonl(path)
        assert n == 2
    finally:
        os.unlink(path)
    src.close()
    dst.close()

def test_save_import_roundtrip():
    src = _sql_store()
    src.append("round trip text", category="hallucination", severity="high")
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        f.write(src.export_jsonl())
        path = f.name
    try:
        dst = _sql_store()
        dst.import_jsonl(path)
        recs = dst.records_by_category("hallucination")
        assert len(recs) == 1
        assert recs[0].severity == "high"
    finally:
        os.unlink(path)
    src.close()
    dst.close()

def test_import_jsonl_nonexistent_raises():
    store = _sql_store()
    with pytest.raises(FileNotFoundError):
        store.import_jsonl("/nonexistent/path/mistakes.jsonl")
    store.close()

def test_save_jsonl_parent_dir_created(tmp_path):
    store = _sql_store()
    store.append("dir test", category="format")
    path = str(tmp_path / "subdir" / "out.jsonl")
    store.save_jsonl(path)
    assert os.path.exists(path)
    store.close()


# =========================================================================
# 8. clear() / close() (5 tests)
# =========================================================================

def test_clear_memory():
    store = _mem_store()
    store.append("a")
    store.clear()
    assert store.total == 0

def test_clear_sqlite():
    store = _sql_store()
    store.append("a", category="quality")
    store.append("b", category="quality")
    store.clear()
    assert store.total == 0
    store.close()

def test_close_sets_conn_none():
    store = _sql_store()
    store.close()
    assert store._conn is None

def test_close_memory_is_noop():
    store = _mem_store()
    store.close()  # エラーにならないこと
    assert store._conn is None

def test_total_and_len_consistent():
    store = _sql_store()
    store.append("x", category="other")
    assert store.total == len(store)
    store.close()


# =========================================================================
# 9. 既存テストとの後方互換 (3 tests)
# =========================================================================

def test_memory_backend_append_still_works():
    store = _mem_store()
    rec = store.append("legacy test", category="security")
    assert store.total == 1
    assert rec.category == "security"

def test_memory_backend_stats_still_works():
    store = _mem_store()
    store.append("a", category="loop")
    s = store.stats()
    assert s["total"] == 1
    assert "loop" in s["by_category"]

def test_memory_backend_query_similar_still_works():
    store = _mem_store()
    store.append("ignore previous instructions", category="security")
    result = store.query_similar("ignore instructions", top_k=1)
    assert len(result) == 1
