"""
tests/conftest.py — pytest 共通フィクスチャ

レート制限リセット:
  各テストモジュール開始前にグローバルレートリミッターをリセットする。
  テスト間の干渉（60 RPM 上限の枯渇）を防ぐ。

共有 TestClient:
  新規テストは各モジュールで TestClient(app) を構築せず
  session スコープの `api_client` fixture を利用すること。
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="module")
def reset_rate_limiter():
    """モジュール単位でレートリミッターをリセットする。"""
    try:
        from serve.auth import _rate_limiter
        _rate_limiter.reset_all()
    except ImportError:
        pass
    yield
    # teardown: 次のモジュールのために再リセット
    try:
        from serve.auth import _rate_limiter
        _rate_limiter.reset_all()
    except ImportError:
        pass


@pytest.fixture(scope="session")
def api_client():
    """serve.api アプリ全体の共有 TestClient。"""
    from fastapi.testclient import TestClient
    from serve.api import app

    with TestClient(app) as client:
        yield client
