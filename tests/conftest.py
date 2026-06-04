"""
tests/conftest.py — pytest 共通フィクスチャ

レート制限リセット:
  各テストモジュール開始前にグローバルレートリミッターをリセットする。
  テスト間の干渉（60 RPM 上限の枯渇）を防ぐ。
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
