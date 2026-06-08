# Contributing to OpenMythos

OpenMythosへの貢献を歓迎します。このガイドに従って開発を進めてください。

---

## 開発環境のセットアップ

```bash
# 1. リポジトリをクローン
git clone <repo-url> OpenMythos
cd OpenMythos

# 2. 仮想環境を作成
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate       # Windows

# 3. 依存関係インストール
pip install -r requirements.txt
pip install fastapi uvicorn pytest

# 4. 環境変数設定
cp .env.example .env
# .env を編集して必要な値を設定

# 5. テストが全て通ることを確認
python -m pytest tests/ -q
```

---

## ブランチ運用

| ブランチ | 用途 |
|---------|------|
| `master` | 本番リリース済みコード |
| `feature/<topic>` | 新機能開発 |
| `fix/<issue>` | バグ修正 |

```bash
# 作業ブランチを作成
git switch -c feature/your-feature-name

# 変更後はテストを実行してから
python -m pytest tests/ -q

# コミット
git add <files>
git commit -m "feat: 機能の説明"

# プッシュしてPRを作成
git push -u origin feature/your-feature-name
```

---

## コミットメッセージ規約

| プレフィックス | 用途 |
|-------------|------|
| `feat:` | 新機能 |
| `fix:` | バグ修正 |
| `docs:` | ドキュメント更新 |
| `refactor:` | リファクタリング |
| `test:` | テスト追加・修正 |
| `chore:` | ビルド・設定変更 |

例: `feat: /v1/assistants にアシスタント検索エンドポイントを追加`

---

## APIエンドポイントを追加する場合

1. **スキルモジュール** (`open_mythos/skills/`) にクラスを追加
2. **APIエンドポイント** (`serve/api.py`) にルートを追加
3. **テスト** (`tests/test_<feature>.py`) を作成し全テストがPASSすることを確認

```python
# serve/api.py へのエンドポイント追加例
from pydantic import BaseModel

class _MyRequest(BaseModel):
    text: str
    options: dict = {}

@app.post("/v1/my-feature", tags=["my-feature"], dependencies=[Depends(verify_api_key)])
def my_feature(req: _MyRequest):
    """機能の説明。"""
    result = do_something(req.text)
    return {"result": result}
```

---

## テスト規約

- テストファイル: `tests/test_<module>.py`
- クラス名: `TestXxx`
- メソッド名: `test_<what>_<condition>`

```python
class TestMyFeature:
    def test_returns_expected_result(self):
        result = my_function("input")
        assert result["key"] == "expected"

    def test_api_endpoint_200(self, client):
        r = client.post("/v1/my-feature",
                        json={"text": "test"},
                        headers={"Authorization": "Bearer dev"})
        assert r.status_code == 200
```

---

## コードスタイル

```bash
# フォーマット確認
black --check open_mythos/ serve/ tests/

# Lint確認
ruff check open_mythos/ serve/ tests/
```

---

## PR (プルリクエスト) のチェックリスト

- [ ] テストが全てPASS (`python -m pytest tests/ -q`)
- [ ] 新機能にはテストを追加した
- [ ] READMEに新機能を記載した (必要な場合)
- [ ] コミットメッセージが規約に沿っている
- [ ] `.env` や秘密情報をコミットしていない

---

## 質問・不具合報告

GitHubのIssueを作成してください。以下の情報を含めると解決が早くなります:

- OS・Pythonバージョン
- 再現手順
- 期待する動作と実際の動作
- エラーメッセージ (あれば)
