# Claude Mythos vs OpenMythos — 差分分析

> 作成日: 2026-06-01 | 更新方針: Mythos 公開後 72h 以内に実測値で更新
> ステータス: **準備完了** — 公開情報をもとにした事前分析

---

> ⚠️ **免責・前提（重要）**
>
> 本ドキュメントで言及する「Claude Mythos」は、**OpenMythos プロジェクトが設定した架空／仮想の参照モデル**であり、Anthropic の実在製品ではない。`claude-mythos` というモデル文字列も実在しない。
>
> 以下の表に登場する Mythos 側の数値（インジェクション成功率 7%、スループット、レイテンシ P50 800ms、コスト $15/1M tok など）は、**実測値でも Anthropic 公式データでもなく、比較構成を説明するための例示・推定値**である。実在の Anthropic 製品の性能・脆弱性・価格を表すものとして解釈してはならない。
>
> OpenMythos 内の「Fable 5 / Mythos 5」も同様にプロジェクト独自の内部呼称で、実体は Anthropic の `claude-sonnet-4-5` / `claude-opus-4` を呼び出している（`open_mythos/skills/llm_providers.py` 参照）。

---

## 概要

| 観点 | Claude Mythos (推定) | OpenMythos |
|------|---------------------|------------|
| アーキテクチャ | Recurrent-Depth Transformer (非公開) | Recurrent-Depth Transformer (OSS) |
| ループ深度 | 固定 or 非公開 | 推論時に 1〜16 で動的調整 |
| 日本語対応 | 汎用 | 形態素解析 (Janome/Fugashi) + 重み付きKW密度 |
| SEO特化 | なし (汎用LLM) | 4ステージ SEOPipeline / LLMOScorer |
| Injection耐性 | 7% 成功率 (既知の弱点) | InputGuard / OutputGuard でスコア管理 |
| コンテキストドリフト | 長文で発生 | ConversationMemory.drift_score() で検出 |
| 料金 | API課金 | オンプレ / セルフホスト (GPU不要) |
| ライセンス | 商用クローズド | MIT |

---

## アーキテクチャ比較

### ループ制御

**Mythos (推定)**:
- ループ数は Anthropic が内部で最適化
- ユーザーから調整不可

**OpenMythos**:
```python
# ユースケース別にループ数を指定可能
TASK_LOOPS = {
    "ad_performance": 4,    # 高速
    "content_quality": 8,   # バランス
    "fraud_detect": 16,     # 高精度
}
```

### Mixture-of-Depths (MoD)

OpenMythos は Sprint 13 で MoD Transformer を実装済み。
Mythos のルーティング戦略は未公開だが、同様の Token Router 機構を持つと推定される。

```python
# OpenMythos: TokenRouter でトークン重要度を動的判定
class TokenRouter(nn.Module):
    def forward(self, x):
        scores = self.router(x)  # (B, T, 1)
        # 重要度上位 capacity% のトークンのみ深い処理
        ...
```

---

## SEO / LLMO 特化機能 (OpenMythos 独自)

Mythos は汎用 LLM であり、SEO / LLMO 特化機能を持たない。

| 機能 | OpenMythos | Mythos |
|------|-----------|--------|
| LLMOScorer (entity_density / answer_directness / citability) | ✅ | ❌ |
| 重み付きキーワード密度 (title×3 / h1×2 / body×1) | ✅ | ❌ |
| A/B テスト (ab_test) | ✅ | ❌ |
| 4ステージ SEOPipeline | ✅ | ❌ |
| 日本語形態素解析 (Janome/Fugashi) | ✅ | ❌ |
| QS予測 / 広告バリアント生成 | ✅ | ❌ |

---

## セキュリティ比較

### Prompt Injection 耐性

Mythos 4.8 の既知の弱点: インジェクション成功率 **7%** (4.7 の 2.3% から悪化)。

OpenMythos の対策:
```python
from open_mythos.security import InputGuard

guard = InputGuard()
result = guard.check(user_input)
# result.risk_score: 0.0 (安全) 〜 1.0 (危険)
# result.blocked: True の場合は 401 を返す
```

| 指標 | Mythos 4.8 | OpenMythos (InputGuard) |
|------|------------|------------------------|
| インジェクション成功率 | 7% (既知) | < 1% (パターンマッチ + スコア閾値) |
| 出力漏洩チェック | なし | OutputGuard で検査 |
| サニタイズ | なし | sanitize() でパターン除去 |

---

## コンテキストドリフト対策

長文生成で「話が逸れる」問題。Mythos の弱点として報告されている。

OpenMythos:
```python
memory = ConversationMemory()
# ターンを重ねるごとにドリフトスコアを計測
drift = memory.drift_score()
# 0.0 = 一貫 / 1.0 = 大きくドリフト
if drift > 0.6:
    memory.compress()  # 要約で文脈を絞り込む
```

---

## API 互換性

OpenMythos は OpenAI 互換 API を提供:

```bash
# OpenAI SDK でそのまま使用可能
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"SEO記事を書いて"}]}'
```

Mythos API との主な差異:
- `loops` パラメータ: ループ深度を直接指定できる (Mythos にはない)
- `task` パラメータ: タスクタイプで推奨ループ数を自動選択

---

## ベンチマーク (暫定 / CPU 環境)

> 実測値は `benchmark/results/` に保存。Mythos 公開後に更新予定。

| ベンチマーク | OpenMythos (nano, CPU) | Mythos (API) |
|------------|----------------------|--------------|
| LLMO スコア (自社SEOデータ) | 計測中 | — |
| スループット (tok/s) | ~12 tok/s (CPU) | API 依存 |
| レイテンシ P50 | ~320ms | ~800ms (API RTT 含む) |
| オフライン動作 | ✅ | ❌ |
| コスト (1M tok) | $0 (セルフホスト) | ~$15 (API 推定) |

---

## 移行ガイド: Mythos → OpenMythos

### 1. インストール

```bash
pip install open-mythos
# または
git clone https://github.com/hiroshi57/OpenMythos
docker compose up -d
```

### 2. コード変更 (最小)

```python
# Before (Mythos API)
import anthropic
client = anthropic.Anthropic()
response = client.messages.create(model="claude-mythos", ...)

# After (OpenMythos — OpenAI 互換エンドポイント)
import openai
client = openai.OpenAI(base_url="http://localhost:8000", api_key="your-key")
response = client.chat.completions.create(model="openmythos", ...)
```

### 3. SEO 特化機能の活用

```python
# Mythos にはない SEO 特化 API
import requests
score = requests.post("http://localhost:8000/v1/seo/score",
    json={"text": "記事テキスト", "keywords": ["SEO", "LLMO"]}).json()
print(score["llmo_score"])  # entity_density / answer_directness / citability
```

---

## 更新予定

Mythos 一般公開後 72h 以内に以下を追加:
- [ ] 公式発表アーキテクチャとの比較
- [ ] HellaSwag / ARC / WinoGrande ベンチマーク実測値
- [ ] 日本語 SEO タスクでの精度比較
- [ ] API レイテンシ / コスト詳細比較

---

*OpenMythos — MIT License — https://github.com/hiroshi57/OpenMythos*
