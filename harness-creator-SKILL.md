---
name: harness-creator
description: |
  AIエージェントハーネスの設計・構築と、タスク単位の構造化実行（PAI Algorithm）を統合したスキル。
  以下のいずれかの言葉が出たら必ずこのスキルを使うこと（部分一致でも起動すること）。

  【ハーネス設計・構築】
  ハーネスを作りたい、ハーネスを設計したい、ハーネスを構築したい、ハーネスを整備したい、
  ハーネスを評価したい、ハーネスを改善したい、ハーネスエンジニアリング、harness engineering、
  harnesscode、hc init、hc start、hc status、
  エージェントの環境を整えたい、エージェント環境を設計したい、エージェント基盤を作りたい、
  エージェント基盤を整備したい、エージェントの土台を作りたい、AIエージェント環境を構築したい、
  Claude Codeのプロジェクト基盤を設計したい、Codexの環境を整えたい、

  【ファイル・設定関連】
  CLAUDE.mdを整備したい、CLAUDE.mdを作りたい、AGENTS.mdを作りたい、AGENTS.mdを整備したい、
  MEMORY.mdを作りたい、MEMORY.mdを整備したい、AIの長期記憶を整備したい、
  feature_list、feature_listを作りたい、feature_listを設計したい、
  init.shを作りたい、session-handoffを作りたい、session-handoff、
  tech-stack.mdを作りたい、技術スタック定義ファイル、
  dev-logを作りたい、ランタイムログを残したい、実行ログを記録したい、
  evaluator-rubricを作りたい、評価スコアカードを作りたい、
  clean-state-checklistを作りたい、セッション終了チェックリスト、

  【エージェント制御・安定化】
  長時間タスクのエージェントを安定させたい、エージェントを安定稼働させたい、
  エージェントを制御したい、エージェントが暴走する、エージェントが勝手に動く、
  AIが言うことを聞かない、エージェントが越境する、作業範囲を限定したい、
  一機能一セッション、スコープを絞りたい、越境させたくない、
  エージェントが完了と言うのに動かない、テストが通らないのに完了と言う、
  エージェントが途中でやめる、止まってしまう、進まない、
  エージェントに仕事をやりきらせたい、放置して動かしたい、無人開発したい、
  長時間タスク、夜間バッチ、自律実行、無人実行、

  【記憶・引き継ぎ】
  セッション間の引き継ぎを整備したい、セッション引き継ぎ、
  記憶がリセットされる、前のセッションを覚えていない、セッションが途切れる、
  エージェントに記憶させたい、記憶設計、AIを育てたい、
  セッション管理をしたい、ライフサイクルを設計したい、

  【マルチエージェント・チーム】
  マルチエージェント構成を設計したい、エージェントチームの土台を作りたい、
  フィードバックループを作りたい、コンテキスト管理を改善したい、
  Orchestrator、Initializer、初期化エージェント、5役エージェント、6役エージェント、
  OpenCode、opencode、

  【検証・品質・可観測性】
  品質ゲートを設定したい、検証ループを作りたい、
  証拠を残したい、エビデンスを記録したい、
  可観測性、観察可能にしたい、エージェントの動きを把握したい、
  エージェントが何をしたか追跡したい、

  【固有名詞・略語】
  Hermes、hermes-agent、5サブシステム、gotchas、
  プログレッシブ開示、progressive disclosure、

  【PAIアルゴリズム関連】
  PAIアルゴリズム、PAI algorithm、paiを使って、アルゴリズムで考えて、7フェーズで、
  科学的に進めて、段階的に検証して、OBSERVE、THINK、PLAN、BUILD、EXECUTE、VERIFY、LEARN、
  ISC、理想状態基準、成功基準を定義して、完了条件を決めて、達成基準を設定して、
  ユーフォリックサプライズ、euphoric surprise、きちんと検証しながら進めたい、
  品質を担保して進めたい、失敗できないタスク、重要な案件、本番に直結する作業、
  MINIMAL、NATIVE、ALGORITHM、E1、E2、E3、E4、E5、手戻りなく進めたい、確実に成功させたい、
  品質を保証して進めたい、道筋を立てて進めたい、開発ループを作りたい、自動開発ループ
---

# Harness Creator + PAI Algorithm 統合スキル

**参照元：**
- Daniel Miessler / Personal_AI_Infrastructure（PAIアルゴリズム）
- yzddp / harnesscode（CLIフレームワーク・Initializerエージェント）
- walkinglabs / learn-harness-engineering（セッションライフサイクル・可観測性・評価）

---

## まずここを読む：3スキルの使い分け

```
やりたいこと                     → 使うスキル
──────────────────────────────────────────────
エージェント環境を設計する         → このスキル（ハーネス設計モード）
任意タスクの品質を保証する         → このスキル（ALGORITHMモード）
コードを書く案件を実行する         → project-management
スキルを作成・改善する             → skill-creator-jp
```

このスキルが担う2つのモード：

| モード | 使う場面 |
|--------|---------|
| **ハーネス設計** | AGENTS.md・MEMORY.md・フック・エージェント構成を新規で作る・直す |
| **ALGORITHM** | 任意のタスクをISC（完了基準）を定義してから7フェーズで実行する |

---

## ─── ハーネスとは何か：根本原則 ───

モデルがどれだけ優秀であっても、環境が整っていなければ実際の開発タスクでは失敗する。Anthropicの検証では、同じモデル・同じ指示で「ハーネスなし」と「ハーネスあり」を比べると、出力の質に質的な差が生まれる。プロンプトを改善する前に、モデルが動く「環境そのもの」を設計することが先決だ。

```
THE HARNESS PATTERN

あなた → タスクを与える → エージェントがハーネスファイルを読む → エージェントが実行する
                                                                      ↓
                                                         ハーネスがすべてのステップを制御する:
                                                          ↓
                                  Instructions: 何を、どの順で、何を読んでから始めるか
                                  Scope:        一度に一機能のみ、越境禁止
                                  State:        進捗ログ、機能リスト、gitの履歴
                                  Verification: テスト・lint・型チェック
                                  Lifecycle:    セッション開始・終了の標準手順
                                          ↓
                                 検証が通るまでエージェントは止まらない
```

---

## ─── パート1：ALGORITHMモード ───

PAI Algorithm v6.3.0 の思想をこのスキルの実行エンジンとして統合する。
ハーネス設計セッション内でも、単独のタスクでも使える。

### まずモードを判断する

| モード | 条件 | 動作 |
|--------|-----|------|
| **MINIMAL** | 1ステップで完結・検証不要 | 直接回答。7フェーズ省略 |
| **NATIVE** | 2〜3ステップ・部分確認で十分 | 簡易計画のみ。ISCは省略可 |
| **ALGORITHM** | 複数ステップ・失敗リスクあり・品質保証が必要 | 以下の7フェーズを全実行 |

| 努力レベル | ISC数 | 目安時間 |
|-----------|------|---------|
| E1 | 1〜3個 | 数分 |
| E2 | 4〜8個 | 数十分 |
| E3 | 9〜15個 | 1〜2時間 |
| E4 | 16〜25個 | 半日〜（複数エージェント推奨） |
| E5 | 26個以上 | 数日〜（フルチーム構成） |

---

### PHASE 1 ─ OBSERVE（観察）

タスクを正確に把握する。推測で動かない。

```
リバースエンジニアリング：
  依頼されたこと：[表面上の要求]
  背景にある意図：[なぜそれが必要か]
  避けるべきこと：[やってほしくないこと]
```

ISC（理想状態基準）をここで必ず作る。完了したと言えるための条件を二値で判定できる形で書く。

```
ISC一覧（例：SEOレポートタスク）
  [ ] 対象ページの現在順位が数値で記録されている
  [ ] 前月比の変化が具体的な数値で示されている
  [ ] 上位3件の競合URLが特定されている
  [ ] 改善施策が3つ以上、優先度付きで提示されている
  [ ] レポートが1000字以内に収まっている
```

ISCはproject-managementのPhase0「成功判定」と同じ概念。

### PHASE 2 ─ THINK（思考）

実行前に思考ツールを選ぶ。使わないツールは「なぜ使わないか」を必ず明示する。

```
思考ツール評価：
  第一原理思考：[使う／使わない ─ 理由]
  逆張り検証  ：[使う／使わない ─ 理由]
  多角的視点  ：[使う／使わない ─ 理由]
  根本原因分析：[使う／使わない ─ 理由]
  リスク先読み：[使う／使わない ─ 理由]
```

エージェント選択（E3以上で明示）：

| エージェント | 役割 |
|------------|-----|
| アーキテクト | 設計・構造の判断 |
| エンジニア | 実装・コード |
| QAテスター | 品質・網羅性確認 |
| リサーチャー | 情報収集・調査 |
| マーケター | 訴求・ユーザー目線 |
| レッドチーム | 弱点・リスク発見 |

### PHASE 3 ─ PLAN（計画）

```
実行計画：
  STEP 1: [具体的な作業] → 対応ISC：[番号]
  STEP 2: [具体的な作業] → 対応ISC：[番号]
  依存：STEP 2 は STEP 1 の完了後に開始
```

### PHASE 4 ─ BUILD（構築）

計画から外れたら止まってOBSERVEに戻る。途中で新しいISCが発見された場合は追加する（削除しない）。

### PHASE 5 ─ EXECUTE（実行）

成果物を実際に動かす・提出する・適用する。

### PHASE 6 ─ VERIFY（検証）

「だいたい動く」は完了ではない。テストが通過し、lintが通り、型チェックが通ることが完了の証拠。

```
ISC検証結果：
  [✓] 対象ページの現在順位が数値で記録されている ← 証拠：〇〇
  [✗] 上位3件の競合URLが特定されている ← 未完了 → BUILDに戻る
```

### PHASE 7 ─ LEARN（学習）

```
学習記録：
  うまくいったこと：[再現性のある成功パターン]
  失敗したこと：[何を・なぜ間違えたか]
  次回への引き継ぎ：[次セッションで使える情報]
```

---

### ALGORITHM出力テンプレート

```
ALGORITHM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
タスク：[8語以内の説明]
モード：ALGORITHM ／ 努力レベル：[E1〜E5]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OBSERVE   依頼の核心：... ／ 意図：... ／ 避けること：...
ISC       [ ] ... ／ [ ] ... ／ [ ] ...
THINK     思考ツール：... ／ エージェント：...
PLAN      STEP 1：... → STEP 2：...
BUILD     [成果物]
EXECUTE   [実行結果]
VERIFY    [ISC番号] ✓/✗ ← 証拠：...
LEARN     [記録事項]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
完了：[完了ISC数]/[全ISC数]
```

---

## ─── パート2：ハーネス設計モード ───

### ハーネスの5サブシステム

| サブシステム | 役割 | 代表ファイル |
|------------|------|------------|
| **Instructions（指示）** | エージェントへの行動規範・ルーティング。巨大な1ファイルではなく段階的に読み込む構造にする | AGENTS.md、CLAUDE.md |
| **State（状態）** | セッションをまたぐ作業状態の引き継ぎ。ディスクに保存し次のセッションが即座に再開できる | feature_list.json、claude-progress.md、session-handoff.md |
| **Verification（検証）** | テスト通過が完了の唯一の証拠 | test_report.json、init.sh |
| **Scope（範囲）** | 一度に一機能のみ。越境禁止 | 完了定義、feature_list.json |
| **Lifecycle（生命周期）** | セッション開始・終了の標準手順。終了時は次回が即再開できる状態に整える | init.sh、clean-state-checklist.md |

---

### ステップ1: インタビューとアセスメント

新規プロジェクトの場合（5問）：

```
Q1. プロジェクトの性質と技術スタックは？
Q2. 使用するAIエージェントツールは？（Claude Code / Codex / OpenCode など）
Q3. タスク規模は？（単一セッション / 複数セッション / 長期プロジェクト）
Q4. チーム利用か個人利用か？
Q5. Human-in-the-Loopが必要か？
```

既存ハーネスの評価（5サブシステムを1〜5点でスコアリング）：
5点＝模範的、3点＝基本は押さえているが不完全、1点＝存在しないか有害。
最低スコアのサブシステムから改善する。

---

### ステップ2: 構成選択

| パターン | 条件 | 生成する構成 |
|---------|------|------------|
| **Minimal** | 個人・単一セッション | AGENTS.md + init.sh + feature_list.json |
| **Standard** | チーム or 複数セッション | Minimal + .claude/rules/ + MEMORY.md + USER.md + claude-progress.md + session-handoff.md |
| **Full** | 長期・マルチエージェント・無人実行 | Standard + 6役エージェント定義 + evaluator-rubric.md + clean-state-checklist.md + test/review report + dev-log.txt |

---

### ステップ3: ファイル構造

Standard 構成：

```
project/
├── AGENTS.md              # メインルーティング（50〜100行以内）
├── .claude/
│   └── rules/
├── docs/
│   ├── ARCHITECTURE.md
│   └── PRODUCT.md
├── input/
│   └── prd/
│       └── tech-stack.md  # 技術スタック定義（必須）
├── context/
│   ├── feature_list.json
│   ├── claude-progress.md # セッションごとの作業記録
│   ├── session-handoff.md
│   ├── MEMORY.md          # エージェント作業記憶（上限2,000文字）
│   └── USER.md            # ユーザー期待値記憶（上限1,000文字）
└── init.sh
```

Full 構成（Standard に追加）：

```
├── agents/
│   ├── orchestrator.md
│   ├── initializer.md     # 初期化専門エージェント
│   ├── coder.md
│   ├── tester.md
│   ├── fixer.md
│   └── reviewer.md
├── .harnesscode/
│   ├── feature_list.json
│   ├── test_report.json
│   ├── review_report.json
│   └── missing_info.json
├── evaluator-rubric.md    # エージェント出力品質スコアカード
├── clean-state-checklist.md  # セッション終了チェックリスト
└── dev-log.txt            # ランタイムログ（可観測性の核心）
```

---

### ステップ4: 各ファイルの内容生成

**AGENTS.md の生成指針（プログレッシブ開示の原則）：**

巨大な1ファイルにすべてを書いてはいけない。最初に「地図」を渡し、詳細は必要に応じて読み込む構造にする。エージェントは読んでいないものは存在しないのと同じに扱う。

```markdown
# AGENTS.md
[プロジェクトを一文で説明]

## 起動時の手順
1. このファイルを読む
2. docs/ARCHITECTURE.md を読む
3. ./init.sh を実行する
4. context/feature_list.json を読む
5. context/claude-progress.md を読む（前回の作業状況を把握する）

## 作業ルール
- 一度に実装する機能は一つのみ
- 「完了」と宣言する前に必ず検証コマンドを実行する
- セッション終了前に claude-progress.md と feature_list.json を更新する
- 詳細な技術スタックは input/prd/tech-stack.md を読むこと

## 完了の定義（ISC形式）
- [ ] 実装が完成している
- [ ] 検証コマンドが通過した
- [ ] エビデンスを feature_list.json に記録した
- [ ] claude-progress.md に今回の作業内容を追記した
```

**tech-stack.md の生成指針：**

技術スタックを定義する必須ファイル。Initializerエージェントがこれを読んで初期ファイル群を生成する。

```markdown
# tech-stack.md
## バックエンド
- 言語: [例: TypeScript / Node.js 22]
- フレームワーク: [例: Express]

## フロントエンド
- [例: React 19 + Vite]

## テスト・検証コマンド
- インストール: npm install
- 検証: npm run check && npm test
- 起動: npm run dev
```

**dev-log.txt の役割：**

エージェントが実際に何をしたかを追跡するランタイムログ。可観測性の核心。これがないと問題の特定が困難になる。

```
[2025-05-18 10:03] ORCHESTRATOR: F003 の実装を Coder に委譲
[2025-05-18 10:08] CODER: src/api/users.ts を実装完了
[2025-05-18 10:09] TESTER: ユニットテスト 全12件 通過
[2025-05-18 10:10] ORCHESTRATOR: F003 完了。F004 へ移行
```

**feature_list.json の生成指針：**

```json
{
  "features": [
    {
      "id": "F001",
      "name": "機能名",
      "description": "何ができるか一文で",
      "dependencies": [],
      "status": "done",
      "evidence": "テスト通過・手動確認 2025-05-01"
    },
    {
      "id": "F990",
      "name": "統合テスト実行",
      "description": "全機能の多層検証",
      "dependencies": [],
      "status": "pending",
      "evidence": ""
    }
  ]
}
```

statusは `pending` / `in-progress` / `done` の3種のみ。

**init.sh の生成指針：**

```bash
#!/bin/bash
set -e
echo "=== 依存パッケージのインストール ==="
npm install
echo "=== 型チェック ==="
npm run check
echo "=== テスト実行 ==="
npm test
echo "=== ビルド ==="
npm run build
echo "=== 検証完了 ==="
```

---

### ステップ5: エージェントセッションライフサイクル（16ステップ）

ハーネスが制御するのは「何を書くか」ではなく「いつ・どこで・どのように書くか」だ。セッションは自由に始めさせない。以下の構造を必ず守らせる。

```
START（開始）
1. AGENTS.md / CLAUDE.md を読む
2. init.sh を実行する（インストール・検証・ヘルスチェック）
3. claude-progress.md を読む（前回の作業状況）
4. feature_list.json を読む（完了済み・未着手の機能一覧）
5. git log を確認する（直近の変更内容）

SELECT（選択）
6. 未完了の機能を1つだけ選ぶ
7. その機能のみに作業する

EXECUTE（実行）
8. 機能を実装する
9. 検証を実行する（テスト・lint・型チェック）
10. 検証失敗 → 修正して再実行
11. 検証通過 → エビデンスを記録する

WRAP UP（後処理）
12. claude-progress.md を更新する
13. feature_list.json を更新する
14. まだ未完了・未検証の部分を明記する
15. コミットする（次のセッションが安全に再開できる状態のみ）
16. 次のセッション向けの再開パスを残す
```

---

### ステップ6: 記憶設計（Standard・Full構成のみ）

Hermes Agent方式の記憶分離。Two-step save invariant（2段階保存の原則）：
エントリは必ず本文に書いた後、インデックスに1行ポインタを追加する。

**claude-progress.md（セッションごとの作業記録）：**

```markdown
# claude-progress.md

## 2025-05-18 セッション記録
- 実施した作業: F003（ユーザー認証API）の実装
- 検証結果: ユニットテスト全12件通過、lint通過
- 次のセッションで着手すること: F004（セッション管理）
- 注意事項: src/auth/token.ts の型定義が複雑なため要注意
```

**MEMORY.md（上限2,000文字）：**

```markdown
# MEMORY.md

## 環境
- OS: {種別}
- 言語バージョン: {例: Node.js 22}
- 起動コマンド: `bash init.sh`

## 学び（AIが実際にやらかしたミスを記録する）
- {日付} {ミスの内容} → {対策}
```

ALGORITHMモードのPHASE 7 LEARN で得た内容はここに書き込む。

**USER.md（上限1,000文字）：**

```markdown
# USER.md

## 好み
- 回答は箇条書きより文章で
- エラーは原因・影響・対策の3点セットで報告

## 期待値
- 速度より品質を重視
```

---

### ステップ7: 6役エージェント定義（Full構成のみ）

**Orchestrator（次の一手を決める司令塔）：**

```
優先度 | 条件                                              | 次のエージェント
1      | 初期化ファイルが不足（tech-stack.md がない等）     | INITIALIZER
2      | missing_info.json に human_action が pending       | PAUSE_FOR_HUMAN
3      | test_report の overall=fail（コードバグ）           | FIXER [module]
4      | review_report の overall=fail                      | FIXER all
5      | status=pending の機能がある（F990, F991除く）       | CODER [module]
6      | F990 が pending                                    | TESTER
7      | F991 が pending                                    | REVIEWER
8      | 全完了 + テスト通過 + レビュー通過                   | PROJECT COMPLETE
```

出力形式：`--- ORCHESTRATOR NEXT: [AGENT] [args] ---`

**Initializer（環境初期化専門エージェント）：**

tech-stack.md を読み、その内容に合わせた初期ファイル群を生成する。feature_list.json の初期バージョンを作成し、init.sh を技術スタックに合わせて書き換える。完了後Orchestratorへ戻る。

**Coder：** セッション開始時に `pwd` → `claude-progress.md` → `git log --oneline -10` → `init.sh` の順で確認してから実装に入る。

**Tester：** 静的解析 → ユニットテスト → コンパイルチェックの3層を独立して実行し、`.harnesscode/test_report.json` に出力する。

**Fixer：** `review_report.json` または `test_report.json` の `status=pending` な問題を読んで修正し、`status: "fixed"` に更新する。

**Reviewer：** `input/techspec/` の仕様ファイルを動的に読み込み、準拠チェックを実行して `.harnesscode/review_report.json` に出力する。

---

### ステップ8: 評価スコアカードとセッション終了チェックリスト（Full構成のみ）

**evaluator-rubric.md（エージェント出力品質スコアカード）：**

```markdown
# evaluator-rubric.md

## 採点項目（各5点満点）
- [ ] 指示された機能のみを実装した（越境なし）
- [ ] テストが存在し、全件通過した
- [ ] 型チェックを通過した
- [ ] feature_list.json を更新した
- [ ] claude-progress.md に今回の作業を記録した
- [ ] コミットメッセージが意図を正確に説明している
- [ ] 未完了の部分が明記されている

## 合格ライン
30点以上（85%以上）で次の機能へ進む。それ未満の場合はFixer に戻す。
```

**clean-state-checklist.md（セッション終了チェックリスト）：**

```markdown
# clean-state-checklist.md

セッション終了前に必ずこのチェックリストを実行すること。

- [ ] 実装した機能のテストが全件通過している
- [ ] lint と型チェックが通過している
- [ ] feature_list.json に今回の変更が反映されている
- [ ] claude-progress.md に今回の作業内容が記録されている
- [ ] 未完了・未検証の部分が明記されている
- [ ] コミットが完了している（テストが通らない状態でのコミット禁止）
- [ ] 次のセッションが迷わず再開できる状態か確認した
```

---

### ステップ9: Human-in-the-Loop（必要な場合のみ）

```json
{
  "missing_items": [
    {
      "id": "M001",
      "desc": "本番DBのホスト名を教えてください",
      "action_type": "human_action",
      "status": "pending",
      "user_input": "",
      "blocks_features": ["F003"]
    }
  ]
}
```

Orchestratorは `action_type=human_action` かつ `status=pending` を検出すると `PAUSE_FOR_HUMAN` を出力して停止する。

---

### ステップ10: Gotchas（落とし穴17選）

1. メモリインデックスの上限はサイレントに発動する — 超えたエントリは無音で消える
2. 優先順位は直感と逆 — ローカル設定がプロジェクト設定より優先される
3. 抽出タイミングに競合窓がある — 次のターンが始まる前に抽出が完了しない場合がある
4. 推論できる内容はメモリに保存しない — リポジトリから読める情報は記録不要
5. 並行分類はツール単位ではなく呼び出し単位 — 同じツールでも入力で安全性が変わる
6. 権限評価は副作用を持つ — キャッシュするな
7. 非同期作業の多くは「pending」状態をスキップする
8. Forkの子はForkを呼び出さない — 再帰Forkはコンテキストコストが指数関数的に膨張する
9. コンテキストビルダーはメモ化されるが手動無効化が必要
10. Hookの信頼はall-or-nothing — 1つが信頼されなければ全Hookがスキップされる
11. Evictionには通知が必要 — 2段階eviction（ディスク先行→メモリ後）で対処する
12. Skillリストの予算は厳しい — descriptionの先頭に独自性の高いトリガーワードを前置きする
13. デフォルトのツール権限は「許可」 — センシティブなツールは明示的に `ask` を設定する
14. チームメモリはauto-memoryの上に成り立つ
15. 孤立トピックファイルが蓄積する — 定期的に掃除する
16. 巨大な1ファイルで指示を与えるとエージェントが読み切れない — プログレッシブ開示で設計すること。AGENTS.md は地図として機能させ、詳細は必要なときに別ファイルを参照させる
17. dev-log.txt がないと問題の追跡が不可能になる — エージェントが何をしたかを記録しないと失敗の原因が分からない。可観測性はハーネス設計の必須要件

---

### ステップ11: ベンチマーク（任意）

1. 実際の作業から2〜3件、検証可能なタスクを選ぶ
2. ハーネスなし・ありで同じタスクを実行する
3. 成功・失敗・所要時間・手戻り回数を記録する
4. 改善率を算出し、過剰設計の部分を削る

---

### ステップ12: 生成後の確認と説明

全ファイルを生成した後、以下を必ず伝える：
1. 生成したファイル一覧と各ファイルの役割
2. tech-stack.md をプロジェクトの実際の技術スタックに合わせて記載すること
3. AGENTS.md の技術スタック部分をプロジェクト実情に合わせて修正すること
4. init.sh のコマンドを使用言語に合わせて変更すること
5. AIが同じミスをしたらMEMORY.mdに追記すること（2,000文字超えたら古い行から削除）
6. セッション終了時は必ず clean-state-checklist.md を実行してからコミットすること
7. dev-log.txt の更新を怠ると問題追跡が困難になること

---

## 参照ファイル

- `references/gotchas.md`：落とし穴17選の詳細版
- `references/memory-persistence-pattern.md`：記憶設計の実装パターン
- `references/context-engineering-pattern.md`：コンテキスト予算管理
- `references/multi-agent-pattern.md`：Coordinator / Fork / Swarm パターン
- `references/progressive-disclosure.md`：プログレッシブ開示の設計パターン
- `references/observability.md`：可観測性の実装パターン
- `templates/`：AGENTS.md・feature_list.json・init.sh・claude-progress.md・evaluator-rubric.md・clean-state-checklist.md のテンプレート群
