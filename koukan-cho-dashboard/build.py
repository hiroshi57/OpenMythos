# -*- coding: utf-8 -*-
"""
官公庁 入札・落札データ分析エンジン
====================================
Data-Walkers「政府調達・入札落札データ」APIの実データ(1000件)を取り込み、
全案件を4観点でスコアリングして dashboard 用 data.json を生成する。

観点:
  1. 業界業種   : project_name / ministry からルールベースで業種分類
  2. 案件概要   : 業種 + 発注元 + 規模帯から要約文を自動生成
  3. DI親和性   : デジタルアイデンティティ(Web/デジタルマーケ/DX/広告)の受注適性 0-100
  4. 落札可能性 : DIが入札した場合の推定勝率 % (親和性×規模適合×競合強度)

APIキーはこのビルド時のみ使用し、生成物(data.json)には含めない。
使い方: python build.py  (raw.json が無ければ API から取得)
"""
import json, os, re, math, urllib.request
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
# APIキーは環境変数 DW_API_KEY から読む(コミットしない)。未設定なら既存 raw.json を再利用。
API_KEY = os.environ.get("DW_API_KEY", "")
DATASET_ID = "12dcea9f-bdc1-469f-a54f-4008dab666ee"
BASE = "https://data-walkers.com/api/v1"


# --------------------------------------------------------------------------
# 0. データ取得
# --------------------------------------------------------------------------
def load_raw():
    p = os.path.join(HERE, "raw.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    if not API_KEY:
        raise SystemExit("raw.json が無く DW_API_KEY 未設定です。"
                         "環境変数 DW_API_KEY にAPIキーを設定して再実行してください。")
    hdr = {"X-API-Key": API_KEY}
    rows = json.load(urllib.request.urlopen(
        urllib.request.Request(f"{BASE}/datasets/{DATASET_ID}/data?limit=1000", headers=hdr)))["data"]
    meta = json.load(urllib.request.urlopen(
        urllib.request.Request(f"{BASE}/datasets/{DATASET_ID}")))["data"]
    obj = {"meta": meta, "rows": rows}
    json.dump(obj, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    return obj


# --------------------------------------------------------------------------
# 1. 業界業種の分類  (優先度順にマッチ)
# --------------------------------------------------------------------------
# (カテゴリ, 表示色キー, キーワード群)  上から順に評価し最初にヒットした業種を採用
INDUSTRY_RULES = [
    ("広報・広告・マーケティング", "mkt",
     ["広報", "広告", "ＰＲ", "ﾌﾟﾛﾓｰｼｮﾝ", "プロモーション", "情報発信", "普及啓発", "啓発",
      "動画", "映像", "ＳＮＳ", "ＳＮＳ", "キャンペーン", "ブランディング", "マーケティング",
      "周知", "特設サイト", "情報提供業務", "広報誌", "パンフレット制作"]),
    ("Web・デジタル", "web",
     ["ホームページ", "ウェブ", "ウェブサイト", "Ｗｅｂ", "ＷＥＢ", "ポータルサイト", "サイト構築",
      "サイト運用", "サイト改修", "オンライン", "電子申請", "ＵＩ", "ＵＸ", "アクセシビリティ",
      "デジタル化", "ＤＸ", "オープンデータ", "アプリ", "アプリケーション開発"]),
    ("システム・IT", "it",
     ["システム", "ソフトウェア", "ネットワーク", "サーバ", "クラウド", "データベース", "プログラム",
      "情報基盤", "情報処理", "ＩＴ", "ＡＩ", "ＲＰＡ", "ＬＡＮ", "電子計算機", "端末", "ＰＣ",
      "コンピュータ", "デジタル人材", "セキュリティ", "認証基盤", "運用管理及び保守", "保守等業務"]),
    ("調査・コンサル・研究", "res",
     ["調査", "分析", "検討", "研究", "コンサル", "診断", "アンケート", "統計", "実態把握",
      "評価業務", "検証", "モニタリング", "推計", "設計業務委託", "計画策定", "支援業務"]),
    ("印刷・製本", "print",
     ["印刷", "製本", "封入", "封緘", "丁合", "発送業務"]),
    ("人材・研修・運営", "hr",
     ["派遣", "研修", "講師", "運営業務", "運営支援", "相談", "窓口", "人材", "セミナー",
      "説明会", "受付業務", "事務局"]),
    ("電気・ガス・エネルギー", "energy",
     ["電気", "ガス", "電力", "エネルギー", "燃料", "灯油", "重油", "ＬＰ", "太陽光"]),
    ("建設・土木・設備工事", "const",
     ["工事", "建設", "土木", "改修工事", "修繕", "舗装", "建築", "解体", "設備工事",
      "電気工事", "空調", "外構", "撤去"]),
    ("保守・管理・警備・清掃", "maint",
     ["保守", "管理業務", "点検", "清掃", "警備", "保全", "メンテナンス", "運転管理",
      "維持管理", "設備管理", "植栽", "除草"]),
    ("医療・環境・検査", "env",
     ["医療", "検査", "環境", "廃棄物", "測定", "分析業務", "衛生", "医薬", "試験"]),
    ("物品・機器・車両調達", "goods",
     ["購入", "調達", "供給", "借入", "賃貸借", "リース", "機器", "装置", "車両", "図書",
      "消耗品", "備品", "用紙", "什器", "食料", "被服"]),
]
INDUSTRY_ORDER = [r[0] for r in INDUSTRY_RULES] + ["その他"]
INDUSTRY_COLORKEY = {r[0]: r[1] for r in INDUSTRY_RULES}
INDUSTRY_COLORKEY["その他"] = "other"


def classify_industry(name, ministry):
    t = (name or "") + " " + (ministry or "")
    for cat, _key, kws in INDUSTRY_RULES:
        for kw in kws:
            if kw in t:
                return cat
    return "その他"


# --------------------------------------------------------------------------
# 2. DI(デジタルアイデンティティ)親和性 0-100
#    DI事業領域: SEO / 運用型広告 / Web制作 / デジタルマーケ / DX支援 / データ分析 / SNS
# --------------------------------------------------------------------------
INDUSTRY_AFFINITY_BASE = {
    "広報・広告・マーケティング": 88,
    "Web・デジタル": 82,
    "調査・コンサル・研究": 64,
    "システム・IT": 52,
    "印刷・製本": 34,
    "人材・研修・運営": 30,
    "医療・環境・検査": 12,
    "保守・管理・警備・清掃": 12,
    "物品・機器・車両調達": 6,
    "電気・ガス・エネルギー": 3,
    "建設・土木・設備工事": 2,
    "その他": 20,
}
# 案件名に含まれると DI 適性が上がるキーワード(+加点)
AFFINITY_BOOST = {
    "ＳＥＯ": 12, "検索": 6, "広告": 10, "運用型": 10, "リスティング": 12,
    "ＳＮＳ": 10, "動画": 8, "映像": 6, "コンテンツ": 8, "サイト": 8, "ホームページ": 8,
    "ウェブ": 8, "Ｗｅｂ": 8, "ＷＥＢ": 8, "デジタルマーケ": 15, "マーケティング": 10,
    "ブランディング": 8, "ＵＩ": 6, "ＵＸ": 6, "アクセス解析": 12, "データ分析": 10,
    "ＤＸ": 8, "普及啓発": 6, "情報発信": 8, "プロモーション": 10, "ＰＲ": 6, "ＡＩ": 5,
}
# 適性を下げるキーワード(専門/大型/インフラ寄り)
AFFINITY_PENALTY = {
    "工事": -20, "建設": -20, "電気の購入": -30, "ガスの": -25, "燃料": -25,
    "清掃": -15, "警備": -15, "医薬": -15, "車両": -12, "什器": -10, "食料": -12,
}


def score_affinity(name, industry):
    s = INDUSTRY_AFFINITY_BASE.get(industry, 20)
    t = name or ""
    for kw, w in AFFINITY_BOOST.items():
        if kw in t:
            s += w
    for kw, w in AFFINITY_PENALTY.items():
        if kw in t:
            s += w
    return max(0, min(100, s))


# --------------------------------------------------------------------------
# 3. 落札可能性(DIが入札したと仮定した推定勝率 %)
#    = 親和性factor × 金額規模適合 × 競合(現落札企業)強度
# --------------------------------------------------------------------------
# 現落札企業(=競合)のタイプ判定
#   entry = DIがその現職に対して「競争の土台に載れる」度合い 0-1
# --------------------------------------------------------------------------
INCUMBENT_TYPES = [
    ("si", "大手SI・コンサル", 0.30,
     ["ＮＴＴデータ", "ＮＥＣ", "日本電気", "富士通", "日立", "アクセンチュア", "野村総合研究所",
      "大和総研", "三菱総合研究所", "デロイト", "ＰｗＣ", "ＫＰＭＧ", "アビーム", "ＩＢＭ",
      "ＳＣＳＫ", "ＴＩＳ", "伊藤忠テクノ", "日本ユニシス", "ＢＩＰＲＯＧＹ", "ＮＥＣソリューション"]),
    ("ad", "大手広告・PR", 0.78,
     ["電通", "博報堂", "ＡＤＫ", "アサツー", "大広", "読売広告", "東急エージェンシー",
      "ジェイアール東日本企画", "サイバーエージェント"]),
    ("carrier", "通信キャリア", 0.15,
     ["ＮＴＴドコモ", "ＫＤＤＩ", "ソフトバンク", "ＮＴＴコミュニケーション", "ＮＴＴ東日本",
      "ＮＴＴ西日本", "楽天モバイル"]),
    ("const", "建設・設備", 0.05,
     ["前田建設", "西松建設", "大成建設", "鹿島", "清水建設", "大林組", "竹中", "戸田建設",
      "五洋建設", "熊谷組", "ＮＩＰＰＯ", "建設", "工業株式会社", "電気工事", "設備株式会社"]),
    ("org", "団体・独法・大学等", 0.55,
     ["財団", "社団", "協会", "機構", "大学", "学校法人", "独立行政法人", "センター", "組合",
      "協同", "ＪＡ", "公社", "事業団"]),
]


def classify_incumbent(company):
    c = company or ""
    for key, label, entry, kws in INCUMBENT_TYPES:
        for kw in kws:
            if kw in c:
                return key, label, entry
    if any(x in c for x in ("株式会社", "合同会社", "有限会社", "㈱")):
        return "general", "一般企業(中堅・中小)", 1.0
    return "unknown", "その他・不明", 0.7


def scale_factor(amount):
    """案件金額が生む参入障壁(入札参加資格の等級・実績要件)のクリアしやすさ 0-1"""
    if not amount or amount <= 0:
        return 0.7
    a = amount
    if a < 1_000_000:        return 1.0    # 100万未満: 少額(D級)、障壁ほぼ無し
    if a < 10_000_000:       return 1.0    # 100万〜1千万: C〜B級で参加可
    if a < 50_000_000:       return 0.9    # 1千万〜5千万: B〜A級
    if a < 150_000_000:      return 0.75   # 5千万〜1.5億: A級+実績要件
    if a < 300_000_000:      return 0.55   # 1.5億〜3億: 実績要件が本格化
    if a < 1_000_000_000:    return 0.3    # 3億〜10億: 大規模実績・体制が壁
    return 0.15                            # 10億超: 単独受注は非現実的


def grade_hint(amount):
    """金額帯 → 全省庁統一資格(役務の提供等)の目安等級"""
    if not amount or amount <= 0:  return "等級C〜D(少額)"
    a = amount
    if a < 1_000_000:   return "等級D(100万円未満)"
    if a < 3_000_000:   return "等級C(〜300万円目安)"
    if a < 10_000_000:  return "等級B〜C"
    if a < 50_000_000:  return "等級A〜B"
    return "等級A(大型)"


def assess_competition(company, affinity, amount):
    """DI競合度を多面評価して level / score / 理由 / 土台条件 を返す"""
    itype, ilabel, entry = classify_incumbent(company)
    sf = scale_factor(amount)
    # 参入度スコア(0-100, 高いほどDIが競争の土台に載りやすい)
    entry_score = round(entry * sf * 100)

    # ドメイン外(親和性が低い)は業種として提供できないので対象外
    if affinity < 20:
        level = "対象外(領域外)"
        reason = f"現落札は{ilabel}。案件はDIの提供役務(Web/広報/デジタル/調査)の範囲外で入札対象にならない。"
        foothold = ["この業種(物品/工事/インフラ等)はDIの事業領域外。母数から除外して営業効率を管理する。"]
        return {"level": level, "score": min(entry_score, 8), "itype": itype,
                "ilabel": ilabel, "reason": reason, "foothold": foothold}

    # レベル判定
    if entry_score >= 72:   level = "参入容易"
    elif entry_score >= 48: level = "競争可能"
    elif entry_score >= 26: level = "要準備"
    elif entry_score >= 10: level = "参入困難"
    else:                   level = "対象外"

    # 理由(現職タイプ別)
    reason_by_type = {
        "si": f"現落札は{ilabel}。大規模実績・体制・随意継続が壁で単独一次受注は難しい。",
        "ad": f"現落札は{ilabel}。マス統合では不利だが、運用型広告・SEO・SNS・Web等の"
              f"デジタル領域に絞ればDIの土俵で戦える。",
        "carrier": f"現落札は{ilabel}。通信インフラ主体で役務範囲外。",
        "const": f"現落札は{ilabel}。建設業許可が必要な領域で対象外。",
        "org": f"現落札は{ilabel}。随意契約が多いが、公募型・企画競争の案件は狙える。",
        "general": f"現落札は{ilabel}。価格・提案・実績で正面から競争でき、DIの規模で十分戦える。",
        "unknown": f"現落札は{ilabel}。競合の素性が読みにくいため個別に要確認。",
    }
    reason = reason_by_type[itype]

    # 「競争の土台に載る」条件(金額・現職・領域から動的生成)
    foothold = []
    foothold.append(f"入札参加資格: 全省庁統一資格『役務の提供等』{grade_hint(amount)}を取得・更新。")
    if amount and amount >= 10_000_000:
        foothold.append("実績要件: 同種(Web構築/広報/デジタルマーケ/調査)の元請実績を提示できる体制を用意。")
    if amount and amount >= 150_000_000:
        foothold.append("体制: 大型のため専任PM・品質保証体制、または再委託/JVで規模要件を満たす。")
    if affinity >= 40 and amount and amount < 300_000_000:
        foothold.append("入札方式: 企画競争(プロポーザル)が多く、価格でなく提案力・実績で差別化できる。")
    # 現職タイプ別の一手
    tips = {
        "si": "まず大手SIの再委託先・パートナーとして官公庁実績を積み、単独入札の資格要件を満たす。",
        "ad": "デジタル(運用型広告/SEO/SNS運用/効果測定)に領域を絞って提案。DIの強みが最も効く。",
        "org": "公募型プロポーザル・企画競争の公告を定点監視し、随意契約化する前に提案を差し込む。",
        "general": "現職より高い提案品質・デジタル専門性・改善実績を前面に出して相見積り/入札で勝負。",
        "unknown": "過去の同種案件の落札者・価格を調べ、勝ち筋(価格 or 提案)を見極めてから参加判断。",
        "carrier": "役務範囲外。DIが関与するならデジタル施策部分の再委託を狙う程度。",
        "const": "役務範囲外。原則見送り。",
    }
    foothold.append(tips[itype])

    return {"level": level, "score": entry_score, "itype": itype,
            "ilabel": ilabel, "reason": reason, "foothold": foothold}


def score_winnability(affinity, amount, company):
    """DIが入札した場合の推定勝率(%) = 得意度 × 現職への参入度 × 規模障壁クリア度"""
    _itype, _il, entry = classify_incumbent(company)
    p = (affinity / 100.0) * entry * scale_factor(amount)
    return round(min(0.95, p) * 100, 1)


def size_attractiveness(amount):
    """案件規模の魅力度 0-100 (対数スケール: 大きいほど高いが逓減)"""
    if not amount or amount <= 0:
        return 15
    import math as _m
    # 100万=~25, 1000万=~45, 1億=~68, 3億=~80, 10億=~92
    v = (_m.log10(amount) - 5.0) / 4.0 * 100  # 10万=0, 10億=100
    return round(max(5, min(100, v)), 1)


def label4(v, hi, mid, lo):
    if v >= hi:
        return "高"
    if v >= mid:
        return "中"
    if v >= lo:
        return "低"
    return "圏外"


# --------------------------------------------------------------------------
# 4. 案件概要の自動生成
# --------------------------------------------------------------------------
def size_band(a):
    if not a:
        return "金額非公開/小規模"
    if a < 3_000_000:
        return "小規模(〜300万円)"
    if a < 50_000_000:
        return "中規模(300万〜5千万円)"
    if a < 150_000_000:
        return "中大規模(5千万〜1.5億円)"
    if a < 1_000_000_000:
        return "大規模(1.5億〜10億円)"
    return "超大型(10億円〜)"


def make_summary(r, industry, aff, win):
    mini = r.get("ministry") or "官公庁"
    agency = r.get("agency") or ""
    org = f"{mini}{('・' + agency) if agency else ''}"
    band = size_band(r.get("award_amount"))
    fy = ""
    m = re.search(r"(令和[０-９0-9]+年度|Ｒ[０-９0-9]+)", r.get("project_name") or "")
    if m:
        fy = m.group(1) + "の"
    aff_note = "DI中核領域" if aff >= 70 else ("DI周辺領域" if aff >= 40 else "DI対象外寄り")
    win_note = "受注機会あり" if win >= 40 else ("挑戦価値あり" if win >= 20 else "現実的に困難")
    return f"{org}が発注した{fy}「{industry}」案件({band})。落札は{r.get('awarded_company') or '—'}。DI視点では{aff_note}・{win_note}。"


# --------------------------------------------------------------------------
# 5. 予測・学習モジュール (TimesFM / TabFM の思想を踏まえた軽量実装)
#    - TabFM流 : 案件名の文字bigramベクトルで k近傍の類似案件を検索し、
#                類似案件の実績から勝率を予測(文脈内学習の代替)
#    - TimesFM流: 官公庁調達の年度再帰性(令和N年度→N+1年度)を利用した
#                 来年度公告の再帰予測(時系列基盤モデルの代替)
#    ※ 本家モデルは数GBのチェックポイント+GPU前提のためこの環境では未使用。
# --------------------------------------------------------------------------
ZEN2ASC = str.maketrans("０１２３４５６７８９（）　", "0123456789() ")
FY_RE = re.compile(r"令和\s*([0-9]{1,2})\s*年度")


def norm_name(s):
    return re.sub(r"\s+", "", (s or "").translate(ZEN2ASC).lower())


def bigram_vec(s):
    t = norm_name(s)
    c = Counter(t[i:i + 2] for i in range(len(t) - 1))
    norm = math.sqrt(sum(v * v for v in c.values())) or 1.0
    return c, norm


def cosine(v1, n1, v2, n2):
    if len(v1) > len(v2):
        v1, v2 = v2, v1
    dot = sum(w * v2.get(g, 0) for g, w in v1.items())
    return dot / (n1 * n2)


def knn_enrich(enriched, pool, k=8):
    """pool内の各案件に対し全件から類似案件を検索し、予測勝率を付与(TabFM流)"""
    vecs = [bigram_vec(e["project_name"]) for e in enriched]
    for e in pool:
        i = e["_idx"]
        v, nv = vecs[i]
        sims = []
        for j, other in enumerate(enriched):
            if j == i:
                continue
            s = cosine(v, nv, vecs[j][0], vecs[j][1])
            if s > 0.12:
                sims.append((s, other))
        sims.sort(key=lambda x: -x[0])
        top = sims[:k]
        if top:
            wsum = sum(s for s, _ in top)
            nb_win = sum(s * o["winnability"] for s, o in top) / wsum
            conf = wsum / len(top)  # 平均類似度
            # ルールベース勝率と類似案件実績のブレンド(類似度が高いほど近傍を信頼)
            alpha = min(0.45, conf)
            pred = round((1 - alpha) * e["winnability"] + alpha * nb_win, 1)
            e["pred_win"] = pred
            e["pred_conf"] = round(conf * 100)
            e["pred_conf_label"] = "高" if conf >= 0.35 else ("中" if conf >= 0.2 else "低")
            e["similar_cases"] = [
                {"name": o["project_name"], "company": o["company"],
                 "amount": o["amount"], "win": o["winnability"], "sim": round(s * 100)}
                for s, o in top[:3]]
            e["nb_median_amount"] = sorted(o["amount"] for _, o in top)[len(top) // 2]
        else:
            e["pred_win"] = e["winnability"]
            e["pred_conf"] = 0
            e["pred_conf_label"] = "低"
            e["similar_cases"] = []
            e["nb_median_amount"] = e["amount"]


RECUR_KW = ["運用", "保守", "支援", "管理", "提供", "発信", "啓発", "広報", "調査",
            "運営", "監視", "業務委託", "情報"]
ONEOFF_KW = ["構築", "開発", "整備", "導入", "改修", "更新", "移行", "刷新"]


def recurrence_prob(name):
    """年度付き案件が来年度も公告される確率(経験則)"""
    p = 0.70
    if any(kw in name for kw in RECUR_KW):
        p = 0.85
    if any(kw in name for kw in ONEOFF_KW):
        p -= 0.25
    return round(max(0.35, min(0.9, p)), 2)


def forecast_upcoming(pool, top_n=30):
    """令和N年度パターンから来年度の再公告を予測(TimesFM流の年度再帰予測)"""
    items = []
    for e in pool:
        m = FY_RE.search((e["project_name"] or "").translate(ZEN2ASC))
        if not m:
            continue
        fy = int(m.group(1))
        next_fy = fy + 1
        year = 2018 + next_fy  # 令和N年度 = 西暦(2018+N)年度
        rec_p = recurrence_prob(e["project_name"])
        pred_win = e.get("pred_win", e["winnability"])
        # 予想金額 = 前年実績70% + 類似案件中央値30%
        pred_amt = round(0.7 * e["amount"] + 0.3 * e.get("nb_median_amount", e["amount"]))
        expected = round(rec_p * pred_win, 1)  # 公告されて勝つ複合確率(%)
        pred_name = re.sub(FY_RE, f"令和{next_fy}年度",
                           e["project_name"].translate(ZEN2ASC))
        items.append({
            "pred_name": pred_name,
            "base_name": e["project_name"],
            "ministry": e["ministry"], "agency": e["agency"],
            "industry": e["industry"],
            "base_amount": e["amount"], "pred_amount": pred_amt,
            "base_company": e["company"],
            "announce_window": f"{year}年1〜3月頃公告 → {year}年4月前後 契約",
            "recurrence_prob": round(rec_p * 100),
            "pred_win": pred_win,
            "pred_conf_label": e.get("pred_conf_label", "低"),
            "expected": expected,
            "comp_level": e["comp_level"],
            "similar_count": len(e.get("similar_cases", [])),
            "rationale": (f"前年(令和{fy}年度)は{e['company']}が{e['amount']:,}円で落札。"
                          f"類似案件{len(e.get('similar_cases', []))}件の学習で予測勝率{pred_win}%、"
                          f"再公告確率{round(rec_p*100)}%。"),
        })
    items.sort(key=lambda x: -x["expected"])
    return items[:top_n]


# --------------------------------------------------------------------------
# 6. 算出ロジックの説明(UI表示用)
# --------------------------------------------------------------------------
METHODOLOGY = {
    "opportunity": {
        "formula": "狙い目スコア(0-100) = 落札可能性 × 0.5 ＋ 案件規模の魅力 × 0.2 ＋ DI親和性 × 0.3",
        "components": [
            {"name": "落札可能性(重み50%)", "desc": "DIが入札した場合の推定勝率。勝てない案件は規模が大きくても優先度を下げる、最重要因子。"},
            {"name": "案件規模の魅力(重み20%)", "desc": "落札金額の対数スケール(100万円≈25点、1千万円≈45点、1億円≈68点)。売上インパクトを加点するが、勝率より重みを小さくして「勝てる案件」優先を維持。"},
            {"name": "DI親和性(重み30%)", "desc": "DI事業領域(SEO/運用型広告/Web制作/DX/データ分析)への合致度。得意領域は提案品質・利益率が高いため加点。"},
        ],
        "reading": "70以上=最優先で入札検討 / 50-69=積極検討 / 30-49=条件次第 / 30未満=見送り",
    },
    "competition": {
        "formula": "DI競合度(参入度 0-100) = 現職タイプ係数 × 金額障壁係数 × 100",
        "incumbents": [
            {"type": "一般企業(中堅・中小)", "coef": 1.0, "desc": "価格・提案・実績の正面勝負が可能。DIの規模で十分戦える。"},
            {"type": "大手広告・PR(電通/博報堂等)", "coef": 0.78, "desc": "マス統合では不利だが、運用型広告・SEO・SNS等のデジタル領域に絞ればDIの土俵。"},
            {"type": "団体・独法・大学等", "coef": 0.55, "desc": "随意契約が多いが、公募型・企画競争案件は狙える。"},
            {"type": "大手SI・コンサル(NTTデータ/NEC等)", "coef": 0.30, "desc": "大規模実績・体制・随意継続が壁。単独一次受注は難しく再委託から。"},
            {"type": "通信キャリア", "coef": 0.15, "desc": "通信インフラ主体で役務範囲外。"},
            {"type": "建設・設備", "coef": 0.05, "desc": "建設業許可が必要な領域で対象外。"},
        ],
        "scale": [
            {"band": "〜1千万円", "coef": 1.0, "note": "統一資格C〜D級で参加可、障壁ほぼ無し"},
            {"band": "1千万〜5千万円", "coef": 0.9, "note": "B〜A級、同種実績の提示が望ましい"},
            {"band": "5千万〜1.5億円", "coef": 0.75, "note": "A級+実績要件"},
            {"band": "1.5億〜3億円", "coef": 0.55, "note": "実績要件が本格化"},
            {"band": "3億〜10億円", "coef": 0.30, "note": "大規模実績・体制が壁"},
            {"band": "10億円超", "coef": 0.15, "note": "単独受注は非現実的、JV/再委託のみ"},
        ],
        "levels": [
            {"level": "参入容易", "range": "72-100", "meaning": "中小現職×少額〜中規模。統一資格があれば即入札可能", "action": "公告を定点監視し、片っ端から提案"},
            {"level": "競争可能", "range": "48-71", "meaning": "正面から競争できるが資格・実績の準備は必要", "action": "同種実績を整理し企画競争で提案力勝負"},
            {"level": "要準備", "range": "26-47", "meaning": "実績・体制の補強、または現職の隙が必要", "action": "小型案件で官公庁実績を先に積む/再委託参画"},
            {"level": "参入困難", "range": "10-25", "meaning": "大手現職×大型。単独一次受注はほぼ不可", "action": "大手のパートナー(再委託先)として入り込む"},
            {"level": "対象外", "range": "0-9 / 領域外", "meaning": "DIの提供役務の範囲外", "action": "見送り。母数から除外して営業効率を管理"},
        ],
        "foothold_common": [
            "① 全省庁統一資格『役務の提供等』を取得・更新する(等級が入札可能金額帯を決める)",
            "② 同種(Web構築/広報/デジタルマーケ/調査)の元請実績を証憑つきで整理する",
            "③ 少額(〜300万円)案件・見積り合わせで官公庁との取引実績を先に作る",
            "④ 企画競争(プロポーザル)案件を狙う — 価格でなく提案力・専門性で差別化できる",
            "⑤ 大型案件は大手SIの再委託・JVで参画し、次年度の単独入札につなげる",
        ],
    },
    "winnability": {
        "formula": "推定勝率(%) = DI親和性/100 × 現職タイプ係数 × 金額障壁係数 × 100 (上限95%)",
        "desc": "「どれだけ得意か × 現職にどれだけ競り込めるか × 資格・実績の壁をどれだけ越えられるか」の積。3因子のどれかが0に近いと勝率も0に近づく。",
    },
    "prediction": {
        "note": "TimesFM(時系列基盤モデル)・TabFM(表形式基盤モデル)の思想を踏まえた軽量実装。"
                "本家モデルは数GBのチェックポイント+GPU前提のため、本画面では"
                "(1)TabFM流=案件名の文字bigramベクトルによるk近傍(k=8)類似案件検索から勝率を学習予測、"
                "(2)TimesFM流=官公庁調達の年度再帰性(令和N年度→N+1年度)を利用した来年度公告予測、で代替。"
                "APIの時系列が単一日付に偏るため、月次時系列予測でなく年度サイクル予測を採用。",
        "pred_win": "予測勝率 = ルールベース勝率×(1-α) + 類似案件の実績勝率(類似度加重平均)×α。α=平均類似度(上限0.45)。",
        "expected": "勝てる期待値(%) = 再公告確率 × 予測勝率。「来年度その案件が出て、かつDIが勝つ」複合確率。",
        "recurrence": "再公告確率: 年度付き案件は基準70%。運用/保守/支援/広報等の継続性キーワードで85%、構築/開発等の単発キーワードで-25pt(下限35%)。",
    },
}


# --------------------------------------------------------------------------
# メイン処理
# --------------------------------------------------------------------------
def main():
    raw = load_raw()
    rows = raw["rows"]
    meta = raw["meta"]

    enriched = []
    for r in rows:
        name = r.get("project_name") or ""
        mini = r.get("ministry")
        amt = r.get("award_amount") or 0
        comp = r.get("awarded_company") or ""
        industry = classify_industry(name, mini)
        aff = score_affinity(name, industry)
        win = score_winnability(aff, amt, comp)
        comp_assess = assess_competition(comp, aff, amt)
        # 狙い目スコア(0-100) = 透明な加重平均。「勝てて・大きくて・得意」な案件ほど高い。
        #   勝率 50% + 案件規模の魅力 20% + DI親和性 30%
        sz = size_attractiveness(amt)
        c_win = round(win * 0.5, 1)
        c_size = round(sz * 0.2, 1)
        c_aff = round(aff * 0.3, 1)
        opp = round(c_win + c_size + c_aff, 1)
        enriched.append({
            "id": r.get("id"),
            "project_name": name,
            "ministry": mini or "(不明)",
            "agency": r.get("agency") or "",
            "company": comp or "(不明)",
            "amount": amt,
            "industry": industry,
            "industry_key": INDUSTRY_COLORKEY[industry],
            "affinity": aff,
            "affinity_label": label4(aff, 70, 40, 20),
            "winnability": win,
            "winnability_label": label4(win, 50, 25, 10),
            "opportunity": opp,
            "opp_parts": {"win": c_win, "size": c_size, "aff": c_aff},
            "size_attractiveness": sz,
            "comp_level": comp_assess["level"],
            "comp_score": comp_assess["score"],
            "comp_itype": comp_assess["itype"],
            "comp_ilabel": comp_assess["ilabel"],
            "comp_reason": comp_assess["reason"],
            "foothold": comp_assess["foothold"],
            "size_band": size_band(amt),
            "summary": make_summary(r, industry, aff, win),
        })

    for idx, e in enumerate(enriched):
        e["_idx"] = idx

    # -------- 予測・学習 (TabFM流 k近傍 + TimesFM流 年度再帰予測) --------
    pool_pred = [e for e in enriched if e["affinity"] >= 40]
    knn_enrich(enriched, pool_pred, k=8)
    upcoming = forecast_upcoming(pool_pred, top_n=30)

    # -------- 集計 --------
    n = len(enriched)
    total_amt = sum(e["amount"] for e in enriched)

    def agg_by(keyfn):
        d = defaultdict(lambda: {"count": 0, "amount": 0, "aff": 0, "win": 0})
        for e in enriched:
            k = keyfn(e)
            d[k]["count"] += 1
            d[k]["amount"] += e["amount"]
            d[k]["aff"] += e["affinity"]
            d[k]["win"] += e["winnability"]
        out = []
        for k, v in d.items():
            out.append({
                "key": k, "count": v["count"], "amount": v["amount"],
                "avg_affinity": round(v["aff"] / v["count"], 1),
                "avg_win": round(v["win"] / v["count"], 1),
                "amount_share": round(v["amount"] / total_amt * 100, 1) if total_amt else 0,
                "count_share": round(v["count"] / n * 100, 1),
            })
        return out

    by_industry = sorted(agg_by(lambda e: e["industry"]),
                         key=lambda x: INDUSTRY_ORDER.index(x["key"]))
    by_ministry = sorted(agg_by(lambda e: e["ministry"]), key=lambda x: -x["count"])
    by_size = agg_by(lambda e: e["size_band"])

    # 企業ランキング(落札額・件数) + 競合タイプ・DI競合度
    comp_d = defaultdict(lambda: {"count": 0, "amount": 0, "max_aff": 0})
    for e in enriched:
        comp_d[e["company"]]["count"] += 1
        comp_d[e["company"]]["amount"] += e["amount"]
        comp_d[e["company"]]["max_aff"] = max(comp_d[e["company"]]["max_aff"], e["affinity"])
    # 企業タイプ→対DI競合度レベル(金額規模に依らない企業単位の見立て)
    ENTRY_LEVEL = [(0.9, "競争可能"), (0.5, "要準備"), (0.25, "参入困難"), (0.0, "対象外")]

    def company_comp_level(company, max_aff):
        _k, ilabel, entry = classify_incumbent(company)
        if max_aff < 20:
            return ilabel, "対象外(領域外)", entry
        for th, lv in ENTRY_LEVEL:
            if entry >= th:
                return ilabel, lv, entry
        return ilabel, "対象外", entry

    top_companies = []
    for kname, v in sorted(comp_d.items(), key=lambda x: -x[1]["amount"])[:20]:
        ilabel, lv, entry = company_comp_level(kname, v["max_aff"])
        top_companies.append({"company": kname, "count": v["count"], "amount": v["amount"],
                              "ilabel": ilabel, "comp_level": lv, "entry": round(entry, 2)})

    # DI競合度レベル別 分布(件数・金額)
    comp_dist = defaultdict(lambda: {"count": 0, "amount": 0})
    for e in enriched:
        comp_dist[e["comp_level"]]["count"] += 1
        comp_dist[e["comp_level"]]["amount"] += e["amount"]
    COMP_ORDER = ["参入容易", "競争可能", "要準備", "参入困難", "対象外", "対象外(領域外)"]
    comp_level_dist = [{"level": lv, **comp_dist[lv]} for lv in COMP_ORDER if lv in comp_dist]

    # DI狙い目案件 Top(親和性40以上 かつ 勝率順/狙い目順)
    di_targets = sorted(
        [e for e in enriched if e["affinity"] >= 40],
        key=lambda e: (-e["opportunity"], -e["winnability"]))[:40]

    # 親和性帯別分布
    aff_bins = {"高(70-100)": 0, "中(40-69)": 0, "低(20-39)": 0, "圏外(0-19)": 0}
    for e in enriched:
        a = e["affinity"]
        if a >= 70: aff_bins["高(70-100)"] += 1
        elif a >= 40: aff_bins["中(40-69)"] += 1
        elif a >= 20: aff_bins["低(20-39)"] += 1
        else: aff_bins["圏外(0-19)"] += 1

    # 勝率帯別分布
    win_bins = {"高(50%+)": 0, "中(25-49%)": 0, "低(10-24%)": 0, "圏外(<10%)": 0}
    for e in enriched:
        w = e["winnability"]
        if w >= 50: win_bins["高(50%+)"] += 1
        elif w >= 25: win_bins["中(25-49%)"] += 1
        elif w >= 10: win_bins["低(10-24%)"] += 1
        else: win_bins["圏外(<10%)"] += 1

    # 金額規模ヒストグラム(対数帯)
    amt_bins = [
        ("〜100万", 0, 1_000_000), ("100万〜1千万", 1_000_000, 10_000_000),
        ("1千万〜1億", 10_000_000, 100_000_000), ("1億〜10億", 100_000_000, 1_000_000_000),
        ("10億〜", 1_000_000_000, 10**15),
    ]
    amt_hist = []
    for lbl, lo, hi in amt_bins:
        c = sum(1 for e in enriched if lo <= (e["amount"] or 0) < hi)
        s = sum(e["amount"] for e in enriched if lo <= (e["amount"] or 0) < hi)
        amt_hist.append({"label": lbl, "count": c, "amount": s})

    # DI獲得可能市場(SAM): 親和性40以上案件の落札額合計と勝率加重期待
    di_pool = [e for e in enriched if e["affinity"] >= 40]
    sam = sum(e["amount"] for e in di_pool)
    expected = sum(e["amount"] * e["winnability"] / 100 for e in di_pool)

    # 全体KPI + 示唆
    avg_aff = round(sum(e["affinity"] for e in enriched) / n, 1)
    avg_win = round(sum(e["winnability"] for e in enriched) / n, 1)
    kpi = {
        "total_count": n,
        "dataset_total": meta.get("row_count"),
        "total_amount": total_amt,
        "avg_amount": round(total_amt / n),
        "distinct_companies": len(comp_d),
        "distinct_ministries": len(by_ministry),
        "avg_affinity": avg_aff,
        "avg_win": avg_win,
        "di_pool_count": len(di_pool),
        "di_sam_amount": sam,
        "di_expected_amount": round(expected),
        "di_high_target_count": sum(1 for e in enriched if e["affinity"] >= 40 and e["winnability"] >= 40),
        "upcoming_count": len(upcoming),
        "upcoming_expected_amount": round(sum(
            u["pred_amount"] * u["expected"] / 100 for u in upcoming)),
    }

    insights = build_insights(kpi, by_industry, di_targets, top_companies, sam, expected, n, upcoming)

    out = {
        "generated_at": meta.get("last_updated"),
        "source": {
            "provider": "Data-Walkers 政府調達・入札落札データ",
            "dataset_id": DATASET_ID,
            "date_range": meta.get("date_range"),
            "note": "APIは1回あたり最大1000件を返すため、本ダッシュボードは代表サンプル1000件("
                    f"全{meta.get('row_count'):,}件)を分析対象とする。金額非公開案件は0円扱い。"
                    "業種・親和性・落札可能性はDI基準のルールベース推定値。",
        },
        "kpi": kpi,
        "by_industry": by_industry,
        "by_ministry": by_ministry[:12],
        "by_size": by_size,
        "amount_hist": amt_hist,
        "aff_bins": aff_bins,
        "win_bins": win_bins,
        "top_companies": top_companies,
        "comp_level_dist": comp_level_dist,
        "di_targets": di_targets,
        "upcoming": upcoming,
        "methodology": METHODOLOGY,
        "records": enriched,
        "insights": insights,
        "industry_order": INDUSTRY_ORDER,
    }
    # 内部インデックスは出力不要
    for e in enriched:
        e.pop("_idx", None)
    json.dump(out, open(os.path.join(HERE, "data.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    # file:// でも動くよう JS 版も出力
    with open(os.path.join(HERE, "data.js"), "w", encoding="utf-8") as f:
        f.write("window.DASH_DATA = ")
        json.dump(out, f, ensure_ascii=False)
        f.write(";")
    print("data.json / data.js 生成完了:")
    print(f"  案件 {n} 件 / 落札総額 {total_amt/1e8:.1f}億円 / 企業 {kpi['distinct_companies']}社")
    print(f"  平均DI親和性 {avg_aff} / 平均勝率 {avg_win}%")
    print(f"  DI対象プール {len(di_pool)}件, SAM {sam/1e8:.1f}億円, 期待受注額 {expected/1e8:.1f}億円")
    print(f"  高優先ターゲット(親和>=40 & 勝率>=40) {kpi['di_high_target_count']}件")


def build_insights(kpi, by_industry, di_targets, top_companies, sam, expected, n, upcoming=None):
    top_ind = max(by_industry, key=lambda x: x["amount"])
    best_di_ind = max(by_industry, key=lambda x: x["avg_win"])
    lines = []
    lines.append(
        f"分析対象{n}件の落札総額は{kpi['total_amount']/1e8:.1f}億円。最大の金額シェアは"
        f"「{top_ind['key']}」({top_ind['amount_share']}%)で、官公庁調達の主軸を占める。")
    lines.append(
        f"DIの中核・周辺領域(親和性40+)は{kpi['di_pool_count']}件・{sam/1e8:.1f}億円(SAM)。"
        f"勝率加重の期待受注額は約{expected/1e8:.1f}億円と試算され、"
        f"1000件サンプルの{kpi['di_pool_count']/n*100:.0f}%が射程に入る。")
    lines.append(
        f"業種別で最もDI勝率が高いのは「{best_di_ind['key']}」(平均勝率{best_di_ind['avg_win']}%)。"
        f"広報・Web・調査系に営業リソースを集中させるのが定石。")
    if di_targets:
        t = di_targets[0]
        lines.append(
            f"最優先ターゲットは「{t['project_name'][:34]}…」"
            f"(親和性{t['affinity']}/勝率{t['winnability']}%/{t['size_band']})。"
            f"現落札は{t['company']}で、規模・領域ともDIが競り込みやすい。")
    lines.append(
        f"逆に「電気・ガス」「建設・土木」「物品調達」は親和性ほぼ0でDI対象外。"
        f"金額は大きいが構造的に入札不可のため、母数から除外して営業効率を見るべき。")
    if upcoming:
        u = upcoming[0]
        lines.append(
            f"年度再帰予測では、来年度に再公告が見込まれるDI射程案件が{len(upcoming)}件。"
            f"期待受注額は約{kpi['upcoming_expected_amount']/1e8:.1f}億円。筆頭は"
            f"「{u['pred_name'][:30]}…」(再公告確率{u['recurrence_prob']}%×予測勝率{u['pred_win']}%)。"
            f"公告時期({u['announce_window'][:10]}頃)の3ヶ月前から仕様書入手・提案準備を始めるのが定石。")
    lines.append(
        "【注意】award_dateがサンプル上ほぼ同一日で時系列分析には不向き。"
        "本画面は業種構造・DI適性・狙い目案件の抽出に主眼を置く(推定値・参考値)。")
    return lines


if __name__ == "__main__":
    main()
