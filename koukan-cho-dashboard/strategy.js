/* 入札戦略シミュレータ
   ① 案件検索 → ② As-Is勝率 → ③ 5エージェント戦略会議 → ④ To-Be勝率 → ⑤ RFP読込 → ⑥ 価格 → ⑦ 判定 */
(function () {
  const D = window.DASH_DATA;
  if (!D) { document.body.innerHTML = "<p style='padding:40px'>data.js が読み込めませんでした。</p>"; return; }

  /* ---------- 共通ヘルパ ---------- */
  const IND_COLOR = {
    "広報・広告・マーケティング": "#e6533c", "Web・デジタル": "#2b5ce6", "システム・IT": "#5a6acf",
    "調査・コンサル・研究": "#00a3b4", "印刷・製本": "#9b8cff", "人材・研修・運営": "#c77dff",
    "電気・ガス・エネルギー": "#f5a623", "建設・土木・設備工事": "#8d99a6",
    "保守・管理・警備・清掃": "#7f8fa6", "医療・環境・検査": "#4caf87",
    "物品・機器・車両調達": "#b0872c", "その他": "#aab3c0",
  };
  const CL_COLOR = { "参入容易": "#1a9d63", "競争可能": "#2b9d8f", "要準備": "#f0a500", "参入困難": "#e07b39", "対象外": "#b9c0cc", "対象外(領域外)": "#b9c0cc" };
  const yen = (v) => v == null || !v ? "非公開" : (v >= 1e8 ? (v / 1e8).toFixed(2) + "億円" : v >= 1e4 ? Math.round(v / 1e4).toLocaleString() + "万円" : v.toLocaleString() + "円");
  const esc = (s) => (s || "").replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const indChip = (n) => `<span class="ind-chip" style="background:${IND_COLOR[n] || "#aab3c0"}">${n}</span>`;
  const clChip = (lv, s) => `<span class="chip" style="background:${CL_COLOR[lv] || "#b9c0cc"}">${lv}${s != null ? " " + s : ""}</span>`;
  const Z2A = s => (s || "").replace(/[０-９]/g, c => String.fromCharCode(c.charCodeAt(0) - 0xFEE0));
  const FY_RE = /令和\s*([0-9]{1,2})\s*年度/;

  /* 現職タイプ係数・金額障壁係数・推定入札者数 (build.py と同一モデル) */
  const ITYPE_COEF = { si: 0.30, ad: 0.78, carrier: 0.15, const: 0.05, org: 0.55, general: 1.0, unknown: 0.7 };
  function scaleFactor(a) {
    if (!a || a <= 0) return 0.7;
    if (a < 1e6) return 1.0; if (a < 1e7) return 1.0; if (a < 5e7) return 0.9;
    if (a < 1.5e8) return 0.75; if (a < 3e8) return 0.55; if (a < 1e9) return 0.3; return 0.15;
  }
  function expectedBidders(a, name) {
    let n = !a || a <= 0 ? 3.5 : a < 1e6 ? 2.5 : a < 1e7 ? 3.5 : a < 5e7 ? 4.5 : a < 1.5e8 ? 5.5 : 6.5;
    if (isProposal(name || "")) n -= 1.0;
    return Math.max(2.0, n);
  }

  /* ---------- ① 案件検索・抽出 ---------- */
  const selInd = document.getElementById("fIndustry");
  D.industry_order.forEach(n => { const o = document.createElement("option"); o.value = n; o.textContent = n; selInd.appendChild(o); });
  const st = { q: "", ind: "", comp: "", amt: "", diOnly: true, recur: false, selectedId: null };
  const $ = id => document.getElementById(id);

  function isRecur(r) { return FY_RE.test(Z2A(r.project_name)); }

  /* 日程表示: 公示日は元データ未収録のため、落札日(実績) + 毎年度型は次回サイクル予測を出す */
  const fmtDate = (d) => {
    if (!d) return null;
    const [y, m, dd] = d.split("-");
    return `${y}/${+m}/${+dd}`;
  };
  function nextCycle(r) {
    // 前年の落札日から次回サイクルを推定: 落札は約1年後の同時期、
    // 公告はその約2ヶ月前、入札期日(提出締切)は落札の約3週間前
    if (!isRecur(r) || !r.award_date) return null;
    const [y, m] = r.award_date.split("-").map(Number);
    const ny = y + 1;
    const annM = ((m - 3 + 12) % 12) + 1;               // 公告 ≈ 2〜3ヶ月前
    const annY = m - 3 < 0 ? ny - 1 : ny;
    const dlM = ((m - 2 + 12) % 12) + 1;                 // 入札期日 ≈ 1〜2ヶ月前
    const dlY = m - 2 < 0 ? ny - 1 : ny;
    return {
      announce: `${annY}年${annM}〜${annM % 12 + 1}月頃`,
      deadline: `${dlY}年${dlM}月〜${ny}年${m}月上旬頃`,
      award: `${ny}年${m}月頃`,
    };
  }
  function scheduleCell(r) {
    const parts = [];
    parts.push(`<div class="mini" style="white-space:nowrap">公示日: ${r.announcement_date ? fmtDate(r.announcement_date) : "未収録"}</div>`);
    parts.push(`<div style="white-space:nowrap;font-weight:600">落札日: ${fmtDate(r.award_date) || "—"}</div>`);
    const nc = nextCycle(r);
    if (nc) parts.push(`<div class="mini" style="white-space:nowrap;color:#2b5ce6" title="毎年度型のため前年サイクルから予測">次回公告予測: ${nc.announce}<br>入札期日予測: ${nc.deadline}</div>`);
    return parts.join("");
  }

  function renderSearch() {
    let rows = D.records.filter(r => {
      if (st.diOnly && r.affinity < 40) return false;
      if (st.ind && r.industry !== st.ind) return false;
      if (st.comp && r.comp_level !== st.comp) return false;
      if (st.recur && !isRecur(r)) return false;
      if (st.amt) { const [lo, hi] = st.amt.split("-").map(Number); if (!(r.amount >= lo && r.amount < hi)) return false; }
      if (st.q) { const q = st.q.toLowerCase(); if (!(r.project_name + r.ministry + r.company).toLowerCase().includes(q)) return false; }
      return true;
    }).sort((a, b) => b.opportunity - a.opportunity);
    $("searchCount").textContent = `${rows.length.toLocaleString()} 件ヒット`;
    const view = rows.slice(0, 200);
    $("searchTable").innerHTML =
      `<thead><tr><th>業種</th><th>案件名 / 発注元</th><th>公示日・入札期日</th><th class="num">落札金額</th><th class="num">現状勝率</th><th>競合度</th><th>年度</th></tr></thead><tbody>` +
      view.map(r => `<tr class="sel-row ${r.id === st.selectedId ? "selected" : ""}" data-id="${r.id}">
        <td>${indChip(r.industry)}</td>
        <td><div style="font-weight:600;max-width:340px">${esc(r.project_name)}</div>
            <div class="mini">${esc(r.ministry)} ／ 現落札: ${esc(r.company)}</div></td>
        <td style="font-size:11.5px">${scheduleCell(r)}</td>
        <td class="num">${yen(r.amount)}</td>
        <td class="num"><b>${r.winnability}%</b></td>
        <td>${clChip(r.comp_level)}</td>
        <td>${isRecur(r) ? '<span class="chip" style="background:#5a6acf">毎年度型</span>' : '<span class="mini">単発?</span>'}</td></tr>`).join("") + "</tbody>";
    [...document.querySelectorAll("#searchTable .sel-row")].forEach(tr =>
      tr.addEventListener("click", () => selectCase(tr.dataset.id)));
  }
  $("q").oninput = e => { st.q = e.target.value; renderSearch(); };
  selInd.onchange = e => { st.ind = e.target.value; renderSearch(); };
  $("fComp").onchange = e => { st.comp = e.target.value; renderSearch(); };
  $("fAmt").onchange = e => { st.amt = e.target.value; renderSearch(); };
  $("pDiOnly").onclick = () => { st.diOnly = !st.diOnly; $("pDiOnly").classList.toggle("on", st.diOnly); renderSearch(); };
  $("pRecur").onclick = () => { st.recur = !st.recur; $("pRecur").classList.toggle("on", st.recur); renderSearch(); };

  /* ---------- エージェント定義 ---------- */
  const REQ_TAGS = [
    ["Web/サイト", ["ウェブ", "Ｗｅｂ", "ＷＥＢ", "ホームページ", "サイト", "ポータル", "アクセシビリティ"]],
    ["広報/情報発信", ["広報", "情報発信", "周知", "普及啓発", "啓発", "ＰＲ", "プロモーション"]],
    ["広告運用", ["広告", "リスティング", "運用型"]],
    ["SNS/動画", ["ＳＮＳ", "動画", "映像", "コンテンツ"]],
    ["調査/分析", ["調査", "分析", "アンケート", "統計", "検証", "モニタリング"]],
    ["データ/KPI", ["データ", "ＫＰＩ", "効果測定", "アクセス解析", "レポート"]],
    ["システム/DX", ["システム", "ＤＸ", "デジタル", "アプリ", "クラウド", "ＡＩ"]],
    ["運営/事務局", ["運営", "事務局", "支援業務", "研修", "セミナー", "説明会"]],
  ];
  const DI_STRENGTH = {
    "Web/サイト": "Web制作・UI/UX改善・アクセシビリティ対応の実績",
    "広報/情報発信": "デジタル広報・オウンドメディア運用の知見",
    "広告運用": "運用型広告(リスティング/ディスプレイ)の運用体制 — DIの中核事業",
    "SNS/動画": "SNSアカウント運用・動画プロモーションの制作網",
    "調査/分析": "デジタルリサーチ・競合調査の分析チーム",
    "データ/KPI": "アクセス解析・KPI設計・ダッシュボード構築 — DIの強み",
    "システム/DX": "DX支援・MAツール導入の技術パートナー網",
    "運営/事務局": "キャンペーン事務局・イベント運営の対応力",
  };
  function extractReqs(text) {
    const hits = [];
    for (const [tag, kws] of REQ_TAGS) if (kws.some(k => text.includes(k))) hits.push(tag);
    return hits;
  }
  const isProposal = (t) => ["支援", "企画", "広報", "調査", "戦略", "検討", "推進"].some(k => t.includes(k));

  function runAgents(r) {
    const t = r.project_name || "";
    const aff = r.affinity, itype = r.comp_itype;
    const entry = ITYPE_COEF[itype] ?? 0.7, sf = scaleFactor(r.amount);
    const reqs = extractReqs(t);
    const outOfScope = aff < 20;

    /* A1 案件分析 */
    const a1 = {
      icon: "🧭", color: "#5a6acf", name: "案件分析エージェント", role: "要求要素の分解とDI適合診断",
      coef: 1.0, coefNote: "診断のみ(他エージェントの入力)",
      findings: [
        `案件名から検出した要求要素: ${reqs.length ? reqs.join(" / ") : "明確な要素なし(RFP読込を推奨)"}`,
        `DI親和性 ${aff}/100(${r.affinity_label}) — ${aff >= 70 ? "DI中核領域" : aff >= 40 ? "DI周辺領域、強み訴求で戦える" : "DI領域外寄り"}`,
        `入札方式の推定: ${isProposal(t) ? "企画競争(プロポーザル)の可能性が高い — 提案点で差別化可能" : "一般競争(価格重視)の可能性 — 価格戦略が重要"}`,
      ],
      verdict: outOfScope ? "DIの提供役務範囲外。原則見送りを提言。" :
        `要求要素とDI事業(SEO/運用型広告/Web/データ分析)の重なりは${reqs.length >= 3 ? "大きい" : reqs.length ? "部分的" : "要確認"}。`,
    };

    /* A2 競合分析 */
    const compPlan = {
      si: { coef: 1.05, dir: "大手SIは体制・実績で勝るが、間接費が高く小回りが利かない。デジタルマーケ専門性では劣る。", win: "「専門特化 × 機動力 × コスト効率」ポジションで差別化。全面勝負を避け、デジタル領域の提案深度で勝つ。" },
      ad: { coef: 1.07, dir: "大手広告代理店はマス統合に強いが、運用型・SEO・解析の内製度はDIが優位に立てる。", win: "運用実績の数値(改善率・CPA等)を前面に。マス前提の見積り構造より低コストで同等以上の効果を提示。" },
      org: { coef: 1.08, dir: "団体・独法は随意的な継続受注が多く、提案品質での競争には弱い傾向。", win: "企画競争化のタイミングを捉え、民間水準の提案品質・スピード・KPI管理で明確な差を見せる。" },
      general: { coef: 1.06, dir: "現職は中堅・中小の一般企業。価格・提案とも正面から競争可能。", win: "現職の納品物を調査し、デジタル専門性(解析・広告・SEO)の上積みを提案の軸に。" },
      carrier: { coef: 1.0, dir: "通信キャリアのインフラ案件で競合対象にならない。", win: "参入対象外。" },
      const: { coef: 1.0, dir: "建設・設備の許可業種で競合対象にならない。", win: "参入対象外。" },
      unknown: { coef: 1.05, dir: "現職の素性が不明。過去落札履歴の調査が先決。", win: "調達ポータルで過去3年の同種案件の落札者・金額を確認し、勝ち筋を見極める。" },
    }[itype] || { coef: 1.03, dir: "-", win: "-" };
    const a2 = {
      icon: "⚔️", color: "#e6533c", name: "競合分析エージェント", role: "現職の方向性分析とポジショニング設計",
      coef: outOfScope ? 1.0 : compPlan.coef, coefNote: "競合理解によるポジショニング効果",
      findings: [
        `現職: ${r.company}(${r.comp_ilabel}) — 参入度スコア ${r.comp_score}/100(${r.comp_level})`,
        `競合の方向性: ${compPlan.dir}`,
      ],
      verdict: `勝ち筋: ${compPlan.win}`,
    };

    /* A3 価格戦略 */
    const estBudget = r.amount ? Math.round(r.amount / 0.88) : 0; // 前年落札額≒予定価格の88%と仮定
    const proposal = isProposal(t);
    const range = proposal ? [0.90, 0.95] : [0.82, 0.88];
    const priceCoefMap = { si: 1.12, general: 1.08, ad: 1.07, org: 1.06, unknown: 1.05, carrier: 1.0, const: 1.0 };
    let priceCoef = priceCoefMap[itype] ?? 1.04;
    if (r.amount && r.amount < 1e6) priceCoef = Math.min(priceCoef, 1.03);
    const a3 = {
      icon: "💴", color: "#f5a623", name: "価格戦略エージェント", role: "予定価格の推定と入札価格レンジの設計",
      coef: outOfScope ? 1.0 : priceCoef, coefNote: "適正価格設計による価格点・費用対効果の向上",
      findings: [
        `推定予定価格: ${yen(estBudget)}(前年落札 ${yen(r.amount)} ÷ 落札率88%と仮定)`,
        `推奨入札レンジ: 予定価格の${Math.round(range[0] * 100)}〜${Math.round(range[1] * 100)}% = ${yen(Math.round(estBudget * range[0]))} 〜 ${yen(Math.round(estBudget * range[1]))}`,
        proposal ? "企画競争型のため無理な安値は不要。技術点重視の価格設定。" : "価格競争型のため前年落札額の5〜10%下をターゲット。最低制限/低入札調査基準に注意。",
        itype === "si" ? "大手SI現職は間接費が高く、DIは10〜20%低い価格で同等品質を出せる = 最大の武器。" : "現職の価格水準(前年落札額)を基準に、コスト構造の差で優位を作る。",
      ],
      verdict: `価格弾力性: ${itype === "si" ? "高(大手比のコスト優位が効く)" : itype === "general" ? "中〜高(効率化で差を作る)" : "中"}。`,
    };

    /* A4 提案・マーケティング戦略 */
    const strengths = reqs.map(q => DI_STRENGTH[q]).filter(Boolean);
    let propCoef = 1 + 0.05 + 0.09 * aff / 100;
    if (proposal) propCoef += 0.03;
    const a4 = {
      icon: "📣", color: "#2b5ce6", name: "提案・マーケティング戦略エージェント", role: "DI強みの提案骨子への変換とKPI設計",
      coef: outOfScope ? 1.0 : +propCoef.toFixed(2), coefNote: "提案品質・技術点の向上(親和性が高いほど大)",
      findings: [
        `活かせるDIの強み: ${strengths.length ? strengths.slice(0, 3).join(" ／ ") : "汎用提案力(RFP読込で具体化を推奨)"}`,
        "提案骨子: ①現状課題のデータ診断(無償プレ分析) ②KPIツリー設計 ③実行体制と月次PDCA ④類似実績の数値成果",
        "技術点対策: 評価基準の配点に合わせ、実績・体制・企画の3章立てで加点を最大化。",
        proposal ? "企画競争のため提案戦略の寄与が最大。プレゼン想定問答まで準備。" : "一般競争でも仕様理解の深さが価格の信頼性を裏付ける。",
      ],
      verdict: `マーケティング戦略の要: 発注者の政策目的(KPI)から逆算した「成果で語る提案」。親和性${aff}の案件では技術点の伸びしろが${aff >= 70 ? "大きい" : "中程度"}。`,
    };

    /* A5 体制・実績 */
    const newSf = Math.min(1, sf + 0.10);
    const orgCoef = +Math.min(1.12, newSf / sf).toFixed(2);
    const a5 = {
      icon: "🏗️", color: "#00a3b4", name: "体制・実績エージェント", role: "資格・実績・体制の障壁解消プラン",
      coef: outOfScope ? 1.0 : orgCoef, coefNote: "金額障壁係数の1段階改善(資格・実績・体制整備)",
      findings: [
        `現在の金額障壁係数: ${sf.toFixed(2)} → 対策後 ${newSf.toFixed(2)}(統一資格の等級確認・同種実績の証憑整理・専任PM体制)`,
        r.amount >= 1.5e8 ? "大型案件のため、再委託網またはJV(共同事業体)で規模要件を補完。" : "この規模なら単独体制で対応可能。実績書類の整備が主対策。",
        "官公庁経験者(仕様書読解・調書作成)をアサインし、事務要件の失点をゼロに。",
      ],
      verdict: orgCoef > 1 ? `体制整備で勝率を${Math.round((orgCoef - 1) * 100)}%相対改善できる。` : "この規模では障壁が既に低く、体制面の追加効果は小さい(現体制で参加可能)。",
    };

    return [a1, a2, a3, a4, a5];
  }

  /* ---------- RFP解析 ---------- */
  const RFP_POS = ["SEO", "ＳＥＯ", "検索", "広告", "リスティング", "運用型", "SNS", "ＳＮＳ", "動画", "コンテンツ",
    "ウェブ", "Web", "Ｗｅｂ", "ホームページ", "サイト", "アクセス解析", "解析", "KPI", "ＫＰＩ", "効果測定",
    "デジタル", "マーケティング", "広報", "情報発信", "プロモーション", "ブランディング", "レポート", "改善提案",
    "ペルソナ", "ターゲット", "CV", "ＣＶ", "エンゲージメント", "オウンドメディア", "リサーチ", "調査", "分析", "戦略"];
  const RFP_NEG = ["工事", "建設", "電気の購入", "ガスの調達", "燃料", "警備", "清掃", "医薬品", "車両", "賃貸借", "リース", "什器"];
  function analyzeRfp(text) {
    if (!text || text.trim().length < 20) return null;
    const pos = [...new Set(RFP_POS.filter(k => text.includes(k)))];
    const neg = [...new Set(RFP_NEG.filter(k => text.includes(k)))];
    const up = Math.min(0.12, pos.length * 0.015);
    const down = Math.min(0.20, neg.length * 0.04);
    return { pos, neg, coef: +(1 + up - down).toFixed(2), textLen: text.length };
  }

  /* ---------- シミュレーション実行 ---------- */
  let currentCase = null, agents = [], rfp = null, wfChart = null;
  // 挑戦者のAs-Is上限は60%(競合・コンペを織り込み)。戦略を尽くしても65%が現実的な天井。
  const CAP = 65;

  function computeWin() {
    const base = currentCase.winnability;
    const steps = [];
    let cum = base;
    for (const a of agents.slice(1)) { // A1は診断のみ
      const next = Math.min(CAP, cum * a.coef);
      steps.push({ name: a.name.replace("エージェント", ""), from: cum, to: next, coef: a.coef });
      cum = next;
    }
    const toBe = +Math.max(base, cum).toFixed(1);
    let final = toBe;
    if (rfp) final = +Math.min(CAP, toBe * rfp.coef).toFixed(1); // RFPで領域外が判明した場合は下がり得る
    const saturated = toBe >= CAP - 1;
    return { base, steps, toBe, final, saturated };
  }

  function selectCase(id) {
    currentCase = D.records.find(r => r.id === id);
    if (!currentCase) return;
    st.selectedId = id;
    agents = runAgents(currentCase);
    rfp = null; $("rfpText").value = ""; $("rfpResult").innerHTML = ""; $("rfpStatus").textContent = "";
    $("simBody").style.display = "";
    $("emptyHint").style.display = "none";
    // ナビのロック解除
    document.querySelectorAll("#nav a.locked").forEach(a => a.classList.remove("locked"));
    const nn = $("navNote");
    if (nn) nn.textContent = "✅ 案件選択済み — 各セクションへ移動できます";
    renderSearch();
    renderAll();
    document.getElementById("baseline").scrollIntoView({ behavior: "smooth" });
  }

  /* 未選択時にロック中ナビを押したら、①へ誘導してハイライト */
  function guideToSearch() {
    document.getElementById("search").scrollIntoView({ behavior: "smooth" });
    const panel = document.querySelector("#search .panel");
    panel.classList.remove("flash-guide");
    void panel.offsetWidth; // reflow でアニメ再発火
    panel.classList.add("flash-guide");
    $("searchCount").textContent = "👈 まず案件の行をクリックして選択してください";
  }
  document.getElementById("nav").addEventListener("click", e => {
    const a = e.target.closest("a");
    if (a && a.classList.contains("locked")) { e.preventDefault(); guideToSearch(); }
  });

  function renderAll() {
    const r = currentCase;
    const W = computeWin();
    const entry = ITYPE_COEF[r.comp_itype] ?? 0.7, sf = scaleFactor(r.amount);

    /* ② 案件カード + As-Is */
    $("caseCard").innerHTML = `
      <h4>${indChip(r.industry)} 選択中の案件</h4>
      <div style="font-size:15px;font-weight:700;margin:6px 0">${esc(r.project_name)}</div>
      <div class="mini" style="margin-bottom:10px">${esc(r.ministry)}${r.agency ? " ・" + esc(r.agency) : ""} ／ 現落札: ${esc(r.company)}(${esc(r.comp_ilabel)}) ／ 落札額 ${yen(r.amount)}</div>
      <div class="mini" style="margin-bottom:10px">📅 公示日: ${r.announcement_date ? fmtDate(r.announcement_date) : "未収録"} ／ 落札日: ${fmtDate(r.award_date) || "—"}${(() => { const nc = nextCycle(r); return nc ? ` ／ <b style="color:#2b5ce6">次回公告予測 ${nc.announce}・入札期日予測 ${nc.deadline}</b>` : ""; })()}</div>
      <div style="font-size:12.5px;color:#4b5563">${esc(r.summary)}</div>
      ${isRecur(r) ? `<div class="mini" style="margin-top:8px">📅 毎年度型案件 — 来年度の再公告が見込まれるため、公告3ヶ月前からの準備が可能。</div>` : ""}`;
    const nb = expectedBidders(r.amount, r.project_name);
    $("baselineCard").innerHTML = `
      <h4>現状のままの勝率(As-Is)</h4>
      <div class="winbig" style="color:${W.base >= 30 ? "#1a9d63" : W.base >= 15 ? "#f0a500" : "#e07b39"}">${W.base}<small>%</small></div>
      <p class="hint" style="margin-top:8px">= 親和性 ${r.affinity}/100 × 現職係数 ${entry.toFixed(2)}(${esc(r.comp_ilabel)}) × 金額障壁 ${sf.toFixed(2)} ÷ 推定入札者数 ${nb}社 × 2</p>
      <div style="font-size:12px;color:#4b5563">戦略なし・現状の資格/実績/体制のまま入札した場合の推定値。
      この規模・方式の案件には<b>約${nb}社</b>の応札が見込まれ、平均的な参加者の勝率は${(100 / nb).toFixed(0)}%。
      競合・コンペの存在を織り込むため挑戦者の上限は60%(戦略を尽くしても75%)としている。競合度: ${clChip(r.comp_level, r.comp_score)}</div>`;

    /* ③ エージェント */
    $("agentGrid").innerHTML = agents.map(a => `
      <div class="agent-card">
        <div class="ahead"><div class="avatar" style="background:${a.color}">${a.icon}</div>
          <div><h4>${a.name}</h4><div class="role">${a.role}</div></div>
          <div class="coef" title="${a.coefNote}">×${a.coef.toFixed ? a.coef.toFixed(2) : a.coef}</div></div>
        <ul>${a.findings.map(f => `<li>${esc(f)}</li>`).join("")}</ul>
        <div class="verdict">💡 ${esc(a.verdict)}</div>
      </div>`).join("");

    renderWaterfall(W);
    renderCompare(W);
    renderPricing(r);
    renderVerdict(r, W);
  }

  function renderWaterfall(W) {
    const labels = ["現状(As-Is)", ...W.steps.map(s => s.name), "戦略後(To-Be)"];
    const data = [[0, W.base], ...W.steps.map(s => [s.from, s.to]), [0, W.toBe]];
    const colors = ["#8d99a6", ...W.steps.map(() => "#1a9d63"), "#2b5ce6"];
    if (rfp) { labels.push("RFP読込効果", "最終予測"); data.push([W.toBe, W.final], [0, W.final]); colors.push(rfp.coef >= 1 ? "#00b3a4" : "#e07b39", "#00b3a4"); }
    if (wfChart) wfChart.destroy();
    wfChart = new Chart($("wfChart"), {
      type: "bar",
      data: { labels, datasets: [{ data, backgroundColor: colors, borderRadius: 4, borderSkipped: false }] },
      options: {
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => { const [a, b] = c.raw; return a ? `${a.toFixed(1)}% → ${b.toFixed(1)}% (+${(b - a).toFixed(1)}pt)` : `${b.toFixed(1)}%`; } } } },
        scales: { y: { min: 0, max: 100, title: { display: true, text: "推定勝率(%)" } }, x: { ticks: { font: { size: 10 } } } }
      }
    });
  }

  function renderCompare(W) {
    const dlt = (a, b) => { const d = +(b - a).toFixed(1); return `<span class="delta-badge ${d < 0 ? "down" : ""}">${d >= 0 ? "+" : ""}${d}pt</span>`; };
    $("compareBox").innerHTML = `
      <div style="display:flex;flex-direction:column;gap:12px;margin-top:6px">
        <div class="compare-card"><div class="t">① 現状のまま</div><div class="winbig" style="font-size:30px;color:#8d99a6">${W.base}<small>%</small></div></div>
        <div class="compare-card"><div class="t">② エージェント戦略立案後</div><div class="winbig" style="font-size:30px;color:#2b5ce6">${W.toBe}<small>%</small></div>${dlt(W.base, W.toBe)}</div>
        <div class="compare-card hero"><div class="t">③ ＋RFP・企画書読込＆マーケ戦略</div><div class="winbig" style="font-size:30px">${rfp ? W.final + "<small>%</small>" : "—"}</div>${rfp ? dlt(W.toBe, W.final) : '<span class="mini" style="color:#dff1ff">⑤にRFPを貼ると算出</span>'}</div>
      </div>
      ${W.saturated ? '<p class="hint" style="margin-top:10px">※ この案件は勝率が戦略後の推定上限(65%)近くに到達。競合・コンペがある以上100%はあり得ないため、エージェントの提言は「取りこぼさない」ための実行チェックリストとして活用。</p>' : ""}`;
  }

  function renderPricing(r) {
    const estBudget = r.amount ? Math.round(r.amount / 0.88) : 0;
    const proposal = isProposal(r.project_name);
    const range = proposal ? [0.90, 0.95] : [0.82, 0.88];
    $("pricingCard").innerHTML = `
      <div class="kpi-grid" style="grid-template-columns:repeat(4,1fr)">
        <div class="kpi"><div class="lbl">前年落札額(実績)</div><div class="val" style="font-size:20px">${yen(r.amount)}</div></div>
        <div class="kpi"><div class="lbl">推定予定価格</div><div class="val" style="font-size:20px">${yen(estBudget)}</div><div class="sub">落札率88%仮定で逆算</div></div>
        <div class="kpi brand"><div class="lbl">推奨入札レンジ</div><div class="val" style="font-size:20px">${yen(Math.round(estBudget * range[0]))}〜${yen(Math.round(estBudget * range[1]))}</div><div class="sub">予定価格の${Math.round(range[0] * 100)}〜${Math.round(range[1] * 100)}%</div></div>
        <div class="kpi"><div class="lbl">入札方式の推定</div><div class="val" style="font-size:16px">${proposal ? "企画競争型" : "価格競争型"}</div><div class="sub">${proposal ? "技術点重視・安値不要" : "価格重視・低入基準に注意"}</div></div>
      </div>
      <p class="hint" style="margin-top:12px">※ 予定価格・落札率は公表データからの推定。実際は入札説明書の予定価格(事前公表の場合)と過去の落札結果(調達ポータル)で必ず検証すること。
      ${r.comp_itype === "si" ? "現職が大手SIのため、間接費差によりDIは10〜20%低い価格で同等品質を提示可能 — 価格が最大の武器。" : ""}</p>`;
  }

  function renderVerdict(r, W) {
    const lines = [];
    const rec = W.final ?? W.toBe;
    if (r.affinity < 20) {
      lines.push("この案件はDIの提供役務の範囲外。戦略を組んでも勝率改善は限定的で、原則【見送り】を推奨。");
    } else {
      const judge = rec >= 40 ? "【入札推奨】複数社競合の中でも頭一つ抜けられる水準。" :
        rec >= 25 ? "【条件付き推奨】平均的な応札者(勝率2〜3割)を上回れる。提案体制と価格の詰めができれば挑戦価値あり。" :
        "【慎重判断】競合を考慮すると現状は分が悪い。実績積み上げか再委託参画から始めるのが現実的。";
      lines.push(`${judge}(現状${W.base}% → 戦略後${W.toBe}%${rfp ? ` → RFP読込後${W.final}%` : ""})`);
    }
    const gain = +(W.toBe - W.base).toFixed(1);
    lines.push(`エージェント戦略の効果は+${gain}pt。内訳: 競合ポジショニング(×${agents[1].coef}) / 価格設計(×${agents[2].coef}) / 提案・マーケ戦略(×${agents[3].coef}) / 体制整備(×${agents[4].coef})。`);
    if (rfp) {
      lines.push(`RFP読込の効果は×${rfp.coef}(${rfp.coef >= 1 ? "+" : ""}${((rfp.coef - 1) * 100).toFixed(0)}%相対)。検出キーワード${rfp.pos.length}件から仕様の解像度が上がり、提案の的中率とマーケティング戦略の具体性が向上。`);
      if (rfp.neg.length) lines.push(`【注意】RFPにDI領域外のキーワード(${rfp.neg.join("/")})を検出。役務範囲の切り分けを仕様確認で必ず行うこと。`);
    } else {
      lines.push("⑤でRFP・仕様書を読み込むと、マーケティング戦略の精度向上分がさらに上乗せされる(適合キーワードで最大+12%相対、領域外検出時は減点)。");
    }
    if (isRecur(r)) {
      const m = Z2A(r.project_name).match(FY_RE);
      if (m) lines.push(`毎年度型のため、令和${+m[1] + 1}年度の公告(${2019 + (+m[1])}年1〜3月頃)に向けて、今から資格・実績・提案骨子を準備すれば間に合う。`);
    }
    lines.push("【免責】勝率はルールベースの推定値。入札判断は公告原文・仕様書・評価基準の確認が前提。");
    $("verdictList").innerHTML = lines.map(t => `<li${t.startsWith("【注意】") || t.startsWith("【免責】") ? " class='warn'" : ""}>${esc(t)}</li>`).join("");
  }

  /* ---------- RFP UI ---------- */
  $("btnAnalyze").onclick = () => {
    const text = $("rfpText").value;
    rfp = analyzeRfp(text);
    if (!rfp) { $("rfpStatus").textContent = "テキストが短すぎます(20文字以上を貼り付けてください)"; return; }
    $("rfpStatus").textContent = `解析完了: ${rfp.textLen.toLocaleString()}文字`;
    const W = computeWin();
    $("rfpResult").innerHTML = `
      <div class="cols c2">
        <div>
          <h4 style="margin:0 0 6px">検出したDI適合キーワード(${rfp.pos.length}件)</h4>
          <div>${rfp.pos.map(k => `<span class="kw-hit">${esc(k)}</span>`).join("") || "<span class='mini'>なし</span>"}</div>
          ${rfp.neg.length ? `<h4 style="margin:12px 0 6px">⚠ 領域外キーワード(${rfp.neg.length}件)</h4><div>${rfp.neg.map(k => `<span class="kw-hit neg">${esc(k)}</span>`).join("")}</div>` : ""}
        </div>
        <div>
          <h4 style="margin:0 0 6px">RFP読込による勝率変化</h4>
          <div style="font-size:13px">戦略後 ${W.toBe}% × RFP係数 ${rfp.coef} → <b style="font-size:20px;color:#00b3a4">${W.final}%</b>
            <span class="delta-badge ${W.final - W.toBe < 0 ? "down" : ""}">${W.final - W.toBe >= 0 ? "+" : ""}${(W.final - W.toBe).toFixed(1)}pt</span></div>
          <p class="hint" style="margin-top:8px">仕様の解像度が上がることで、提案の的中率・評価基準への適合・マーケティング戦略(ターゲット/KPI/チャネル設計)の具体性が向上する効果を係数化。</p>
        </div>
      </div>`;
    renderWaterfall(W); renderCompare(W); renderVerdict(currentCase, W);
  };
  /* ---------- RFPファイル読込 (ドラッグ&ドロップ / ファイル選択 / PDF対応) ---------- */
  const PDFJS_URL = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
  const PDFJS_WORKER = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
  let pdfjsReady = null;
  function loadPdfjs() {
    if (window.pdfjsLib) return Promise.resolve();
    if (pdfjsReady) return pdfjsReady;
    pdfjsReady = new Promise((res, rej) => {
      const s = document.createElement("script");
      s.src = PDFJS_URL;
      s.onload = () => { window.pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS_WORKER; res(); };
      s.onerror = () => rej(new Error("pdf.js の読み込みに失敗(オフライン?)"));
      document.head.appendChild(s);
    });
    return pdfjsReady;
  }
  async function extractPdfText(file) {
    await loadPdfjs();
    const buf = await file.arrayBuffer();
    const pdf = await window.pdfjsLib.getDocument({ data: buf }).promise;
    let text = "";
    const maxPages = Math.min(pdf.numPages, 60);
    for (let p = 1; p <= maxPages; p++) {
      const page = await pdf.getPage(p);
      const tc = await page.getTextContent();
      text += tc.items.map(i => i.str).join(" ") + "\n";
    }
    if (pdf.numPages > maxPages) text += `\n(※ ${pdf.numPages}ページ中 先頭${maxPages}ページを抽出)`;
    return text;
  }
  async function readRfpFiles(files) {
    const list = [...files].filter(f =>
      f.type === "application/pdf" || /\.(pdf|txt|md|csv|text)$/i.test(f.name) || f.type.startsWith("text/"));
    if (!list.length) { $("rfpStatus").textContent = "対応形式は PDF / .txt / .md / .csv です"; return; }
    $("rfpStatus").textContent = "読み込み中…";
    const parts = [];
    for (const f of list) {
      try {
        const isPdf = f.type === "application/pdf" || /\.pdf$/i.test(f.name);
        const text = isPdf ? await extractPdfText(f) : await f.text();
        parts.push(`【${f.name}】\n${text.trim()}`);
      } catch (e) {
        parts.push(`【${f.name}】(読み込み失敗: ${e.message})`);
      }
    }
    const merged = parts.join("\n\n");
    $("rfpText").value = ($("rfpText").value.trim() ? $("rfpText").value.trim() + "\n\n" : "") + merged;
    $("rfpDropHint").innerHTML = "📄 読み込み済み: " +
      list.map(f => `<span class="rfp-file-chip">${esc(f.name)}</span>`).join("") +
      '<span class="mini">追加のファイルをドロップ、またはこのまま解析</span>';
    $("rfpStatus").textContent = `${list.length}ファイル読込完了 → 自動解析します`;
    if (currentCase) $("btnAnalyze").click();
  }
  const dropZone = $("rfpDrop");
  ["dragenter", "dragover"].forEach(ev => dropZone.addEventListener(ev, e => {
    e.preventDefault(); e.stopPropagation(); dropZone.classList.add("dragover");
  }));
  ["dragleave", "drop"].forEach(ev => dropZone.addEventListener(ev, e => {
    e.preventDefault(); e.stopPropagation(); dropZone.classList.remove("dragover");
  }));
  dropZone.addEventListener("drop", e => {
    if (e.dataTransfer && e.dataTransfer.files.length) readRfpFiles(e.dataTransfer.files);
  });
  $("rfpFile").addEventListener("change", e => {
    if (e.target.files.length) readRfpFiles(e.target.files);
    e.target.value = "";
  });

  $("btnSample").onclick = () => {
    $("rfpText").value = "本業務は、当省が運営するウェブサイトについて、アクセス解析に基づく課題抽出及びSEO改善、SNSを活用した情報発信の強化、運用型広告(リスティング広告等)の企画・運用、並びに効果測定(KPI設計・月次レポート)を行うものである。受託者は、ターゲットの分析(ペルソナ設計)を行い、コンテンツの企画制作及び改善提案を継続的に実施すること。デジタルマーケティングに関する専門的知見を有する者を配置し、広報戦略の策定を支援すること。";
    $("btnAnalyze").click();
  };

  /* ---------- 初期化 (URL ?id= 対応 / デモボタン) ---------- */
  renderSearch();
  const btnDemo = $("btnDemo");
  if (btnDemo) btnDemo.onclick = () => selectCase(D.di_targets[0].id);
  const pid = new URLSearchParams(location.search).get("id");
  if (pid) selectCase(pid);

  /* nav scrollspy */
  const navLinks = [...document.querySelectorAll("#nav a")];
  const secs = navLinks.map(a => document.querySelector(a.getAttribute("href")));
  const obs = new IntersectionObserver(ents => {
    ents.forEach(en => { if (en.isIntersecting) navLinks.forEach(a => a.classList.toggle("active", a.getAttribute("href") === "#" + en.target.id)); });
  }, { rootMargin: "-20% 0px -70% 0px" });
  secs.forEach(s => s && obs.observe(s));
})();
