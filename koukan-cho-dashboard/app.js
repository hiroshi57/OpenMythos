/* 官公庁調達 × DI ダッシュボード 描画ロジック */
(function () {
  const D = window.DASH_DATA;
  if (!D) { document.body.innerHTML = "<p style='padding:40px'>data.js が読み込めませんでした。build.py を実行してください。</p>"; return; }

  const IND_COLOR = {
    "広報・広告・マーケティング": "#e6533c", "Web・デジタル": "#2b5ce6", "システム・IT": "#5a6acf",
    "調査・コンサル・研究": "#00a3b4", "印刷・製本": "#9b8cff", "人材・研修・運営": "#c77dff",
    "電気・ガス・エネルギー": "#f5a623", "建設・土木・設備工事": "#8d99a6",
    "保守・管理・警備・清掃": "#7f8fa6", "医療・環境・検査": "#4caf87",
    "物品・機器・車両調達": "#b0872c", "その他": "#aab3c0",
  };
  const yen = (v) => v == null ? "—" : (
    v >= 1e8 ? (v / 1e8).toFixed(1) + "億円" :
    v >= 1e4 ? Math.round(v / 1e4).toLocaleString() + "万円" :
    v.toLocaleString() + "円");
  const oku = (v) => (v / 1e8).toFixed(1);
  const num = (v) => (v ?? 0).toLocaleString();
  const esc = (s) => (s || "").replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const lab = (l) => `<span class="chip lab-${l}">${l}</span>`;
  const indChip = (name) => `<span class="ind-chip" style="background:${IND_COLOR[name] || "#aab3c0"}">${name}</span>`;
  const scoreBar = (v, color) => `<span class="scorebar"><i style="width:${v}%;background:${color}"></i></span> <b>${v}</b>`;

  /* ---------- head / footer ---------- */
  const src = D.source;
  document.getElementById("head-note").textContent = src.note;
  document.getElementById("sidefoot").innerHTML =
    `出典: ${esc(src.provider)}<br>収録 ${num(D.kpi.dataset_total)} 件<br>期間 ${src.date_range.start}〜${src.date_range.end}`;
  document.getElementById("footnote").innerHTML =
    `<b>データ根拠・免責</b>：出典=${esc(src.provider)}（dataset ${src.dataset_id}）。` +
    `APIは1回あたり最大1000件を返すため、本画面は代表サンプル1000件（全${num(D.kpi.dataset_total)}件）を分析。` +
    `「業界業種」「DI親和性」「落札可能性」「狙い目スコア」はDI（デジタルアイデンティティ）の事業領域を基準にした` +
    `ルールベースの<b>推定・参考値</b>であり、実際の入札可否・勝敗を保証するものではありません。金額非公開案件は0円として集計。` +
    `「予測・今後の勝てそうなネタ」はTimesFM/TabFMの思想を踏まえた軽量実装（類似案件k近傍学習＋年度再帰予測）による` +
    `<b>先読みの試算</b>で、実際の公告・予算措置を保証しません。入札判断は必ず公告原文・仕様書で確認してください。`;

  /* ---------- KPI ---------- */
  const k = D.kpi;
  const kpis = [
    { lbl: "分析対象案件", val: num(k.total_count), unit: "件", sub: `全${num(k.dataset_total)}件の代表サンプル`, cls: "" },
    { lbl: "落札総額", val: oku(k.total_amount), unit: "億円", sub: `平均 ${yen(k.avg_amount)}/件`, cls: "" },
    { lbl: "落札企業数", val: num(k.distinct_companies), unit: "社", sub: `${k.distinct_ministries}省庁が発注`, cls: "" },
    { lbl: "平均DI親和性", val: k.avg_affinity, unit: "/100", sub: `平均勝率 ${k.avg_win}%`, cls: "" },
    { lbl: "DI射程案件(親和≥40)", val: num(k.di_pool_count), unit: "件", sub: `全体の${(k.di_pool_count / k.total_count * 100).toFixed(0)}%`, cls: "brand" },
    { lbl: "DI獲得可能市場 SAM", val: oku(k.di_sam_amount), unit: "億円", sub: "射程案件の落札額合計", cls: "brand" },
    { lbl: "勝率加重 期待受注額", val: oku(k.di_expected_amount), unit: "億円", sub: "SAM×推定勝率", cls: "teal" },
    { lbl: "高優先ターゲット", val: num(k.di_high_target_count), unit: "件", sub: "親和≥40 かつ 勝率≥40%", cls: "teal" },
    { lbl: "来年度 再公告予測(勝てそうなネタ)", val: num(k.upcoming_count), unit: "件", sub: "年度再帰×類似案件学習による先読み", cls: "" },
    { lbl: "来年度 予測期待受注額", val: oku(k.upcoming_expected_amount), unit: "億円", sub: "予想金額×勝てる期待値の合計", cls: "" },
  ];
  document.getElementById("kpiGrid").innerHTML = kpis.map(x =>
    `<div class="kpi ${x.cls}"><div class="lbl">${x.lbl}</div>
     <div class="val">${x.val}<small>${x.unit}</small></div><div class="sub">${x.sub}</div></div>`).join("");

  /* ---------- insights ---------- */
  document.getElementById("insightList").innerHTML = D.insights.map(t => {
    const warn = t.startsWith("【注意】") ? " class='warn'" : "";
    return `<li${warn}>${esc(t)}</li>`;
  }).join("");

  /* ---------- SAM box ---------- */
  const captureRate = k.di_sam_amount ? (k.di_expected_amount / k.di_sam_amount * 100).toFixed(1) : 0;
  document.getElementById("samBox").innerHTML =
    `<div style="display:flex;flex-direction:column;gap:12px;margin-top:6px">
      <div><div class="mini">射程案件(親和性40以上)</div>
        <div style="font-size:22px;font-weight:800">${num(k.di_pool_count)}<small style="font-size:12px">件</small></div></div>
      <div><div class="mini">SAM（獲得可能市場）</div>
        <div style="font-size:22px;font-weight:800;color:#2b5ce6">${oku(k.di_sam_amount)}<small style="font-size:12px">億円</small></div></div>
      <div><div class="mini">勝率加重の期待受注額</div>
        <div style="font-size:22px;font-weight:800;color:#00b3a4">${oku(k.di_expected_amount)}<small style="font-size:12px">億円</small></div>
        <div class="mini">実効獲得率 約${captureRate}%</div></div>
    </div>`;

  /* ---------- charts ---------- */
  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  Chart.defaults.font.size = 11;
  Chart.defaults.color = "#6b7686";

  const donut = (id, obj, colors) => new Chart(document.getElementById(id), {
    type: "doughnut",
    data: { labels: Object.keys(obj), datasets: [{ data: Object.values(obj), backgroundColor: colors, borderWidth: 2, borderColor: "#fff" }] },
    options: { cutout: "58%", plugins: { legend: { position: "bottom", labels: { boxWidth: 10, padding: 8 } } } }
  });
  donut("affChart", D.aff_bins, ["#1a9d63", "#f0a500", "#e07b39", "#b9c0cc"]);
  donut("winChart", D.win_bins, ["#1a9d63", "#f0a500", "#e07b39", "#b9c0cc"]);

  // 業種別 金額シェア (横棒)
  const indSorted = [...D.by_industry].sort((a, b) => b.amount - a.amount);
  new Chart(document.getElementById("indAmtChart"), {
    type: "bar",
    data: {
      labels: indSorted.map(x => x.key),
      datasets: [{ label: "落札金額(億円)", data: indSorted.map(x => +oku(x.amount)),
        backgroundColor: indSorted.map(x => IND_COLOR[x.key]) }]
    },
    options: {
      indexAxis: "y", plugins: { legend: { display: false },
        tooltip: { callbacks: { label: c => `${c.parsed.x}億円 / ${indSorted[c.dataIndex].count}件 / シェア${indSorted[c.dataIndex].amount_share}%` } } },
      scales: { x: { title: { display: true, text: "落札金額(億円)" } } }
    }
  });

  // 業種別 親和性×勝率 バブル
  new Chart(document.getElementById("indScatter"), {
    type: "bubble",
    data: {
      datasets: D.by_industry.map(x => ({
        label: x.key,
        data: [{ x: x.avg_affinity, y: x.avg_win, r: Math.max(5, Math.sqrt(x.count) * 1.6) }],
        backgroundColor: (IND_COLOR[x.key] || "#aab3c0") + "cc"
      }))
    },
    options: {
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: c => { const d = D.by_industry[c.datasetIndex]; return `${d.key}: 親和${d.avg_affinity}/勝率${d.avg_win}%/${d.count}件`; } } }
      },
      scales: {
        x: { title: { display: true, text: "平均DI親和性 →" }, min: 0, max: 100 },
        y: { title: { display: true, text: "平均勝率(%) →" }, min: 0, max: 100 }
      }
    }
  });

  // 金額規模 histogram (棒+折れ線)
  new Chart(document.getElementById("sizeChart"), {
    data: {
      labels: D.amount_hist.map(x => x.label),
      datasets: [
        { type: "bar", label: "件数", data: D.amount_hist.map(x => x.count), backgroundColor: "#2b5ce6", yAxisID: "y" },
        { type: "line", label: "金額合計(億円)", data: D.amount_hist.map(x => +oku(x.amount)), borderColor: "#f5a623", backgroundColor: "#f5a623", tension: .3, yAxisID: "y1" }
      ]
    },
    options: {
      plugins: { legend: { position: "bottom", labels: { boxWidth: 10 } } },
      scales: { y: { position: "left", title: { display: true, text: "件数" } },
        y1: { position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "億円" } } }
    }
  });

  // 省庁 Top12
  new Chart(document.getElementById("miniChart"), {
    type: "bar",
    data: {
      labels: D.by_ministry.map(x => x.key),
      datasets: [{ label: "件数", data: D.by_ministry.map(x => x.count), backgroundColor: "#00b3a4" }]
    },
    options: { indexAxis: "y", plugins: { legend: { display: false },
      tooltip: { callbacks: { label: c => `${c.parsed.x}件 / ${oku(D.by_ministry[c.dataIndex].amount)}億円` } } } }
  });

  /* ---------- industry table (sortable) ---------- */
  const maxIndAmt = Math.max(...D.by_industry.map(x => x.amount));
  function renderIndTable(rows) {
    const head = `<thead><tr>
      <th data-k="key">業種</th><th class="num" data-k="count">件数</th>
      <th class="num" data-k="count_share">件数%</th><th class="num" data-k="amount">落札金額</th>
      <th class="num" data-k="amount_share">金額%</th><th class="num" data-k="avg_affinity">平均親和性</th>
      <th class="num" data-k="avg_win">平均勝率</th></tr></thead>`;
    const body = rows.map(x => `<tr>
      <td>${indChip(x.key)}</td>
      <td class="num">${num(x.count)}</td>
      <td class="num">${x.count_share}%</td>
      <td class="num bar-cell"><span class="bar" style="width:${x.amount / maxIndAmt * 100}%;background:${IND_COLOR[x.key]}"></span><span class="txt">${yen(x.amount)}</span></td>
      <td class="num">${x.amount_share}%</td>
      <td class="num">${scoreBar(x.avg_affinity, "#2b5ce6")}</td>
      <td class="num">${scoreBar(x.avg_win, "#00b3a4")}</td></tr>`).join("");
    document.getElementById("indTable").innerHTML = head + "<tbody>" + body + "</tbody>";
  }
  makeSortable("indTable", D.by_industry, renderIndTable, "amount");

  /* ---------- competition level helpers ---------- */
  const CL_COLOR = {
    "参入容易": "#1a9d63", "競争可能": "#2b9d8f", "要準備": "#f0a500",
    "参入困難": "#e07b39", "対象外": "#b9c0cc", "対象外(領域外)": "#b9c0cc",
  };
  const clChip = (lv, score) =>
    `<span class="chip" style="background:${CL_COLOR[lv] || "#b9c0cc"}" title="参入度スコア ${score ?? "—"}/100">${lv}${score != null ? " " + score : ""}</span>`;

  /* ---------- company ranking ---------- */
  const maxCompAmt = Math.max(...D.top_companies.map(x => x.amount));
  document.getElementById("compTable").innerHTML =
    `<thead><tr><th>#</th><th>落札企業</th><th>タイプ</th><th class="num">落札額合計</th><th class="num">件数</th><th>対DI競合度</th><th class="num">参入係数</th></tr></thead><tbody>` +
    D.top_companies.map((x, i) => `<tr><td class="num">${i + 1}</td>
        <td>${esc(x.company)}</td>
        <td><span class="mini">${esc(x.ilabel)}</span></td>
        <td class="num bar-cell"><span class="bar" style="width:${x.amount / maxCompAmt * 100}%;background:#5a6acf"></span><span class="txt">${yen(x.amount)}</span></td>
        <td class="num">${x.count}</td>
        <td>${clChip(x.comp_level)}</td>
        <td class="num" title="現職タイプ係数(1.0=正面勝負可能)">${x.entry.toFixed(2)}</td></tr>`).join("") +
    `</tbody>`;

  /* ---------- competition logic section ---------- */
  const M = D.methodology;
  document.getElementById("compFormula").innerHTML =
    `<b style="color:var(--ink)">${esc(M.competition.formula)}</b><br>` +
    `勝率の算出は「${esc(M.winnability.formula)}」。${esc(M.winnability.desc)}`;
  document.getElementById("compLevelTable").innerHTML =
    `<thead><tr><th>レベル</th><th>スコア帯</th><th>意味</th><th>DIの取るべきアクション</th></tr></thead><tbody>` +
    M.competition.levels.map(l => `<tr>
      <td>${clChip(l.level)}</td><td class="num">${l.range}</td>
      <td style="font-size:12px">${esc(l.meaning)}</td>
      <td style="font-size:12px;color:#4b5563">${esc(l.action)}</td></tr>`).join("") + "</tbody>";
  document.getElementById("incumbentTable").innerHTML =
    `<thead><tr><th>現落札企業のタイプ</th><th class="num">係数</th><th>判定理由</th></tr></thead><tbody>` +
    M.competition.incumbents.map(x => `<tr>
      <td style="white-space:nowrap;font-weight:600">${esc(x.type)}</td>
      <td class="num">${scoreBar(x.coef * 100, x.coef >= 0.7 ? "#1a9d63" : x.coef >= 0.4 ? "#f0a500" : "#e07b39")}</td>
      <td style="font-size:12px;color:#4b5563">${esc(x.desc)}</td></tr>`).join("") + "</tbody>";
  document.getElementById("scaleTable").innerHTML =
    `<thead><tr><th>落札金額帯</th><th class="num">係数</th><th>障壁の内容</th></tr></thead><tbody>` +
    M.competition.scale.map(x => `<tr>
      <td style="white-space:nowrap;font-weight:600">${esc(x.band)}</td>
      <td class="num">${scoreBar(x.coef * 100, x.coef >= 0.7 ? "#1a9d63" : x.coef >= 0.4 ? "#f0a500" : "#e07b39")}</td>
      <td style="font-size:12px;color:#4b5563">${esc(x.note)}</td></tr>`).join("") + "</tbody>";
  document.getElementById("footholdCommon").innerHTML =
    M.competition.foothold_common.map(t => `<li>✅ ${esc(t)}</li>`).join("");

  // 競合度レベル別分布(件数=棒, 金額=第2軸)
  new Chart(document.getElementById("compDistChart"), {
    data: {
      labels: D.comp_level_dist.map(x => x.level),
      datasets: [
        { type: "bar", label: "件数", data: D.comp_level_dist.map(x => x.count),
          backgroundColor: D.comp_level_dist.map(x => CL_COLOR[x.level]), yAxisID: "y" },
        { type: "line", label: "金額(億円)", data: D.comp_level_dist.map(x => +oku(x.amount)),
          borderColor: "#5a6acf", backgroundColor: "#5a6acf", tension: .3, yAxisID: "y1" },
      ]
    },
    options: {
      plugins: { legend: { position: "bottom", labels: { boxWidth: 10 } } },
      scales: { y: { title: { display: true, text: "件数" } },
        y1: { position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "億円" } } }
    }
  });

  /* ---------- opportunity score explainer ---------- */
  const OPP_COLORS = { win: "#00b3a4", size: "#f5a623", aff: "#2b5ce6" };
  const OPP_NAMES = { win: "落札可能性×0.5", size: "規模の魅力×0.2", aff: "親和性×0.3" };
  document.getElementById("oppFormula").innerHTML =
    `<b style="color:var(--ink)">${esc(M.opportunity.formula)}</b><br>読み方: ${esc(M.opportunity.reading)}`;
  document.getElementById("oppComponents").innerHTML = M.opportunity.components.map(c =>
    `<div class="opp-card"><b>${esc(c.name)}</b><span style="color:#4b5563">${esc(c.desc)}</span></div>`).join("");
  document.getElementById("oppLegend").innerHTML =
    Object.keys(OPP_COLORS).map(k => `<span><i style="background:${OPP_COLORS[k]}"></i>${OPP_NAMES[k]}</span>`).join("") +
    `<span class="mini">— 表の「狙い目」列の色帯が3成分の内訳</span>`;

  const oppStack = (t) => {
    const p = t.opp_parts || { win: 0, size: 0, aff: 0 };
    return `<div class="oppstack" title="勝率成分${p.win} + 規模成分${p.size} + 親和成分${p.aff} = ${t.opportunity}">` +
      ["win", "size", "aff"].map(k => `<i style="width:${p[k]}%;background:${OPP_COLORS[k]}"></i>`).join("") + `</div>`;
  };

  const detailBox = (t) => {
    const sims = (t.similar_cases || []).map(s =>
      `<li class="sim">類似度${s.sim}%: ${esc(s.name)}（${esc(s.company)}・${yen(s.amount)}・勝率${s.win}%）</li>`).join("");
    return `<details class="rowdetail"><summary>詳細（競合評価・土台条件・類似案件）</summary>
      <div class="dbox">
        <div><h5>⚔️ 競合評価 ${clChip(t.comp_level, t.comp_score)}</h5>
          <div style="color:#4b5563">${esc(t.comp_reason)}</div></div>
        <div><h5>🏁 競争の土台に載る条件（この案件の場合）</h5>
          <ul>${(t.foothold || []).map(f => `<li>${esc(f)}</li>`).join("")}</ul></div>
        ${t.pred_win != null ? `<div><h5>🔮 類似案件学習による予測勝率: ${t.pred_win}%（確信度${t.pred_conf_label}）</h5>
          <ul>${sims || "<li class='sim'>十分に類似する案件なし（ルールベース値を採用）</li>"}</ul></div>` : ""}
      </div></details>`;
  };

  /* ---------- DI targets ---------- */
  function targetRows(rows) {
    return rows.map((t, i) => `<tr>
      <td class="num">${i + 1}</td>
      <td>${indChip(t.industry)}</td>
      <td><div style="font-weight:600">${esc(t.project_name)} <a class="simlink" href="strategy.html?id=${t.id}" title="入札戦略シミュレータで勝率改善を試算">🎯</a></div>
          <div class="mini">${esc(t.ministry)}${t.agency ? " ・" + esc(t.agency) : ""} ／ 現落札: ${esc(t.company)}</div>
          ${detailBox(t)}</td>
      <td class="num">${yen(t.amount)}</td>
      <td class="num">${scoreBar(t.affinity, "#2b5ce6")} ${lab(t.affinity_label)}</td>
      <td class="num">${scoreBar(t.winnability, "#00b3a4")} ${lab(t.winnability_label)}<br>
          <span class="mini" title="類似案件学習(k近傍)による予測">予測 ${t.pred_win != null ? t.pred_win + "%" : "—"}</span></td>
      <td>${clChip(t.comp_level, t.comp_score)}</td>
      <td class="num"><b style="font-size:14px">${t.opportunity}</b><br>${oppStack(t)}</td></tr>`).join("");
  }
  document.getElementById("targetTable").innerHTML =
    `<thead><tr><th>#</th><th>業種</th><th>案件名 / 発注元・現落札 / 詳細</th><th class="num">落札金額</th>
      <th class="num">DI親和性</th><th class="num">落札可能性</th><th>競合度</th><th class="num">狙い目</th></tr></thead>
     <tbody>${targetRows(D.di_targets)}</tbody>`;

  /* ---------- forecast (今後の勝てそうなネタ) ---------- */
  document.getElementById("predNote").textContent = M.prediction.note;
  document.getElementById("predMethod").innerHTML = [
    { t: "予測勝率", d: M.prediction.pred_win },
    { t: "再公告確率", d: M.prediction.recurrence },
    { t: "勝てる期待値", d: M.prediction.expected },
  ].map(x => `<div class="opp-card"><b>${esc(x.t)}</b><span style="color:#4b5563">${esc(x.d)}</span></div>`).join("");

  document.getElementById("forecastTable").innerHTML =
    `<thead><tr><th>#</th><th>業種</th><th>予測案件（来年度の再公告見込み）</th><th>公告予測時期</th>
      <th class="num">予想金額</th><th class="num">再公告確率</th><th class="num">予測勝率</th><th>競合度</th><th class="num">勝てる期待値</th></tr></thead><tbody>` +
    D.upcoming.map((u, i) => `<tr>
      <td class="num">${i + 1}</td>
      <td>${indChip(u.industry)}</td>
      <td><div style="font-weight:600">${esc(u.pred_name)}</div>
        <div class="mini">${esc(u.ministry)}${u.agency ? " ・" + esc(u.agency) : ""}</div>
        <details class="rowdetail"><summary>予測根拠</summary>
          <div class="dbox"><div style="color:#4b5563">${esc(u.rationale)}</div>
          <div class="sim">前年実績: ${esc(u.base_name)}（${esc(u.base_company)}・${yen(u.base_amount)}）</div></div>
        </details></td>
      <td style="font-size:11.5px;white-space:nowrap">${esc(u.announce_window)}</td>
      <td class="num">${yen(u.pred_amount)}</td>
      <td class="num">${u.recurrence_prob}%</td>
      <td class="num">${u.pred_win}% <span class="mini">確信${u.pred_conf_label}</span></td>
      <td>${clChip(u.comp_level)}</td>
      <td class="num"><b style="font-size:14px;color:${u.expected >= 50 ? "#1a9d63" : u.expected >= 30 ? "#f0a500" : "#6b7686"}">${u.expected}%</b></td></tr>`).join("") +
    "</tbody>";

  /* ---------- explorer ---------- */
  const selInd = document.getElementById("fIndustry");
  D.industry_order.forEach(n => { const o = document.createElement("option"); o.value = n; o.textContent = n; selInd.appendChild(o); });
  const state = { q: "", ind: "", aff: 0, sort: "opportunity", diOnly: true };
  const els = { q: document.getElementById("q"), ind: selInd, aff: document.getElementById("fAff"),
    sort: document.getElementById("fSort"), diOnly: document.getElementById("pDiOnly"),
    count: document.getElementById("explCount"), table: document.getElementById("explTable") };

  function renderExplorer() {
    let rows = D.records.filter(r => {
      if (state.diOnly && r.affinity < 40) return false;
      if (r.affinity < state.aff) return false;
      if (state.ind && r.industry !== state.ind) return false;
      if (state.q) { const q = state.q.toLowerCase();
        if (!(r.project_name.toLowerCase().includes(q) || r.company.toLowerCase().includes(q))) return false; }
      return true;
    });
    rows.sort((a, b) => (b[state.sort] || 0) - (a[state.sort] || 0));
    els.count.textContent = `${rows.length.toLocaleString()} 件表示`;
    const view = rows.slice(0, 300);
    els.table.innerHTML =
      `<thead><tr><th>業種</th><th>案件名</th><th class="num">金額</th>
        <th class="num">親和性</th><th class="num">勝率</th><th>競合度</th><th>案件概要（DI視点）</th></tr></thead><tbody>` +
      view.map(r => `<tr>
        <td>${indChip(r.industry)}</td>
        <td><div style="font-weight:600;max-width:280px">${esc(r.project_name)} <a class="simlink" href="strategy.html?id=${r.id}" title="入札戦略シミュレータで勝率改善を試算">🎯</a></div>
            <div class="mini">${esc(r.ministry)} ／ ${esc(r.company)}</div></td>
        <td class="num">${yen(r.amount)}</td>
        <td class="num">${scoreBar(r.affinity, "#2b5ce6")}<br>${lab(r.affinity_label)}</td>
        <td class="num">${scoreBar(r.winnability, "#00b3a4")}<br>${lab(r.winnability_label)}</td>
        <td>${clChip(r.comp_level, r.comp_score)}</td>
        <td style="max-width:340px;font-size:11.5px;color:#4b5563">${esc(r.summary)}
          <details class="rowdetail"><summary>競合評価・土台条件</summary>
            <div class="dbox"><div style="color:#4b5563">${esc(r.comp_reason)}</div>
            <ul>${(r.foothold || []).map(f => `<li>${esc(f)}</li>`).join("")}</ul></div>
          </details></td></tr>`).join("") +
      "</tbody>";
    if (rows.length > 300) els.table.insertAdjacentHTML("beforeend",
      `<tfoot><tr><td colspan="7" class="mini" style="text-align:center">上位300件を表示（該当${rows.length}件）— さらに絞り込んでください</td></tr></tfoot>`);
  }
  els.q.oninput = e => { state.q = e.target.value; renderExplorer(); };
  els.ind.onchange = e => { state.ind = e.target.value; renderExplorer(); };
  els.aff.onchange = e => { state.aff = +e.target.value; renderExplorer(); };
  els.sort.onchange = e => { state.sort = e.target.value; renderExplorer(); };
  els.diOnly.onclick = () => { state.diOnly = !state.diOnly; els.diOnly.classList.toggle("on", state.diOnly); renderExplorer(); };
  renderExplorer();

  /* ---------- sortable helper ---------- */
  function makeSortable(id, data, render, defKey) {
    let key = defKey, asc = false;
    const sort = () => [...data].sort((a, b) => {
      const va = a[key], vb = b[key];
      const r = typeof va === "string" ? va.localeCompare(vb) : va - vb;
      return asc ? r : -r;
    });
    render(sort());
    document.getElementById(id).addEventListener("click", e => {
      const th = e.target.closest("th[data-k]"); if (!th) return;
      const kk = th.dataset.k; if (kk === key) asc = !asc; else { key = kk; asc = false; }
      render(sort());
    });
  }

  /* ---------- nav scrollspy ---------- */
  const navLinks = [...document.querySelectorAll("#nav a")];
  const secs = navLinks.map(a => document.querySelector(a.getAttribute("href")));
  const obs = new IntersectionObserver(ents => {
    ents.forEach(en => { if (en.isIntersecting) {
      navLinks.forEach(a => a.classList.toggle("active", a.getAttribute("href") === "#" + en.target.id));
    } });
  }, { rootMargin: "-20% 0px -70% 0px" });
  secs.forEach(s => s && obs.observe(s));
})();
